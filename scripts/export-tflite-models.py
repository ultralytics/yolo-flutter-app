# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Export official YOLO26 LiteRT (.tflite) assets for the Flutter Android release.

Uses the Ultralytics `format=litert` export (litert-torch + ai-edge-quantizer) with `quantize="w8a32"` (dynamic INT8:
int8 weights + FP32 activations), which requires `ultralytics>=8.4.83` and runs on Linux x86 or macOS with Python>=3.10.
w8a32 needs no calibration data, compiles on the LiteRT GPU delegate, and is the smallest of the GPU-capable formats.

Usage from the repository root:

    uv venv --python 3.12 .venv
    uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
    uv pip install "ultralytics-opencv-headless[export-litert]>=8.4.83"
    uv run python scripts/export-tflite-models.py --verify
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "exports" / "yolo26-tflite"
DEFAULT_REPO = "ultralytics/yolo-flutter-app"
DEFAULT_TAG = "models-v1.0.0"
QUANTIZE = "w8a32"
SIZES = ("n", "s", "m", "l", "x")


@dataclass(frozen=True)
class TaskSpec:
    """Export settings for one YOLO task family."""

    suffix: str
    imgsz: int


TASKS: dict[str, TaskSpec] = {
    "detect": TaskSpec("", 640),
    "segment": TaskSpec("-seg", 640),
    "semantic": TaskSpec("-sem", 640),
    "depth": TaskSpec("-depth", 640),
    "classify": TaskSpec("-cls", 224),
    "pose": TaskSpec("-pose", 640),
    "obb": TaskSpec("-obb", 640),
}

_TASK_NAMES_CACHE: dict[str, dict[int, str]] = {}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=list(SIZES))
    parser.add_argument("--tasks", nargs="+", choices=TASKS.keys(), default=list(TASKS))
    parser.add_argument("--verify", action="store_true", help="Run one zero-input TFLite inference per exported file.")
    parser.add_argument("--force", action="store_true", help="Re-export assets that already exist in release-assets/.")
    parser.add_argument("--worker-model-id", help=argparse.SUPPRESS)
    parser.add_argument("--worker-imgsz", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload generated .tflite files to a new GitHub release.",
    )
    return parser.parse_args()


def verify_tflite(path: Path, imgsz: int) -> list[tuple[int, ...]]:
    """Verify the fixed input size, run one zero-input inference, and return output shapes."""
    # ai_edge_litert ships with ultralytics[export-litert]; TensorFlow is not installed in that environment
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    shape = input_detail["shape"]
    if list(shape).count(imgsz) != 2:
        raise ValueError(f"{path.name} input is {shape.tolist()}; expected two {imgsz}-pixel spatial dimensions")
    dtype = input_detail["dtype"]
    sample = np.zeros(shape, dtype=dtype)
    if not np.issubdtype(dtype, np.floating):
        sample.fill(input_detail.get("quantization", (0.0, 0))[1])
    interpreter.set_tensor(input_detail["index"], sample)
    interpreter.invoke()
    return [tuple(detail["shape"].tolist()) for detail in interpreter.get_output_details()]


def tflite_metadata(path: Path) -> dict | None:
    """Return embedded Ultralytics TFLite metadata when present."""
    try:
        with zipfile.ZipFile(path) as zf:
            infos = [
                info for info in zf.infolist() if info.filename in {"metadata.json", "TFLITE_ULTRALYTICS_METADATA.json"}
            ]
            if infos:
                return json.loads(zf.read(infos[-1]))
    except Exception:
        return None
    return None


def task_names(task_name: str, suffix: str) -> dict[int, str]:
    """Return class names for the representative model in a task family."""
    if task_name not in _TASK_NAMES_CACHE:
        from ultralytics import YOLO

        weights_name = f"yolo26m{suffix}.pt"
        model = YOLO(str(ROOT / weights_name) if (ROOT / weights_name).exists() else weights_name)
        _TASK_NAMES_CACHE[task_name] = {int(k): str(v) for k, v in model.names.items()}
    return _TASK_NAMES_CACHE[task_name]


def append_tflite_metadata(path: Path, model_id: str, task_name: str, task: TaskSpec) -> None:
    """Append Ultralytics metadata to a TFLite model archive."""
    from ultralytics import __version__ as ultralytics_version

    metadata = {
        "description": f"Ultralytics {model_id} w8a32 LiteRT model",
        "author": "Ultralytics",
        "date": time.strftime("%Y-%m-%d"),
        "version": ultralytics_version,
        "task": task_name,
        "batch": 1,
        "imgsz": [task.imgsz, task.imgsz],
        "names": task_names(task_name, task.suffix),
        "channels": 3,
        "stride": 32,
        "format": "litert",
        "int8": False,
        "nms": False,
        "end2end": False,
    }
    with zipfile.ZipFile(path, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metadata.json", json.dumps(metadata, separators=(",", ":")))


def ensure_tflite_metadata(path: Path, model_id: str, task_name: str, task: TaskSpec) -> None:
    """Add TFLite metadata when it is missing or incomplete."""
    metadata = tflite_metadata(path)
    names = (metadata or {}).get("names")
    # The ultralytics LiteRT export already embeds a metadata.json with the fields the app reads (task, imgsz, names,
    # end2end, ...); only re-append when one of those is missing/wrong, so a complete export is left single-entry.
    if (
        metadata is None
        or metadata.get("task") != task_name
        or metadata.get("end2end") is not False
        or metadata.get("imgsz") != [task.imgsz, task.imgsz]
        or not isinstance(names, dict)
        or not names
    ):
        append_tflite_metadata(path, model_id, task_name, task)


def upload_assets(repo: str, tag: str, assets: list[Path]) -> None:
    """Upload generated assets to a GitHub release."""
    if not assets:
        return
    command = ["gh", "release", "upload", tag, "--repo", repo, *(str(path) for path in assets)]
    subprocess.run(command, check=True)


def display_path(path: Path) -> str:
    """Return a repository-relative path when possible."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def export_one(model_id: str, imgsz: int, output_dir: Path) -> None:
    """Export one YOLO model to w8a32 LiteRT (no calibration data required)."""
    from ultralytics import YOLO

    os.chdir(output_dir)
    YOLO(str(output_dir / f"{model_id}.pt")).export(
        format="litert",
        quantize=QUANTIZE,
        nms=False,
        end2end=False,
        imgsz=imgsz,
        batch=1,
    )


def exported_tflite_path(output_dir: Path, model_id: str) -> Path | None:
    """Find the exported LiteRT .tflite for a model (single-file `<model_id>_w8a32.tflite`)."""
    candidates = (
        output_dir / f"{model_id}_{QUANTIZE}.tflite",
        Path("/ultralytics/weights") / f"{model_id}_{QUANTIZE}.tflite",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def run_export_worker(model_id: str, task: TaskSpec, output_dir: Path) -> Path:
    """Run a child export process (for memory isolation) and return the generated LiteRT .tflite path."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--output-dir",
        str(output_dir),
        "--worker-model-id",
        model_id,
        "--worker-imgsz",
        str(task.imgsz),
    ]
    returncode = subprocess.run(command).returncode
    exported = exported_tflite_path(output_dir, model_id)
    if exported is None:
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, command)
        raise FileNotFoundError(f"No LiteRT export found for {model_id}")
    if returncode != 0:
        print(f"worker for {model_id} exited {returncode}; using generated {exported.name}", flush=True)
    return exported


def main() -> None:
    """Export requested YOLO LiteRT release assets."""
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker_model_id:
        if args.worker_imgsz is None:
            raise ValueError("--worker-imgsz is required with --worker-model-id")
        export_one(args.worker_model_id, args.worker_imgsz, output_dir)
        return

    os.chdir(output_dir)
    release_dir = output_dir / "release-assets"
    release_dir.mkdir(parents=True, exist_ok=True)

    assets: list[Path] = []
    for task_name in args.tasks:
        task = TASKS[task_name]
        for size in args.sizes:
            model_id = f"yolo26{size}{task.suffix}"
            target = release_dir / f"{model_id}_{QUANTIZE}.tflite"
            if target.exists() and not args.force:
                ensure_tflite_metadata(target, model_id, task_name, task)
                outputs = verify_tflite(target, task.imgsz) if args.verify else []
                suffix = f" outputs={outputs}" if outputs else ""
                print(f"\nSkipping {model_id}; asset exists at {display_path(target)}{suffix}")
                assets.append(target)
                continue
            print(f"\nExporting {model_id} ({task_name}, imgsz={task.imgsz}, quantize={QUANTIZE})")
            exported = exported_tflite_path(output_dir, model_id)
            if exported is not None and not args.force:
                print(f"using existing generated {display_path(exported)}")
            else:
                exported = run_export_worker(model_id, task, output_dir)
            shutil.copy2(exported, target)
            ensure_tflite_metadata(target, model_id, task_name, task)
            outputs = verify_tflite(target, task.imgsz) if args.verify else []
            suffix = f" outputs={outputs}" if outputs else ""
            print(f"asset {display_path(target)} size={target.stat().st_size / 1_000_000:.2f} MB{suffix}")
            assets.append(target)

    if args.upload:
        upload_assets(args.repo, args.tag, assets)

    print(f"\nPrepared {len(assets)} LiteRT release assets in {release_dir}")


if __name__ == "__main__":
    main()
