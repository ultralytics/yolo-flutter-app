# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Export official YOLO26 TFLite assets for the Flutter Android release.

Run in a Linux Python 3.13 environment. The macOS Python 3.13+ TFLite path is
blocked by the ai-edge-litert macOS wheel.

Usage from the repository root:

    uv venv --python 3.13 .venv
    uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
    uv pip install -e "../ultralytics" "tensorflow>2.19.0" "onnx>=1.20.0" "onnxslim>=0.1.82" \
      "tf_keras>2.19.0" "sng4onnx>=1.0.1" "onnx_graphsurgeon>=0.3.26" \
      "ai-edge-litert>=1.2.0" "onnxruntime" "protobuf>=6.31.1,<7.0.0" \
      --extra-index-url https://pypi.ngc.nvidia.com --index-strategy unsafe-best-match
    uv pip uninstall opencv-python
    uv pip install opencv-python-headless
    uv pip install --no-deps "onnx2tf>=2.3.0,<2.3.16"
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
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "exports" / "yolo26-tflite"
DEFAULT_REPO = "ultralytics/yolo-flutter-app"
DEFAULT_TAG = "v0.3.5"
SIZES = ("n", "s", "m", "l", "x")


@dataclass(frozen=True)
class TaskSpec:
    suffix: str
    imgsz: int


TASKS: dict[str, TaskSpec] = {
    "detect": TaskSpec("", 640),
    "segment": TaskSpec("-seg", 640),
    "semantic": TaskSpec("-sem", 640),
    "classify": TaskSpec("-cls", 224),
    "pose": TaskSpec("-pose", 640),
    "obb": TaskSpec("-obb", 640),
}

_TASK_NAMES_CACHE: dict[str, dict[int, str]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument(
        "--data",
        help=(
            "Override calibration data for every task. By default the script uses "
            "ultralytics.cfg.TASK2CALIBRATIONDATA per task."
        ),
    )
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=list(SIZES))
    parser.add_argument("--tasks", nargs="+", choices=TASKS.keys(), default=list(TASKS))
    parser.add_argument("--verify", action="store_true", help="Run one zero-input TFLite inference per exported file.")
    parser.add_argument("--force", action="store_true", help="Re-export assets that already exist in release-assets/.")
    parser.add_argument("--worker-model-id", help=argparse.SUPPRESS)
    parser.add_argument("--worker-imgsz", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload generated .tflite files to the GitHub release with gh release upload --clobber.",
    )
    return parser.parse_args()


def verify_tflite(path: Path) -> list[tuple[int, ...]]:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    shape = input_detail["shape"]
    dtype = input_detail["dtype"]
    sample = np.zeros(shape, dtype=dtype)
    if not np.issubdtype(dtype, np.floating):
        sample.fill(input_detail.get("quantization", (0.0, 0))[1])
    interpreter.set_tensor(input_detail["index"], sample)
    interpreter.invoke()
    return [tuple(detail["shape"].tolist()) for detail in interpreter.get_output_details()]


def tflite_metadata(path: Path) -> dict | None:
    try:
        with zipfile.ZipFile(path) as zf:
            infos = [
                info
                for info in zf.infolist()
                if info.filename in {"metadata.json", "TFLITE_ULTRALYTICS_METADATA.json"}
            ]
            if infos:
                return json.loads(zf.read(infos[-1]))
    except Exception:
        return None
    return None


def task_names(task_name: str, suffix: str) -> dict[int, str]:
    if task_name not in _TASK_NAMES_CACHE:
        from ultralytics import YOLO

        weights_name = f"yolo26m{suffix}.pt"
        model = YOLO(str(ROOT / weights_name) if (ROOT / weights_name).exists() else weights_name)
        _TASK_NAMES_CACHE[task_name] = {int(k): str(v) for k, v in model.names.items()}
    return _TASK_NAMES_CACHE[task_name]


def append_tflite_metadata(path: Path, model_id: str, task_name: str, task: TaskSpec) -> None:
    metadata = {
        "description": f"Ultralytics {model_id} int8 TFLite model",
        "author": "Ultralytics",
        "date": time.strftime("%Y-%m-%d"),
        "version": "8.4.0",
        "task": task_name,
        "batch": 1,
        "imgsz": [task.imgsz, task.imgsz],
        "names": task_names(task_name, task.suffix),
        "channels": 3,
        "stride": 32,
        "format": "tflite",
        "int8": True,
        "nms": False,
        "end2end": False,
    }
    with zipfile.ZipFile(path, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metadata.json", json.dumps(metadata, separators=(",", ":")))


def ensure_tflite_metadata(path: Path, model_id: str, task_name: str, task: TaskSpec) -> None:
    metadata = tflite_metadata(path)
    names = (metadata or {}).get("names")
    if (
        metadata is None
        or metadata.get("task") != task_name
        or metadata.get("int8") is not True
        or metadata.get("nms") is not False
        or metadata.get("end2end") is not False
        or metadata.get("imgsz") != [task.imgsz, task.imgsz]
        or not isinstance(names, dict)
        or not names
    ):
        append_tflite_metadata(path, model_id, task_name, task)


def upload_assets(repo: str, tag: str, assets: list[Path]) -> None:
    if not assets:
        return
    command = ["gh", "release", "upload", tag, "--repo", repo, "--clobber", *(str(path) for path in assets)]
    subprocess.run(command, check=True)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def export_one(model_id: str, imgsz: int, data: str, output_dir: Path) -> None:
    from ultralytics import YOLO

    os.chdir(output_dir)
    YOLO(f"{model_id}.pt").export(
        format="tflite",
        int8=True,
        data=data,
        nms=False,
        end2end=False,
        imgsz=imgsz,
        batch=1,
    )


def ensure_coco128_dataset() -> Path:
    dataset_root = Path("/datasets/coco128")
    images_dir = dataset_root / "images" / "train2017"
    if images_dir.is_dir():
        return dataset_root
    datasets_dir = Path("/datasets")
    datasets_dir.mkdir(parents=True, exist_ok=True)
    archive = datasets_dir / "coco128.zip"
    if not archive.exists():
        urllib.request.urlretrieve("https://ultralytics.com/assets/coco128.zip", archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(datasets_dir)
    return dataset_root


def classify_calibration_data(data: str, output_dir: Path) -> str:
    source = Path(data)
    if source.is_dir():
        return str(source.resolve())
    if data != "coco128.yaml":
        return data

    coco128 = ensure_coco128_dataset()
    images = sorted((coco128 / "images" / "train2017").glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No coco128 calibration images found under {coco128}")

    cls_root = output_dir / "coco128-cls-calibration"
    train_dir = cls_root / "train" / "coco128"
    val_dir = cls_root / "val" / "coco128"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for i, image in enumerate(images):
        target_dir = val_dir if i % 5 == 0 else train_dir
        target = target_dir / image.name
        if target.exists():
            continue
        try:
            target.symlink_to(image)
        except OSError:
            shutil.copy2(image, target)

    return str(cls_root)


def image_only_calibration_data(data: str, output_dir: Path, task_name: str) -> str:
    source = Path(data)
    if source.is_file():
        if data != "coco128.yaml":
            return data
    elif source.exists():
        return str(source.resolve())

    coco128 = ensure_coco128_dataset()
    images = sorted((coco128 / "images" / "train2017").glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No coco128 calibration images found under {coco128}")

    calib_root = output_dir / f"coco128-{task_name}-calibration"
    image_dir = calib_root / "images" / "val"
    image_dir.mkdir(parents=True, exist_ok=True)
    for image in images:
        target = image_dir / image.name
        if target.exists():
            continue
        try:
            target.symlink_to(image)
        except OSError:
            shutil.copy2(image, target)

    yaml_lines = [
        f"path: {calib_root}",
        "train: images/val",
        "val: images/val",
        "names:",
        "  0: coco128",
    ]
    if task_name == "pose":
        yaml_lines.append("kpt_shape: [17, 3]")
    yaml_path = calib_root / "data.yaml"
    yaml_path.write_text("\n".join(yaml_lines) + "\n")
    return str(yaml_path)


def calibration_data(task_name: str, data: str | None, output_dir: Path) -> str:
    if data is None:
        from ultralytics.cfg import TASK2CALIBRATIONDATA

        return TASK2CALIBRATIONDATA[task_name]
    if task_name == "classify":
        return classify_calibration_data(data, output_dir)
    if task_name in {"pose", "obb"}:
        return image_only_calibration_data(data, output_dir, task_name)
    return data


def exported_tflite_path(output_dir: Path, model_id: str) -> Path | None:
    saved_model_dirs = [
        output_dir / f"{model_id}_saved_model",
        Path("/ultralytics/weights") / f"{model_id}_saved_model",
    ]
    for saved_model_dir in saved_model_dirs:
        candidates = (
            saved_model_dir / f"{model_id}_int8.tflite",
            saved_model_dir / f"{model_id}_dynamic_range_quant.tflite",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def wait_for_stable_file(path: Path, checks: int = 2, interval: float = 1.0) -> bool:
    last_size = -1
    stable_checks = 0
    while path.exists():
        size = path.stat().st_size
        if size == last_size and size > 0:
            stable_checks += 1
            if stable_checks >= checks:
                return True
        else:
            stable_checks = 0
            last_size = size
        time.sleep(interval)
    return False


def stop_process(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_export_worker(
    model_id: str,
    task: TaskSpec,
    args: argparse.Namespace,
    output_dir: Path,
    data: str,
) -> Path:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--output-dir",
        str(output_dir),
        "--data",
        data,
        "--worker-model-id",
        model_id,
        "--worker-imgsz",
        str(task.imgsz),
    ]
    process = subprocess.Popen(command)
    dynamic_tflite = output_dir / f"{model_id}_saved_model" / f"{model_id}_dynamic_range_quant.tflite"
    while process.poll() is None:
        if dynamic_tflite.exists() and wait_for_stable_file(dynamic_tflite):
            print(f"dynamic-range TFLite ready for {model_id}; stopping discarded full-integer conversions", flush=True)
            stop_process(process)
            break
        time.sleep(2)

    returncode = process.wait()
    exported = exported_tflite_path(output_dir, model_id)
    if exported is None:
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, command)
        raise FileNotFoundError(f"No TFLite export found for {model_id}")
    if returncode != 0:
        print(f"worker for {model_id} exited {returncode}; using generated {exported.name}", flush=True)
    return exported


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker_model_id:
        if args.worker_imgsz is None:
            raise ValueError("--worker-imgsz is required with --worker-model-id")
        export_one(args.worker_model_id, args.worker_imgsz, args.data, output_dir)
        return

    os.chdir(output_dir)
    release_dir = output_dir / "release-assets"
    release_dir.mkdir(parents=True, exist_ok=True)

    assets: list[Path] = []
    for task_name in args.tasks:
        task = TASKS[task_name]
        for size in args.sizes:
            model_id = f"yolo26{size}{task.suffix}"
            target = release_dir / f"{model_id}_int8.tflite"
            if target.exists() and not args.force:
                ensure_tflite_metadata(target, model_id, task_name, task)
                outputs = verify_tflite(target) if args.verify else []
                suffix = f" outputs={outputs}" if outputs else ""
                print(f"\nSkipping {model_id}; asset exists at {target.relative_to(ROOT)}{suffix}")
                assets.append(target)
                continue
            export_data = calibration_data(task_name, args.data, output_dir)
            print(f"\nExporting {model_id} ({task_name}, imgsz={task.imgsz}, data={export_data})")
            if args.data and export_data != args.data:
                print(f"using {task_name} calibration data from {args.data}: {export_data}")
            exported = exported_tflite_path(output_dir, model_id)
            if exported is not None and not args.force:
                print(f"using existing generated {display_path(exported)}")
            else:
                exported = run_export_worker(model_id, task, args, output_dir, export_data)
            shutil.copy2(exported, target)
            ensure_tflite_metadata(target, model_id, task_name, task)
            outputs = verify_tflite(target) if args.verify else []
            suffix = f" outputs={outputs}" if outputs else ""
            print(f"asset {target.relative_to(ROOT)} size={target.stat().st_size / 1_000_000:.2f} MB{suffix}")
            assets.append(target)

    if args.upload:
        upload_assets(args.repo, args.tag, assets)

    print(f"\nPrepared {len(assets)} TFLite release assets in {release_dir}")


if __name__ == "__main__":
    main()
