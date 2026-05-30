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
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--data", default="coco128.yaml", help="Calibration data used for all int8 exports.")
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=list(SIZES))
    parser.add_argument("--tasks", nargs="+", choices=TASKS.keys(), default=list(TASKS))
    parser.add_argument("--verify", action="store_true", help="Run one zero-input TFLite inference per exported file.")
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


def upload_assets(repo: str, tag: str, assets: list[Path]) -> None:
    if not assets:
        return
    command = ["gh", "release", "upload", tag, "--repo", repo, "--clobber", *(str(path) for path in assets)]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)
    release_dir = output_dir / "release-assets"
    release_dir.mkdir(parents=True, exist_ok=True)

    assets: list[Path] = []
    for task_name in args.tasks:
        task = TASKS[task_name]
        for size in args.sizes:
            model_id = f"yolo26{size}{task.suffix}"
            print(f"\nExporting {model_id} ({task_name}, imgsz={task.imgsz}, data={args.data})")
            exported = Path(
                YOLO(f"{model_id}.pt").export(
                    format="tflite",
                    int8=True,
                    data=args.data,
                    nms=False,
                    imgsz=task.imgsz,
                    batch=1,
                )
            )
            target = release_dir / exported.name
            shutil.copy2(exported, target)
            outputs = verify_tflite(target) if args.verify else []
            suffix = f" outputs={outputs}" if outputs else ""
            print(f"asset {target.relative_to(ROOT)} size={target.stat().st_size / 1_000_000:.2f} MB{suffix}")
            assets.append(target)

    if args.upload:
        upload_assets(args.repo, args.tag, assets)

    print(f"\nPrepared {len(assets)} TFLite release assets in {release_dir}")


if __name__ == "__main__":
    main()
