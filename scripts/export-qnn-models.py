# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Export official YOLO26 QNN assets for the Flutter Android release.

QNN export requires Windows x64 or Linux x86-64 with a QNN-enabled ONNX Runtime build. Official assets target HTP v73
and v81, use W8A16 quantization, and are named `<model>_v<arch>_qnn.onnx`.

Usage from the repository root:

    uv venv --python 3.12 .venv
    uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
    uv pip install "ultralytics[export]"
    uv run python scripts/export-qnn-models.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import onnx
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "exports" / "yolo26-qnn"
DEFAULT_REPO = "ultralytics/yolo-flutter-app"
DEFAULT_TAG = "models-v1.0.0"
ARCHITECTURES = ("73", "81")


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


def display_path(path: Path) -> str:
    """Return a repository-relative path when possible."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    parser.add_argument("--tasks", nargs="+", choices=TASKS.keys(), default=list(TASKS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload generated assets to a new GitHub release.",
    )
    return parser.parse_args()


def verify_qnn(path: Path, task_name: str, imgsz: int) -> None:
    """Verify a QNN context binary's fixed channel-last input and metadata."""
    model = onnx.load(str(path))
    inputs = model.graph.input
    if len(inputs) != 1:
        raise ValueError(f"{path.name} has {len(inputs)} inputs; expected 1")
    shape = [dimension.dim_value for dimension in inputs[0].type.tensor_type.shape.dim]
    if shape != [1, imgsz, imgsz, 3]:
        raise ValueError(f"{path.name} input is {shape}; expected [1, {imgsz}, {imgsz}, 3]")
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    if metadata.get("task") != task_name or json.loads(metadata.get("imgsz", "null")) != [imgsz, imgsz]:
        raise ValueError(f"{path.name} metadata does not declare task={task_name}, imgsz=[{imgsz}, {imgsz}]")


def main() -> None:
    """Export and verify the official QNN release assets."""
    args = parse_args()
    output_dir = args.output_dir.resolve()
    release_dir = output_dir / "release-assets"
    release_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)

    assets: list[Path] = []
    for task_name in args.tasks:
        task = TASKS[task_name]
        model_id = f"yolo26n{task.suffix}"
        for architecture in args.architectures:
            target = release_dir / f"{model_id}_v{architecture}_qnn.onnx"
            if target.exists() and not args.force:
                verify_qnn(target, task_name, task.imgsz)
                print(f"Skipping {target.name}; verified input={task.imgsz}x{task.imgsz}")
                assets.append(target)
                continue
            exported = Path(
                YOLO(str(output_dir / f"{model_id}.pt")).export(
                    format="qnn",
                    name=architecture,
                    imgsz=task.imgsz,
                    batch=1,
                    nms=False,
                    end2end=False,
                )
            )
            shutil.move(exported, target)
            verify_qnn(target, task_name, task.imgsz)
            print(f"asset {display_path(target)} input={task.imgsz}x{task.imgsz}")
            assets.append(target)

    if args.upload:
        subprocess.run(
            [
                "gh",
                "release",
                "upload",
                args.tag,
                "--repo",
                args.repo,
                *(str(path) for path in assets),
            ],
            check=True,
        )

    print(f"\nPrepared {len(assets)} QNN release assets in {release_dir}")


if __name__ == "__main__":
    main()
