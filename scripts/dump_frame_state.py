#!/usr/bin/env python3
"""Dump the pose (T_WC) and canonical surfel points (X_canon) for one frame.

This script reproduces the first-step logic from ``main.py``:
- load the requested dataset frame
- run MASt3R inference to populate the surfel map
- print pose and surfel statistics so you can compare desktop vs web outputs

Example usage::

    python scripts/dump_frame_state.py \
        --dataset datasets/tum/rgbd_dataset_freiburg1_desk \
        --config config/base.yaml \
        --frame-index 0
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, Optional

import numpy as np
import torch
import yaml

from mast3r_slam.config import config, load_config, set_global_config
from mast3r_slam.dataloader import Intrinsics, load_dataset
from mast3r_slam.frame import create_frame
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.mast3r_utils import load_mast3r, mast3r_inference_mono
from lietorch import Sim3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect pose and surfel data for a frame.")
    parser.add_argument("--dataset", required=True, help="Dataset root directory.")
    parser.add_argument("--config", default="config/base.yaml", help="Path to config YAML.")
    parser.add_argument("--frame-index", type=int, default=0, help="Dataset frame index to inspect.")
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for inference (defaults to CUDA when available).",
    )
    parser.add_argument(
        "--calib",
        type=pathlib.Path,
        default=None,
        help="Optional calibration YAML (same schema as main.py --calib).",
    )
    parser.add_argument(
        "--output-json",
        type=pathlib.Path,
        default=None,
        help="Optional path to write a JSON summary.",
    )
    parser.add_argument(
        "--save-npy-dir",
        type=pathlib.Path,
        default=None,
        help="Optional directory to save raw NumPy arrays (T_WC.npy, X_canon.npy, C.npy).",
    )
    parser.add_argument(
        "--max-print-points",
        type=int,
        default=5,
        help="Number of X_canon entries to show in the console (default: 5).",
    )
    return parser.parse_args()


def maybe_apply_calibration(dataset, calib_path: Optional[pathlib.Path]) -> None:
    if calib_path is None:
        return
    with calib_path.open("r") as f:
        intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
    dataset.use_calibration = True
    dataset.camera_intrinsics = Intrinsics.from_calib(
        dataset.img_size,
        intrinsics["width"],
        intrinsics["height"],
        intrinsics["calibration"],
    )


def summarize_points(points: torch.Tensor) -> Dict[str, object]:
    pts = points.detach().cpu().view(-1, 3).numpy()
    return {
        "shape": list(pts.shape),
        "min": pts.min(axis=0).tolist(),
        "max": pts.max(axis=0).tolist(),
        "mean": pts.mean(axis=0).tolist(),
        "std": pts.std(axis=0).tolist(),
    }


def to_serializable(payload: Dict[str, object]) -> Dict[str, object]:
    serializable: Dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            serializable[key] = value.detach().cpu().numpy().tolist()
        elif isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        else:
            serializable[key] = value
    return serializable


def main() -> None:
    torch.set_grad_enabled(False)
    args = parse_args()

    load_config(args.config)
    set_global_config(config)

    dataset = load_dataset(args.dataset)
    dataset.subsample(config["dataset"]["subsample"])
    maybe_apply_calibration(dataset, args.calib)

    if args.frame_index < 0 or args.frame_index >= len(dataset):
        raise IndexError(
            f"frame-index {args.frame_index} out of bounds for dataset size {len(dataset)}"
        )

    timestamp, img = dataset[args.frame_index]

    device = torch.device(args.device)

    frame = create_frame(
        i=args.frame_index,
        img=img,
        T_WC=Sim3.Identity(1, device=device),
        img_size=dataset.img_size,
        device=device,
        use_fp16=False,
    )

    model = load_mast3r(device=args.device)
    model.eval()

    X, C = mast3r_inference_mono(model, frame)
    frame.update_pointmap(X, C)

    pose = as_SE3(frame.T_WC).matrix().detach().cpu().numpy()
    x_canon = frame.X_canon.detach().cpu()
    confidence = frame.C.detach().cpu() if frame.C is not None else None

    stats = summarize_points(frame.X_canon)

    print("=== Frame dump ===")
    print(f"dataset       : {args.dataset}")
    print(f"frame index   : {args.frame_index}")
    print(f"timestamp     : {timestamp}")
    print(f"image shape   : {tuple(frame.img_shape.flatten().tolist())}")
    print("\nT_WC (4x4 pose matrix):")
    print(np.array2string(pose, precision=5, suppress_small=True))

    print("\nX_canon stats:")
    for key, value in stats.items():
        print(f"  {key:>5}: {value}")

    sample_count = min(args.max_print_points, x_canon.view(-1, 3).shape[0])
    print(f"\nFirst {sample_count} surfel positions (world frame):")
    print(x_canon.view(-1, 3)[:sample_count])

    if args.save_npy_dir is not None:
        args.save_npy_dir.mkdir(parents=True, exist_ok=True)
        np.save(args.save_npy_dir / "T_WC.npy", pose)
        np.save(args.save_npy_dir / "X_canon.npy", x_canon.numpy())
        if confidence is not None:
            np.save(args.save_npy_dir / "C.npy", confidence.numpy())
        print(f"\nSaved arrays to {args.save_npy_dir}")

    if args.output_json is not None:
        payload = {
            "dataset": args.dataset,
            "frame_index": args.frame_index,
            "timestamp": float(timestamp),
            "image_shape": tuple(frame.img_shape.flatten().tolist()),
            "T_WC": pose,
            "X_canon_stats": stats,
            "X_canon_head": x_canon.view(-1, 3)[: args.max_print_points],
        }
        if confidence is not None:
            payload["confidence_head"] = confidence.view(-1)[: args.max_print_points]
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as f:
            json.dump(to_serializable(payload), f, indent=2)
        print(f"Summary written to {args.output_json}")


if __name__ == "__main__":
    main()
