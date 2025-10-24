#!/usr/bin/env python
"""
Utility script to export MASt3R checkpoints to ONNX using Transformer Engine helpers.

This relies on torch>=2.4 and transformer_engine>=1.10 with ONNX export support.
The resulting ONNX graph emits intermediate features so that the SLAM pipeline
can operate entirely on the exported graphs during runtime.
"""

import argparse
import pathlib
import sys
from typing import Tuple

import torch
import torch._dynamo as dynamo

from transformer_engine.pytorch.export import onnx_export, te_translation_table


# Local imports must happen after project root is on sys.path
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mast3r_slam.mast3r_utils import load_mast3r  # noqa: E402
import thirdparty.mast3r.dust3r.croco.models.curope.curope2d as curope2d_mod  # noqa: E402


ORIGINAL_CUROPE2D = curope2d_mod.cuRoPE2D


class Mast3RExportWrapper(torch.nn.Module):
    """Wrap MASt3R to expose all tensors required by the SLAM pipeline."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        img1: torch.Tensor,
        true_shape1: torch.Tensor,
        img2: torch.Tensor,
        true_shape2: torch.Tensor,
    ):
        feat1, pos1, _ = self.model._encode_image(img1, true_shape1)
        feat2, pos2, _ = self.model._encode_image(img2, true_shape2)

        dec12_a, dec12_b = self.model._decoder(feat1, pos1, feat2, pos2)
        res12_a = self.model._downstream_head(1, dec12_a, true_shape1)
        res12_b = self.model._downstream_head(2, dec12_b, true_shape2)

        dec21_a, dec21_b = self.model._decoder(feat2, pos2, feat1, pos1)
        res21_a = self.model._downstream_head(1, dec21_a, true_shape2)
        res21_b = self.model._downstream_head(2, dec21_b, true_shape1)

        return (
            feat1,
            pos1,
            feat2,
            pos2,
            res12_a["pts3d"],
            res12_a["conf"],
            res12_a["desc"],
            res12_a["desc_conf"],
            res12_b["pts3d"],
            res12_b["conf"],
            res12_b["desc"],
            res12_b["desc_conf"],
            res21_a["pts3d"],
            res21_a["conf"],
            res21_a["desc"],
            res21_a["desc_conf"],
            res21_b["pts3d"],
            res21_b["conf"],
            res21_b["desc"],
            res21_b["desc_conf"],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MASt3R checkpoint to ONNX.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
        help="Path to MASt3R checkpoint (same argument as load_mast3r).",
    )
    parser.add_argument(
        "--onnx-dir",
        default="onnx",
        help="Directory where ONNX files are written.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for export (cuda or cpu).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(512, 512),
        help="Image resolution used for the dummy export input.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version to use.",
    )
    parser.add_argument(
        "--precision",
        choices=["fp16", "fp32"],
        default="fp16",
        help="Numeric precision of exported weights and inputs.",
    )
    return parser.parse_args()


class TorchRoPE2D(torch.nn.Module):
    """Pure PyTorch implementation of the cuRoPE2D kernel for export."""

    def __init__(self, freq: float = 100.0, F0: float = 1.0):
        super().__init__()
        self.base = float(freq)
        self.F0 = float(F0)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        B, N, H, D_total = tokens.shape
        assert D_total % 4 == 0, "Expected final dimension divisible by 4 for 2D RoPE"
        D = D_total // 4

        dtype = tokens.dtype
        device = tokens.device

        tokens_view = tokens.view(B, N, H, 2, 2, D)
        pos = positions.to(device=device, dtype=dtype)

        idx = torch.arange(D, device=device, dtype=dtype) / D
        base = torch.tensor(self.base, device=device, dtype=dtype)
        denom = torch.pow(base, idx)
        coeff = self.F0 / denom

        for axis in range(2):
            angle = pos[..., axis].unsqueeze(-1) * coeff
            angle = angle.unsqueeze(2)
            sin = torch.sin(angle)
            cos = torch.cos(angle)

            u = tokens_view[..., axis, 0, :]
            v = tokens_view[..., axis, 1, :]
            tokens_view[..., axis, 0, :] = u * cos - v * sin
            tokens_view[..., axis, 1, :] = v * cos + u * sin

        return tokens


def replace_rope_modules(module: torch.nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, ORIGINAL_CUROPE2D):
            setattr(module, name, TorchRoPE2D(child.base, child.F0))
        else:
            replace_rope_modules(child)


curope2d_mod.cuRoPE2D = TorchRoPE2D


def make_dummy_inputs(
    device: torch.device, hw: Tuple[int, int], dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    h, w = hw
    img = torch.randn(1, 3, h, w, device=device, dtype=dtype)
    true_shape = torch.tensor([[h, w]], device=device, dtype=torch.int32)
    return img, true_shape


def export_mast3r_onnx(args: argparse.Namespace) -> pathlib.Path:
    device = torch.device(args.device)
    checkpoint = pathlib.Path(args.checkpoint)
    output_dir = pathlib.Path(args.onnx_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_mast3r(path=str(checkpoint), device=str(device), use_fp8=False)
    model.eval()

    replace_rope_modules(model)

    if args.precision == "fp16":
        model = model.half()
        input_dtype = torch.float16
    else:
        input_dtype = torch.float32

    wrapper = Mast3RExportWrapper(model)
    wrapper.eval()

    dummy_img, dummy_shape = make_dummy_inputs(device, tuple(args.img_size), input_dtype)

    # Warm-up pass required by Transformer Engine export helpers.
    with torch.no_grad():
        _ = wrapper(dummy_img, dummy_shape, dummy_img, dummy_shape)
    setattr(wrapper, "forwarded_at_least_once", True)

    output_path = output_dir / checkpoint.name.replace(".pth", ".onnx")

    input_names = ["img1", "true_shape1", "img2", "true_shape2"]
    output_names = [
        "feat1",
        "pos1",
        "feat2",
        "pos2",
        "pts3d_12_a",
        "conf_12_a",
        "desc_12_a",
        "desc_conf_12_a",
        "pts3d_12_b",
        "conf_12_b",
        "desc_12_b",
        "desc_conf_12_b",
        "pts3d_21_a",
        "conf_21_a",
        "desc_21_a",
        "desc_conf_21_a",
        "pts3d_21_b",
        "conf_21_b",
        "desc_21_b",
        "desc_conf_21_b",
    ]
    dynamo.config.dynamic_shapes = False

    try:
        with onnx_export(enabled=True):
            torch.onnx.export(
                wrapper,
                (dummy_img, dummy_shape, dummy_img, dummy_shape),
                output_path.as_posix(),
                export_params=True,
                do_constant_folding=False,
                opset_version=args.opset,
                input_names=input_names,
                output_names=output_names,
                custom_translation_table=te_translation_table,
                dynamo=True,
            )
    except ModuleNotFoundError as exc:
        if "onnxscript" in str(exc):
            raise ModuleNotFoundError(
                "onnxscript is required for dynamo ONNX export. Install it via `pip install onnxscript>=0.2.5`."
            ) from exc
        raise

    return output_path


def main() -> None:
    args = parse_args()
    path = export_mast3r_onnx(args)
    print(f"Exported ONNX model to {path}")


if __name__ == "__main__":
    main()
