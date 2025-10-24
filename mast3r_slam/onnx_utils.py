"""ONNX runtime helpers for MASt3R inference."""

from __future__ import annotations

import pathlib
from typing import Dict, Iterable, List, Optional

import einops
import numpy as np
import torch

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "onnxruntime is required for ONNX execution. Install it via pip install onnxruntime-gpu"
    ) from exc

from mast3r_slam.mast3r_utils import downsample


class Mast3ROnnxRunner:
    """Wrapper around an ONNX-exported MASt3R model."""

    def __init__(
        self,
        model_path: pathlib.Path | str,
        device: str = "cuda:0",
        providers: Optional[Iterable[str]] = None,
    ) -> None:
        model_path = pathlib.Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(model_path)

        self.device = torch.device(device)
        self.cpu_device = torch.device("cpu")

        if providers is None:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if torch.cuda.is_available()
                else ["CPUExecutionProvider"]
            )

        sess_options = ort.SessionOptions()
        self.session = ort.InferenceSession(
            model_path.as_posix(),
            sess_options=sess_options,
            providers=list(providers),
        )
        self.input_names = [node.name for node in self.session.get_inputs()]
        self.output_names = [node.name for node in self.session.get_outputs()]

        type_mapping = {"tensor(float)": np.float32, "tensor(float16)": np.float16}
        shape_mapping = {"tensor(int32)": np.int32, "tensor(int64)": np.int64}
        self.img_dtype = np.float32
        self.shape_dtype = np.int32
        for node in self.session.get_inputs():
            if node.name == "img1":
                self.img_dtype = type_mapping.get(node.type, np.float32)
            if node.name == "true_shape1":
                self.shape_dtype = shape_mapping.get(node.type, np.int32)
        self.requires_fp16 = self.img_dtype == np.float16

    def _frame_to_numpy(self, frame) -> Dict[str, np.ndarray]:
        img = frame.img.detach().to(device=self.cpu_device).cpu().numpy()
        if img.dtype != self.img_dtype:
            img = img.astype(self.img_dtype)
        true_shape = (
            frame.img_true_shape.detach()
            .to(device=self.cpu_device, dtype=torch.int32)
            .cpu()
            .numpy()
        ).astype(self.shape_dtype)
        return {"img": img, "true_shape": true_shape}

    def _run_pair(self, frame_i, frame_j) -> Dict[str, np.ndarray]:
        inputs_i = self._frame_to_numpy(frame_i)
        inputs_j = self._frame_to_numpy(frame_j)
        ort_inputs = {
            "img1": inputs_i["img"],
            "true_shape1": inputs_i["true_shape"],
            "img2": inputs_j["img"],
            "true_shape2": inputs_j["true_shape"],
        }
        outputs = self.session.run(None, ort_inputs)
        return {name: value for name, value in zip(self.output_names, outputs)}

    def _to_tensor(self, array: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        tensor = torch.from_numpy(array).to(device=self.device)
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        return tensor

    def _extract_shared_features(
        self,
        outputs: Dict[str, np.ndarray],
        frame_i,
        frame_j,
    ) -> None:
        feat1 = self._to_tensor(outputs["feat1"])
        pos1 = self._to_tensor(outputs["pos1"])
        feat2 = self._to_tensor(outputs["feat2"])
        pos2 = self._to_tensor(outputs["pos2"])

        frame_i.feat = feat1
        frame_i.pos = pos1
        frame_j.feat = feat2
        frame_j.pos = pos2

    def mast3r_inference_mono(self, frame):
        outputs = self._run_pair(frame, frame)
        self._extract_shared_features(outputs, frame, frame)

        names_pts = ["pts3d_12_a", "pts3d_12_b"]
        names_conf = ["conf_12_a", "conf_12_b"]
        names_desc = ["desc_12_a", "desc_12_b"]
        names_q = ["desc_conf_12_a", "desc_conf_12_b"]

        X, C, D, Q = self._collect_outputs(outputs, names_pts, names_conf, names_desc, names_q)
        X, C, D, Q = downsample(X, C, D, Q)

        Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
        Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")
        return Xii, Cii

    def mast3r_asymmetric_inference(self, frame_i, frame_j):
        outputs = self._run_pair(frame_i, frame_j)
        self._extract_shared_features(outputs, frame_i, frame_j)

        names_pts = ["pts3d_12_a", "pts3d_12_b"]
        names_conf = ["conf_12_a", "conf_12_b"]
        names_desc = ["desc_12_a", "desc_12_b"]
        names_q = ["desc_conf_12_a", "desc_conf_12_b"]
        return self._collect_outputs(outputs, names_pts, names_conf, names_desc, names_q)

    def mast3r_symmetric_inference(self, frame_i, frame_j):
        outputs = self._run_pair(frame_i, frame_j)
        self._extract_shared_features(outputs, frame_i, frame_j)

        names_pts = ["pts3d_12_a", "pts3d_12_b", "pts3d_21_a", "pts3d_21_b"]
        names_conf = ["conf_12_a", "conf_12_b", "conf_21_a", "conf_21_b"]
        names_desc = ["desc_12_a", "desc_12_b", "desc_21_a", "desc_21_b"]
        names_q = [
            "desc_conf_12_a",
            "desc_conf_12_b",
            "desc_conf_21_a",
            "desc_conf_21_b",
        ]
        X, C, D, Q = self._collect_outputs(outputs, names_pts, names_conf, names_desc, names_q)
        X, C, D, Q = downsample(X, C, D, Q)
        return X, C, D, Q

    def _collect_outputs(
        self,
        outputs: Dict[str, np.ndarray],
        pts_names: List[str],
        conf_names: List[str],
        desc_names: List[str],
        q_names: List[str],
    ) -> List[torch.Tensor]:
        pts = [self._to_tensor(outputs[name])[0] for name in pts_names]
        conf = [self._to_tensor(outputs[name])[0] for name in conf_names]
        desc = [self._to_tensor(outputs[name])[0] for name in desc_names]
        q = [self._to_tensor(outputs[name])[0] for name in q_names]

        X = torch.stack(pts)
        C = torch.stack(conf)
        D = torch.stack(desc)
        Q = torch.stack(q)
        return X, C, D, Q


def patch_mast3r_utils_for_onnx(runner: Mast3ROnnxRunner) -> None:
    """Replace heavy PyTorch inference helpers with ONNX-backed versions."""

    import mast3r_slam.mast3r_utils as mast3r_utils

    def inference_mono(model, frame):  # pylint: disable=unused-argument
        return runner.mast3r_inference_mono(frame)

    def asymmetric_inference(model, frame_i, frame_j):  # pylint: disable=unused-argument
        return runner.mast3r_asymmetric_inference(frame_i, frame_j)

    def symmetric_inference(model, frame_i, frame_j):  # pylint: disable=unused-argument
        return runner.mast3r_symmetric_inference(frame_i, frame_j)

    mast3r_utils.mast3r_inference_mono = inference_mono
    mast3r_utils.mast3r_asymmetric_inference = asymmetric_inference
    mast3r_utils.mast3r_symmetric_inference = symmetric_inference
