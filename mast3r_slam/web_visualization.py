"""Web visualization module for MASt3R-SLAM."""

from __future__ import annotations

import base64
import io
import json
import math
import threading
import time
from dataclasses import dataclass, field
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from mast3r_slam.config import set_global_config
from mast3r_slam.frame import Frame, Mode
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.visualization import WindowMsg


@dataclass
class CachedPointCloud:
    """Serializable buffer for surfel data."""

    points: List[float] = field(default_factory=list)
    normals: List[float] = field(default_factory=list)
    colors: List[float] = field(default_factory=list)
    count: int = 0
    version: int = 0

    def extend(self, other: "CachedPointCloud") -> None:
        if other.count == 0:
            return
        self.points.extend(other.points)
        self.normals.extend(other.normals)
        self.colors.extend(other.colors)
        self.count += other.count


@dataclass
class KeyframeEntry:
    """Pose and cached surfels for a single keyframe."""

    index: int
    frame_id: int
    pose: List[List[float]]
    position: List[float]
    version: int
    cloud: Optional[CachedPointCloud] = None


class WebVisualizationProvider:
    """Produces JSON payloads consumed by the browser client."""

    def __init__(
        self,
        states,
        keyframes,
        *,
        conf_threshold: float = 1.5,
        current_stride: int = 4,
        keyframe_stride: int = 8,
    ) -> None:
        self.states = states
        self.keyframes = keyframes
        self.conf_threshold = conf_threshold
        self.current_stride = current_stride
        self.keyframe_stride = keyframe_stride
        self._lock = threading.Lock()
        self._keyframe_cache: Dict[int, KeyframeEntry] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_conf_threshold(self, value: float) -> None:
        with self._lock:
            self.conf_threshold = float(value)

    def build_state(self) -> Dict[str, object]:
        with self._lock:
            frame = self.states.get_frame()
            pose_matrix, position = self._pose_and_position(frame)
            image_uri = self._encode_image(frame.uimg)

            keyframe_entries = self._refresh_keyframes()
            edges = self._collect_edges(keyframe_entries)

            current_cloud = self._compute_point_cloud(
                frame,
                stride=self.current_stride,
                conf_threshold=self.conf_threshold,
            )

            yaw_raw, yaw_timestamp = self.states.get_imu_yaw()
            yaw_corrected = None
            if yaw_raw is not None:
                yaw_corrected, _ = self.states.get_imu_yaw(corrected=True)
            calibrated = self.states.is_imu_yaw_calibrated()
            heading_angle = yaw_corrected if yaw_corrected is not None else yaw_raw
            heading_vector = None
            if heading_angle is not None:
                heading_vector = [
                    math.sin(heading_angle),
                    0.0,
                    math.cos(heading_angle),
                ]
            imu_payload = {
                "available": yaw_raw is not None,
                "calibrated": bool(calibrated),
                "rawYawRad": yaw_raw,
                "rawYawDeg": math.degrees(yaw_raw) if yaw_raw is not None else None,
                "correctedYawRad": yaw_corrected,
                "correctedYawDeg": math.degrees(yaw_corrected) if yaw_corrected is not None else None,
                "timestamp": yaw_timestamp,
                "headingVector": heading_vector,
            }

            aggregated = CachedPointCloud()
            aggregated.extend(current_cloud)
            for entry in keyframe_entries:
                if entry.cloud is not None:
                    aggregated.extend(entry.cloud)

            payload = {
                "ready": aggregated.count > 0,
                "confThreshold": self.conf_threshold,
                "currentFrame": {
                    "id": int(frame.frame_id),
                    "pose": pose_matrix,
                    "position": position,
                    "image": image_uri,
                    "pointCount": current_cloud.count,
                },
                "scene": {
                    "cameraPose": pose_matrix,
                    "points": aggregated.points,
                    "normals": aggregated.normals,
                    "colors": aggregated.colors,
                    "count": aggregated.count,
                },
                "keyframes": [
                    {
                        "index": entry.index,
                        "frameId": entry.frame_id,
                        "pose": entry.pose,
                        "position": entry.position,
                        "pointCount": entry.cloud.count if entry.cloud else 0,
                    }
                    for entry in keyframe_entries
                ],
                "edges": edges,
                "imu": imu_payload,
            }
        return payload

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _pose_and_position(self, frame: Frame) -> (List[List[float]], List[float]):
        pose = as_SE3(frame.T_WC).matrix().detach().cpu().numpy()
        if pose.ndim == 3:
            pose = pose[0]
        if pose.shape == (3, 4):
            full_pose = np.eye(4, dtype=np.float32)
            full_pose[:3, :3] = pose[:, :3]
            full_pose[:3, 3] = pose[:, 3]
            pose = full_pose
        position = pose[:3, 3]
        return pose.tolist(), position.astype(np.float32).tolist()

    def _refresh_keyframes(self) -> List[KeyframeEntry]:
        entries: List[KeyframeEntry] = []
        for idx in range(len(self.keyframes)):
            kf = self.keyframes[idx]
            pose, position = self._pose_and_position(kf)
            version = int(kf.N_updates)
            frame_id = int(kf.frame_id)
            cached = self._keyframe_cache.get(idx)
            if cached is None or cached.version != version or cached.frame_id != frame_id:
                cloud = self._compute_point_cloud(
                    kf,
                    stride=self.keyframe_stride,
                    conf_threshold=self.conf_threshold,
                )
                cached = KeyframeEntry(
                    index=idx,
                    frame_id=frame_id,
                    pose=pose,
                    position=position,
                    version=version,
                    cloud=cloud if cloud.count > 0 else None,
                )
                self._keyframe_cache[idx] = cached
            entries.append(cached)

        stale = [idx for idx in self._keyframe_cache if idx >= len(self.keyframes)]
        for idx in stale:
            del self._keyframe_cache[idx]

        return entries

    def _collect_edges(self, keyframes: List[KeyframeEntry]) -> List[Dict[str, object]]:
        if not keyframes:
            return []
        positions = {entry.index: entry.position for entry in keyframes}
        edges: List[Dict[str, object]] = []
        with self.states.lock:
            ii = list(self.states.edges_ii)
            jj = list(self.states.edges_jj)
        for src, dst in zip(ii, jj):
            src_idx = int(src)
            dst_idx = int(dst)
            if src_idx not in positions or dst_idx not in positions:
                continue
            edges.append(
                {
                    "a": src_idx,
                    "b": dst_idx,
                    "points": [positions[src_idx], positions[dst_idx]],
                }
            )
        return edges

    def _compute_point_cloud(
        self,
        frame: Frame,
        *,
        stride: int,
        conf_threshold: float,
    ) -> CachedPointCloud:
        average_conf = frame.get_average_conf()
        if frame.X_canon is None or frame.C is None or average_conf is None:
            return CachedPointCloud()

        img_shape = frame.img_shape.view(-1).detach().cpu().tolist()
        if len(img_shape) < 2:
            return CachedPointCloud()
        h, w = int(img_shape[0]), int(img_shape[1])
        if h < 3 or w < 3:
            return CachedPointCloud()

        stride = max(1, int(stride))
        stride = min(stride, h - 1, w - 1)

        X = frame.X_canon.detach().view(h, w, -1).cpu().numpy()
        if X.shape[-1] > 3:
            X = X[..., :3]
        C = average_conf.detach().view(h, w, -1).cpu().numpy()[..., 0]
        colors = frame.uimg.cpu().numpy().astype(np.float32)

        pose = frame.T_WC.matrix().detach().cpu().numpy()
        if pose.ndim == 3:
            pose = pose[0]
        if pose.shape == (3, 4):
            R = pose[:, :3]
            t = pose[:, 3]
        else:
            R = pose[:3, :3]
            t = pose[:3, 3]
        R = np.asarray(R, dtype=np.float32)
        t = np.asarray(t, dtype=np.float32).reshape(3)

        points: List[float] = []
        normals: List[float] = []
        rgb: List[float] = []
        count = 0

        y_max = h - 1
        x_max = w - 1

        for y in range(1, y_max, stride):
            for x in range(1, x_max, stride):
                if C[y, x] < conf_threshold:
                    continue
                x_r = min(x + stride, x_max)
                y_d = min(y + stride, y_max)
                p_cam = X[y, x, :3]
                px_cam = X[y, x_r, :3]
                py_cam = X[y_d, x, :3]

                p_world = (R @ p_cam) + t
                px_world = (R @ px_cam) + t
                py_world = (R @ py_cam) + t

                normal = np.cross(py_world - p_world, px_world - p_world)
                norm = float(np.linalg.norm(normal))
                if norm < 1e-8:
                    continue
                normal /= norm

                points.extend(p_world.astype(np.float32).tolist())
                normals.extend(normal.astype(np.float32).tolist())
                rgb.extend(colors[y, x, :3].tolist())
                count += 1

        return CachedPointCloud(points=points, normals=normals, colors=rgb, count=count, version=frame.N_updates)

    def _encode_image(self, tensor: torch.Tensor) -> str:
        array = tensor.detach().cpu().numpy()
        array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
        image = Image.fromarray(array)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"


class VisualizationRequestHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, directory: str, provider: WebVisualizationProvider, **kwargs):
        self.provider = provider
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/api/state"):
            self._respond_json(self.provider.build_state())
            return
        if self.path.startswith("/api/config"):
            self._respond_json({"confThreshold": self.provider.conf_threshold})
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.startswith("/api/config"):
            self.send_error(404, "Unsupported endpoint")
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            data = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return
        if "confThreshold" in data:
            self.provider.set_conf_threshold(float(data["confThreshold"]))
        self._respond_json({"status": "ok", "confThreshold": self.provider.conf_threshold})

    def log_message(self, fmt: str, *args) -> None:  # noqa: N802
        del fmt, args

    def _respond_json(self, payload: Dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_web_visualization(
    cfg,
    states,
    keyframes,
    main2viz,
    viz2main,
    *,
    host: str = "0.0.0.0",
    port: int = 7860,
    conf_threshold: float = 1.5,
    current_stride: int = 4,
    keyframe_stride: int = 8,
) -> None:
    del main2viz
    set_global_config(cfg)

    provider = WebVisualizationProvider(
        states,
        keyframes,
        conf_threshold=conf_threshold,
        current_stride=current_stride,
        keyframe_stride=keyframe_stride,
    )

    static_dir = (Path(__file__).parent.parent / "resources" / "web").resolve()
    static_dir.mkdir(parents=True, exist_ok=True)

    handler = partial(VisualizationRequestHandler, directory=str(static_dir), provider=provider)
    server = ThreadingHTTPServer((host, port), handler)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[WebViz] Serving MASt3R-SLAM web viewer at http://{host}:{port}")

    try:
        while True:
            if states.get_mode() == Mode.TERMINATED:
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        viz2main.put(WindowMsg(is_terminated=True))
