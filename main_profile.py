#!/usr/bin/env python3

import argparse
import collections
import time
import types
from typing import Deque, Dict, List

import lietorch
import torch
import yaml

from mast3r_slam.config import config, load_config
from mast3r_slam.dataloader import Intrinsics, load_dataset
from mast3r_slam.frame import Frame, create_frame
from mast3r_slam.global_opt import FactorGraph
import mast3r_slam.matching as matching
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_slam.tracker import FrameTracker


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class StageStats:
    def __init__(self):
        self.total = 0.0
        self.count = 0
        self.max = 0.0
        self.min = float("inf")

    def add(self, duration: float):
        self.total += duration
        self.count += 1
        self.max = max(self.max, duration)
        self.min = min(self.min, duration)


class PipelineProfiler:
    def __init__(self):
        self.global_stats: Dict[str, StageStats] = collections.defaultdict(StageStats)
        self.per_frame: Dict[int, Dict[str, float]] = {}
        self.frame_order: List[int] = []
        self.stage_order: List[str] = []
        self.current_frame: int | None = None

    def start_frame(self, frame_id: int):
        self.current_frame = frame_id
        self.per_frame[frame_id] = collections.defaultdict(float)
        self.frame_order.append(frame_id)

    def record(self, stage: str, duration: float):
        stats = self.global_stats[stage]
        stats.add(duration)
        if stage not in self.stage_order:
            self.stage_order.append(stage)
        if self.current_frame is not None:
            self.per_frame[self.current_frame][stage] += duration

    def end_frame(self, frame_id: int):
        if frame_id not in self.per_frame:
            return
        frame_stats = self.per_frame[frame_id]
        if frame_stats:
            parts = []
            for stage in self.stage_order:
                if stage == "frame_total":
                    continue
                if stage in frame_stats:
                    parts.append(f"{stage}={frame_stats[stage] * 1000.0:.2f}ms")
            if "frame_total" in frame_stats:
                parts.append(f"frame_total={frame_stats['frame_total'] * 1000.0:.2f}ms")
            print(f"[Frame {frame_id:04d}] {', '.join(parts)}")
        self.current_frame = None

    def summary(self):
        print("\n=== Aggregate Timings ===")
        total_frame_time = self.global_stats.get("frame_total", StageStats()).total
        for stage, stats in sorted(
            self.global_stats.items(), key=lambda item: item[1].total, reverse=True
        ):
            if stats.count == 0:
                continue
            avg_ms = stats.total / stats.count * 1000.0
            max_ms = stats.max * 1000.0
            min_ms = stats.min * 1000.0 if stats.min < float("inf") else 0.0
            pct = (
                stats.total / total_frame_time * 100.0
                if total_frame_time > 0.0
                else 0.0
            )
            print(
                f"{stage:>20}: total={stats.total:.3f}s "
                f"count={stats.count:4d} avg={avg_ms:.2f}ms "
                f"min={min_ms:.2f}ms max={max_ms:.2f}ms "
                f"percent_of_frame={pct:5.1f}%"
            )

    @property
    def num_frames(self) -> int:
        return len(self.frame_order)


class LocalKeyframes:
    def __init__(self, device: str):
        self._frames: List[Frame] = []
        self.device = device
        self._K: torch.Tensor | None = None

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx: int) -> Frame:
        return self._frames[idx]

    def __setitem__(self, idx: int, value: Frame) -> None:
        self._frames[idx] = value

    def append(self, frame: Frame):
        self._frames.append(frame)

    def pop_last(self):
        if self._frames:
            self._frames.pop()

    def last_keyframe(self) -> Frame | None:
        return self._frames[-1] if self._frames else None

    def update_T_WCs(self, T_WCs: lietorch.Sim3, idx: torch.Tensor):
        for pose, frame_idx in zip(T_WCs, idx):
            self._frames[int(frame_idx)].T_WC = pose

    def set_intrinsics(self, K: torch.Tensor):
        self._K = K

    def get_intrinsics(self) -> torch.Tensor | None:
        return self._K


def instrument_model(model, profiler: PipelineProfiler):
    def wrap_method(attr_name: str, stage_name: str):
        method = getattr(model, attr_name)
        if not hasattr(method, "__func__"):
            return
        fn = method.__func__

        def wrapped(self, *args, **kwargs):
            sync_cuda()
            start = time.perf_counter()
            result = fn(self, *args, **kwargs)
            sync_cuda()
            profiler.record(stage_name, time.perf_counter() - start)
            return result

        setattr(model, attr_name, types.MethodType(wrapped, model))

    wrap_method("_encode_image", "encoding")
    wrap_method("_decoder", "decoder")
    wrap_method("_downstream_head", "downstream_head")


def instrument_matching(profiler: PipelineProfiler):
    original_match = matching.match

    def wrapped_match(*args, **kwargs):
        sync_cuda()
        start = time.perf_counter()
        result = original_match(*args, **kwargs)
        sync_cuda()
        profiler.record("matching", time.perf_counter() - start)
        return result

    matching.match = wrapped_match


def instrument_tracker(tracker: FrameTracker, profiler: PipelineProfiler):
    def wrap(attr_name: str):
        method = getattr(tracker, attr_name)
        if not hasattr(method, "__func__"):
            return
        fn = method.__func__

        def wrapped(self, *args, **kwargs):
            sync_cuda()
            start = time.perf_counter()
            out = fn(self, *args, **kwargs)
            sync_cuda()
            profiler.record("tracking_optimization", time.perf_counter() - start)
            return out

        setattr(tracker, attr_name, types.MethodType(wrapped, tracker))

    wrap("opt_pose_ray_dist_sim3")
    wrap("opt_pose_calib_sim3")


def instrument_factor_graph(factor_graph: FactorGraph, profiler: PipelineProfiler):
    def wrap(attr_name: str):
        method = getattr(factor_graph, attr_name)
        if not hasattr(method, "__func__"):
            return
        fn = method.__func__

        def wrapped(self, *args, **kwargs):
            sync_cuda()
            start = time.perf_counter()
            out = fn(self, *args, **kwargs)
            sync_cuda()
            profiler.record("backend_optimization", time.perf_counter() - start)
            return out

        setattr(factor_graph, attr_name, types.MethodType(wrapped, factor_graph))

    wrap("solve_GN_rays")
    wrap("solve_GN_calib")


def process_backend_tasks(
    pending: Deque[int],
    keyframes: LocalKeyframes,
    factor_graph: FactorGraph,
    retrieval_db,
    profiler: PipelineProfiler,
):
    if not pending:
        return

    local_opt_cfg = config["local_opt"]
    retrieval_cfg = config["retrieval"]
    n_consec = 1

    while pending:
        idx = pending.popleft()
        frame = keyframes[idx]

        sync_cuda()
        start = time.perf_counter()

        kf_idx = []
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)

        retrieval_inds = retrieval_db.update(
            frame,
            add_after_query=True,
            k=retrieval_cfg["k"],
            min_thresh=retrieval_cfg["min_thresh"],
        )
        kf_idx.extend(retrieval_inds)

        unique_inds = set(kf_idx)
        if idx in unique_inds:
            unique_inds.remove(idx)

        if unique_inds:
            ii = list(unique_inds)
            jj = [idx] * len(ii)
            factor_graph.add_factors(ii, jj, local_opt_cfg["min_match_frac"])

        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

        sync_cuda()
        profiler.record("backend_total", time.perf_counter() - start)


def load_calibration(calib_path: str, dataset, device: str):
    with open(calib_path, "r") as f:
        intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
    dataset.camera_intrinsics = Intrinsics.from_calib(
        dataset.img_size,
        intrinsics["width"],
        intrinsics["height"],
        intrinsics["calibration"],
    )
    dataset.use_calibration = True
    K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
        device, dtype=torch.float32
    )
    config["use_calib"] = True
    return K


def main():
    parser = argparse.ArgumentParser(
        description="Profile MASt3R-SLAM pipeline components."
    )
    parser.add_argument(
        "--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk"
    )
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--calib", default="")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--no-backend", action="store_true")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    load_config(args.config)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dataset = load_dataset(args.dataset)
    dataset.subsample(config["dataset"]["subsample"])

    if args.calib:
        K = load_calibration(args.calib, dataset, device)
    elif config["use_calib"] and dataset.has_calib():
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            device, dtype=torch.float32
        )
    else:
        K = None

    profiler = PipelineProfiler()

    model = load_mast3r(device=device, use_fp8=args.fp8)
    instrument_model(model, profiler)
    instrument_matching(profiler)

    keyframes = LocalKeyframes(device=device)
    tracker = FrameTracker(model, keyframes, device)
    instrument_tracker(tracker, profiler)

    retrieval_db = load_retriever(model, device=device)
    factor_graph = FactorGraph(model, keyframes, K, device)
    instrument_factor_graph(factor_graph, profiler)

    if K is not None:
        keyframes.set_intrinsics(K)

    pending_backend: Deque[int] = collections.deque()

    total_timer_start = time.perf_counter()
    current_pose = lietorch.Sim3.Identity(1, device=device)

    max_frames = args.max_frames if args.max_frames > 0 else len(dataset)

    for idx in range(max_frames):
        timestamp, img = dataset[idx]

        profiler.start_frame(idx)
        sync_cuda()
        frame_start = time.perf_counter()

        sync_cuda()
        create_start = time.perf_counter()
        frame = create_frame(
            idx,
            img,
            current_pose,
            img_size=dataset.img_size,
            device=device,
            use_fp16=args.fp8,
        )
        if K is not None:
            frame.K = K
        sync_cuda()
        profiler.record("frame_creation", time.perf_counter() - create_start)

        if len(keyframes) == 0:
            sync_cuda()
            init_start = time.perf_counter()
            X_init, C_init = mast3r_inference_mono(model, frame)
            sync_cuda()
            profiler.record("initialization", time.perf_counter() - init_start)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)
            current_pose = frame.T_WC
            if not args.no_backend:
                pending_backend.append(len(keyframes) - 1)
        else:
            sync_cuda()
            track_start = time.perf_counter()
            new_kf, _, try_reloc = tracker.track(frame)
            sync_cuda()
            profiler.record("tracking_total", time.perf_counter() - track_start)

            if try_reloc:
                sync_cuda()
                reloc_start = time.perf_counter()
                X_rel, C_rel = mast3r_inference_mono(model, frame)
                sync_cuda()
                profiler.record("relocalization", time.perf_counter() - reloc_start)
                frame.update_pointmap(X_rel, C_rel)

            if new_kf:
                keyframes.append(frame)
                if not args.no_backend:
                    pending_backend.append(len(keyframes) - 1)

            current_pose = frame.T_WC

        sync_cuda()
        profiler.record("frame_total", time.perf_counter() - frame_start)
        profiler.end_frame(idx)

        if not args.no_backend:
            process_backend_tasks(
                pending_backend, keyframes, factor_graph, retrieval_db, profiler
            )

    total_elapsed = time.perf_counter() - total_timer_start

    print("\n=== Run Summary ===")
    print(f"Frames processed : {profiler.num_frames}")
    if total_elapsed > 0 and profiler.num_frames > 0:
        fps = profiler.num_frames / total_elapsed
        print(f"Average FPS     : {fps:.2f}")
    print(f"Total runtime   : {total_elapsed:.3f}s")

    profiler.summary()


if __name__ == "__main__":
    main()
