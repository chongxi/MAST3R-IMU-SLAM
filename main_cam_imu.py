#!/usr/bin/env python3
"""
Live camera runner with IMU monitoring and frame-rate diagnostics.

This script mirrors the behaviour of `main_cam.py` but spawns a dedicated
process that continuously reads IMU packets from a serial port. The main
loop records both frame processing intervals and the number of IMU samples
received between frames, giving us the raw data needed to align IMU and
visual pipelines before deeper integration.
"""

from __future__ import annotations

import argparse
import queue as pyqueue
import statistics
import sys
import time
import math
from collections import deque
from typing import Deque, Dict, List, Optional

import cv2
import lietorch
import numpy as np
import serial
import torch
import torch.multiprocessing as mp
import yaml

from main import run_backend
from mast3r_slam.config import config, load_config
from mast3r_slam.dataloader import Intrinsics
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import load_mast3r, mast3r_inference_mono
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.lietorch_utils import yaw_from_sim3
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.visualization import WindowMsg, run_visualization

IMU_CALIBRATION_SAMPLES = 15


# ---------------------------------------------------------------------------
# Camera helpers (duplicated from main_cam.py to keep this file standalone)
# ---------------------------------------------------------------------------

def parse_camera_source(src: str):
    try:
        return int(src)
    except ValueError:
        return src


def open_camera(args):
    source = parse_camera_source(args.camera)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera source {args.camera}")

    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps > 0:
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    return cap


def grab_frame(cap, use_calib=False, intrinsics: Intrinsics | None = None):
    ret, frame_bgr = cap.read()
    if not ret:
        return None
    if use_calib and intrinsics is not None:
        frame_bgr = intrinsics.remap(frame_bgr)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# IMU reader process
# ---------------------------------------------------------------------------

def parse_imu_packet(line: bytes) -> Optional[Dict[str, float]]:
    try:
        text = line.decode("utf-8").strip()
    except UnicodeDecodeError:
        return None
    if not text:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        return None
    try:
        vx, vy, yaw = (float(v) for v in parts)
    except ValueError:
        return None
    return {"vx": vx, "vy": vy, "yaw": yaw}


def imu_reader_process(port: str, baudrate: int, queue: mp.Queue, stop_event: mp.Event):
    ser = None
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        queue.put({"type": "status", "status": "opened", "port": port})
        last_time = None
        while not stop_event.is_set():
            line = ser.readline()
            now = time.perf_counter()
            packet = parse_imu_packet(line)
            if packet is None:
                continue
            dt = None if last_time is None else now - last_time
            last_time = now
            packet.update({"timestamp": now, "dt": dt})
            message = {"type": "imu", "data": packet}
            try:
                queue.put_nowait(message)
            except pyqueue.Full:
                try:
                    queue.get_nowait()
                except pyqueue.Empty:
                    pass
                try:
                    queue.put_nowait(message)
                except pyqueue.Full:
                    pass
    except serial.SerialException as exc:
        queue.put({"type": "error", "message": str(exc)})
    finally:
        if ser is not None:
            ser.close()
        queue.put({"type": "status", "status": "closed", "port": port})


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main():
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description="Live camera runner with IMU logging")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--camera", default="/dev/video4", help="Camera index or path")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--calib", default="")
    parser.add_argument("--img-size", type=int, choices=(224, 512), default=224)
    parser.add_argument("--fp8", action="store_true", help="Enable FP8 acceleration")
    parser.add_argument("--log-interval", type=int, default=30, help="Frames between log prints")
    parser.add_argument("--frame-window", type=int, default=120, help="Window for averaging frame intervals")
    parser.add_argument("--imu-port", default="/dev/ttyACM0", help="Serial device for IMU data")
    parser.add_argument("--imu-baudrate", type=int, default=115200, help="IMU serial baudrate")
    parser.add_argument("--imu-queue-size", type=int, default=1000, help="Max IMU samples buffered between frames")

    args = parser.parse_args()

    load_config(args.config)
    if config["use_calib"] and not args.calib:
        print("[Error] Calibration enabled in config but no --calib provided.")
        sys.exit(1)

    subsample = max(1, int(config["dataset"].get("subsample", 1)))

    try:
        cap = open_camera(args)
    except RuntimeError as err:
        print(err)
        sys.exit(1)

    first_frame = grab_frame(cap, use_calib=False)
    if first_frame is None:
        print("Failed to capture initial frame")
        cap.release()
        sys.exit(1)

    camera_intrinsics = None
    if args.calib:
        with open(args.calib, "r") as f:
            intrinsics_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        config["use_calib"] = True
        camera_intrinsics = Intrinsics.from_calib(
            args.img_size,
            intrinsics_cfg["width"],
            intrinsics_cfg["height"],
            intrinsics_cfg["calibration"],
        )
        first_frame = grab_frame(cap, use_calib=True, intrinsics=camera_intrinsics)
        if first_frame is None:
            print("Failed to capture frame with calibration applied")
            cap.release()
            sys.exit(1)

    temp_frame = create_frame(
        0,
        first_frame,
        lietorch.Sim3.Identity(1, device=args.device),
        img_size=args.img_size,
        device=args.device,
        use_fp16=args.fp8,
    )
    h, w = temp_frame.img.shape[-2:]

    manager = mp.Manager()
    main2viz = new_queue(manager, args.no_viz)
    viz2main = new_queue(manager, args.no_viz)

    keyframes = SharedKeyframes(manager, h, w, device=args.device)
    states = SharedStates(manager, h, w, device=args.device)

    if config["use_calib"] and camera_intrinsics is not None:
        K = torch.from_numpy(camera_intrinsics.K_frame).to(
            args.device, dtype=torch.float32
        )
        keyframes.set_intrinsics(K)
    else:
        K = None

    if not args.no_viz:
        viz = mp.Process(
            target=run_visualization,
            args=(config, states, keyframes, main2viz, viz2main),
        )
        viz.start()
    else:
        viz = None

    model = load_mast3r(device=args.device, use_fp8=args.fp8)
    model.share_memory()

    tracker = FrameTracker(model, keyframes, args.device)
    last_msg = WindowMsg()

    backend = mp.Process(
        target=run_backend,
        args=(config, model, states, keyframes, K, args.fp8),
    )
    backend.start()

    # Start IMU reader process
    imu_queue: mp.Queue = mp.Queue(maxsize=args.imu_queue_size)
    imu_stop = mp.Event()
    imu_process = mp.Process(
        target=imu_reader_process,
        args=(args.imu_port, args.imu_baudrate, imu_queue, imu_stop),
        daemon=True,
    )
    imu_process.start()

    imu_since_last_frame: List[Dict[str, float]] = []
    imu_dt_window: Deque[float] = deque(maxlen=args.frame_window * 4)
    last_imu_sample: Optional[Dict[str, float]] = None
    last_frame_imu_count = 0
    last_frame_imu_avg_dt = float("nan")
    imu_stream_ok = True

    raw_idx = -1
    processed_idx = 0
    fps_timer = time.time()
    frame_last_ts: Optional[float] = None
    frame_intervals: Deque[float] = deque(maxlen=args.frame_window)
    pending_frame = first_frame

    try:
        while True:
            # Drain IMU queue each loop to keep buffers small
            try:
                while True:
                    message = imu_queue.get_nowait()
                    mtype = message.get("type")
                    if mtype == "imu":
                        sample = message["data"]
                        imu_since_last_frame.append(sample)
                        yaw_deg = sample.get("yaw")
                        if yaw_deg is not None:
                            states.set_imu_yaw(math.radians(yaw_deg), timestamp=sample.get("timestamp"))
                    elif mtype == "error":
                        if imu_stream_ok:
                            print(f"[IMU] Error: {message['message']}")
                        imu_stream_ok = False
                    elif mtype == "status":
                        status = message.get("status")
                        port = message.get("port", args.imu_port)
                        if status == "opened":
                            print(f"[IMU] Connected to {port} at {args.imu_baudrate} baud")
                        elif status == "closed":
                            print(f"[IMU] Closed connection to {port}")
            except pyqueue.Empty:
                pass

            mode = states.get_mode()
            msg = try_get_msg(viz2main)
            if msg is not None:
                last_msg = msg

            if last_msg.is_terminated:
                states.set_mode(Mode.TERMINATED)
                break

            if last_msg.is_paused and not last_msg.next:
                states.pause()
                time.sleep(0.01)
                continue

            if not last_msg.is_paused:
                states.unpause()

            if pending_frame is not None:
                img_np = pending_frame
                pending_frame = None
            else:
                frame_np = grab_frame(cap, config["use_calib"], camera_intrinsics)
                if frame_np is None:
                    print("Failed to capture frame from camera")
                    time.sleep(0.05)
                    continue
                img_np = frame_np

            raw_idx += 1
            if subsample > 1 and (raw_idx % subsample) != 0:
                continue

            if processed_idx > 0:
                T_WC = states.get_frame().T_WC
            else:
                T_WC = lietorch.Sim3.Identity(1, device=args.device)

            frame = create_frame(
                processed_idx,
                img_np,
                T_WC,
                img_size=args.img_size,
                device=args.device,
                use_fp16=args.fp8,
            )
            if K is not None:
                frame.K = K

            frame_now = time.perf_counter()
            if frame_last_ts is not None:
                dt_frame = frame_now - frame_last_ts
                frame_intervals.append(dt_frame)
            else:
                dt_frame = None
            frame_last_ts = frame_now

            if mode == Mode.INIT:
                X_init, C_init = mast3r_inference_mono(model, frame)
                frame.update_pointmap(X_init, C_init)
                keyframes.append(frame)
                states.queue_global_optimization(len(keyframes) - 1)
                states.set_mode(Mode.TRACKING)
                states.set_frame(frame)
                processed_idx += 1
            elif mode == Mode.TRACKING:
                add_new_kf, _, try_reloc = tracker.track(frame)
                if not try_reloc:
                    camera_yaw = yaw_from_sim3(frame.T_WC)
                    if states.record_imu_calibration(camera_yaw, required_samples=IMU_CALIBRATION_SAMPLES):
                        bias = states.get_imu_yaw_bias()
                        if bias is not None:
                            print(f"[IMU] Yaw calibration locked (bias {math.degrees(bias):.1f} deg).")
                if try_reloc:
                    states.set_mode(Mode.RELOC)
                states.set_frame(frame)

                if add_new_kf:
                    keyframes.append(frame)
                    states.queue_global_optimization(len(keyframes) - 1)
                    while config["single_thread"]:
                        with states.lock:
                            if len(states.global_optimizer_tasks) == 0:
                                break
                        time.sleep(0.01)
                processed_idx += 1
            elif mode == Mode.RELOC:
                X, C = mast3r_inference_mono(model, frame)
                frame.update_pointmap(X, C)
                states.set_frame(frame)
                states.queue_reloc()
                while config["single_thread"]:
                    with states.lock:
                        if states.reloc_sem.value == 0:
                            break
                    time.sleep(0.01)
                processed_idx += 1
            else:
                raise RuntimeError("Invalid tracking mode")

            imu_samples_this_frame = imu_since_last_frame
            imu_since_last_frame = []
            last_frame_imu_count = len(imu_samples_this_frame)
            imu_dt_samples = [sample["dt"] for sample in imu_samples_this_frame if sample.get("dt")]
            if imu_dt_samples:
                last_frame_imu_avg_dt = statistics.mean(imu_dt_samples)
                imu_dt_window.extend(imu_dt_samples)
                last_imu_sample = imu_samples_this_frame[-1]
            else:
                last_frame_imu_avg_dt = float("nan")

            should_log = (
                processed_idx > 0 and args.log_interval > 0 and processed_idx % args.log_interval == 0
            )
            if should_log:
                avg_dt = statistics.mean(frame_intervals) if frame_intervals else float("nan")
                inst_freq = 1.0 / dt_frame if dt_frame and dt_frame > 0 else float("nan")
                avg_freq = 1.0 / avg_dt if avg_dt and avg_dt > 0 else float("nan")

                imu_window_avg_dt = statistics.mean(imu_dt_window) if imu_dt_window else float("nan")
                imu_window_avg_freq = (
                    1.0 / imu_window_avg_dt if imu_window_avg_dt and imu_window_avg_dt > 0 else float("nan")
                )
                imu_frame_avg_freq = (
                    1.0 / last_frame_imu_avg_dt
                    if last_frame_imu_avg_dt == last_frame_imu_avg_dt and last_frame_imu_avg_dt > 0
                    else float("nan")
                )

                print(
                    f"Frame {processed_idx:05d} | dt={dt_frame or 0:.4f}s | inst={inst_freq:7.2f}Hz | "
                    f"avg={avg_freq:7.2f}Hz | imu_samples={last_frame_imu_count:3d} | "
                    f"imu_frame_avg={imu_frame_avg_freq:7.2f}Hz | imu_avg={imu_window_avg_freq:7.2f}Hz | "
                    f"imu_latest={last_imu_sample}"
                )

            if processed_idx % 30 == 0 and processed_idx > 0:
                FPS = processed_idx / (time.time() - fps_timer)
                print(f"[Stats] Processed FPS (since start): {FPS:.2f}")

    except KeyboardInterrupt:
        print("Interrupted by user, shutting down.")
        states.set_mode(Mode.TERMINATED)
    finally:
        cap.release()
        imu_stop.set()
        if imu_process.is_alive():
            imu_process.join(timeout=2.0)
        imu_queue.close()
        imu_queue.join_thread()

    backend.join()
    if viz is not None:
        viz.join()


if __name__ == "__main__":
    main()
