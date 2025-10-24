import argparse
import sys
import time

import cv2
import lietorch
import numpy as np
import torch
import torch.multiprocessing as mp
import yaml

from main import run_backend
from mast3r_slam.config import config, load_config
from mast3r_slam.dataloader import Intrinsics
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import load_mast3r, mast3r_inference_mono
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.visualization import WindowMsg, run_visualization


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


def main():
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser()
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

    raw_idx = -1
    processed_idx = 0
    fps_timer = time.time()
    pending_frame = first_frame

    try:
        while True:
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

            if mode == Mode.INIT:
                X_init, C_init = mast3r_inference_mono(model, frame)
                frame.update_pointmap(X_init, C_init)
                keyframes.append(frame)
                states.queue_global_optimization(len(keyframes) - 1)
                states.set_mode(Mode.TRACKING)
                states.set_frame(frame)
                processed_idx += 1
                continue

            if mode == Mode.TRACKING:
                add_new_kf, _, try_reloc = tracker.track(frame)
                if try_reloc:
                    states.set_mode(Mode.RELOC)
                states.set_frame(frame)

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
            else:
                raise RuntimeError("Invalid tracking mode")

            if mode == Mode.TRACKING and add_new_kf:
                keyframes.append(frame)
                states.queue_global_optimization(len(keyframes) - 1)
                while config["single_thread"]:
                    with states.lock:
                        if len(states.global_optimizer_tasks) == 0:
                            break
                    time.sleep(0.01)

            if processed_idx % 30 == 0 and processed_idx > 0:
                fps = processed_idx / (time.time() - fps_timer)
                print(f"Processed FPS: {fps:.2f}")
            processed_idx += 1
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down.")
        states.set_mode(Mode.TERMINATED)
    finally:
        cap.release()

    backend.join()
    if viz is not None:
        viz.join()


if __name__ == "__main__":
    main()
