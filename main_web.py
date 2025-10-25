import argparse
import datetime
import math
import pathlib
import sys
import time

import cv2
import numpy as np
import lietorch
import torch
import tqdm
import yaml
import torch.multiprocessing as mp

from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.dataloader import Intrinsics, load_dataset
import mast3r_slam.evaluate as eval
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.lietorch_utils import yaw_from_sim3
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.visualization import WindowMsg
from mast3r_slam.web_visualization import run_web_visualization

IMU_CALIBRATION_SAMPLES = 15


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


def relocalization(frame, keyframes, factor_graph, retrieval_database):
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)
            frame_idx = [n_kf - 1] * len(kf_idx)
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("Success! Relocalized")
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                print("Failed to relocalize")

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


def build_pose_with_imu_yaw(pose_data: torch.Tensor, yaw_rad: float | None) -> lietorch.Sim3:
    """Return a Sim3 pose that preserves translation/scale and applies the given yaw."""
    updated = pose_data.clone()
    if yaw_rad is not None:
        half_yaw = yaw_rad * 0.5
        updated[..., 3:7] = updated.new_tensor([0.0, math.sin(half_yaw), 0.0, math.cos(half_yaw)])
    return lietorch.Sim3(updated)


def parse_imu_packet(line: bytes):
    """Parse an IMU packet of the form "vx,vy,yaw"."""
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


def seed_anchor_frames(anchor_dir, model, keyframes, states, *, img_size, device, use_fp16=False, K=None):
    if not anchor_dir:
        return 0
    anchor_path = pathlib.Path(anchor_dir)
    if not anchor_path.exists():
        print(f"[Anchors] Directory {anchor_dir} does not exist; skipping.")
        return 0
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files = []
    for pattern in patterns:
        files.extend(anchor_path.glob(pattern))
    files = sorted({f.resolve() for f in files if f.is_file()})
    if not files:
        print(f"[Anchors] No image files found in {anchor_dir}; skipping.")
        return 0
    capacity = getattr(keyframes, 'buffer', len(files)) - len(keyframes)
    if capacity <= 0:
        print('[Anchors] Keyframe store is full; skipping anchors.')
        return 0
    if len(files) > capacity:
        print(f"[Anchors] Only {capacity} anchors will be loaded (capacity reached).")
        files = files[:capacity]
    loaded = 0
    for image_path in files:
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            print(f"[Anchors] Failed to read {image_path}; skipping.")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frame_id = -(len(keyframes) + 1)
        frame = create_frame(
            frame_id,
            img_rgb,
            lietorch.Sim3.Identity(1, device=device),
            img_size=img_size,
            device=device,
            use_fp16=use_fp16,
        )
        if K is not None:
            frame.K = K
        try:
            X_anchor, C_anchor = mast3r_inference_mono(model, frame)
        except Exception as exc:  # noqa: BLE001
            print(f"[Anchors] Inference failed for {image_path}: {exc}")
            continue
        frame.update_pointmap(X_anchor, C_anchor)
        keyframes.append(frame)
        states.queue_global_optimization(len(keyframes) - 1)
        loaded += 1
    if loaded > 0:
        print(f"[Anchors] Loaded {loaded} anchor frames from {anchor_dir}.")
    else:
        print(f"[Anchors] No anchors were loaded from {anchor_dir}.")
    return loaded


def imu_reader_process(device: str, baudrate: int, states, stop_event):
    """Background worker that streams yaw readings from a serial IMU."""
    import math
    import time
    try:
        import serial
    except ImportError:
        print("[IMU] pyserial not installed; cannot open IMU device.")
        return

    try:
        ser = serial.Serial(device, baudrate, timeout=1)
    except Exception as exc:  # noqa: BLE001 broad but we only log
        print(f"[IMU] Failed to open {device}: {exc}")
        return

    print(f"[IMU] Reading IMU yaw from {device} at {baudrate} baud")
    last_log = 0.0
    try:
        while not stop_event.is_set():
            line = ser.readline()
            if not line:
                continue
            packet = parse_imu_packet(line)
            if packet is None:
                continue
            yaw_deg = packet["yaw"]
            yaw_rad = math.radians(yaw_deg)
            states.set_imu_yaw(yaw_rad)
            now = time.time()
            if now - last_log >= 0.5:
                label = "raw"
                yaw_display = yaw_deg
                if states.is_imu_yaw_calibrated():
                    corrected, _ = states.get_imu_yaw(corrected=True)
                    if corrected is not None:
                        yaw_display = math.degrees(corrected)
                        label = "calibrated"
                print(f"[IMU] yaw ({label}) = {yaw_display:.2f} deg")
                last_log = now
    finally:
        try:
            ser.close()
        except Exception:
            pass
        print(f"[IMU] IMU device {device} closed")


def run_backend(cfg, model, states, keyframes, K, use_fp8=False):
    set_global_config(cfg)

    if use_fp8:
        import mast3r_slam.mast3r_utils as mutils

        mutils.USE_FP8 = True

    device = keyframes.device
    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)

    mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        mode = states.get_mode()
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue
        if mode == Mode.RELOC:
            frame = states.get_frame()
            success = relocalization(frame, keyframes, factor_graph, retrieval_database)
            if success:
                states.set_mode(Mode.TRACKING)
            states.dequeue_reloc()
            continue
        idx = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue

        kf_idx = []
        n_consec = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)
        frame = keyframes[idx]
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds

        lc_inds = set(retrieval_inds)
        lc_inds.discard(idx - 1)
        if len(lc_inds) > 0:
            print("Database retrieval", idx, ": ", lc_inds)

        kf_idx = set(kf_idx)
        kf_idx.discard(idx)
        kf_idx = list(kf_idx)
        frame_idx = [idx] * len(kf_idx)
        if kf_idx:
            factor_graph.add_factors(
                kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
            )

        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()

        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks.pop(0)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)
    device = "cuda:0"
    save_frames = False
    datetime_now = str(datetime.datetime.now()).replace(" ", "_")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--save-as", default="default")
    parser.add_argument("--calib", default="")
    parser.add_argument("--fp8", action="store_true", help="Enable FP8 acceleration (4x speedup on Thor GPU)")
    parser.add_argument("--no-web", action="store_true", help="Disable the web visualization.")
    parser.add_argument("--web-host", default="0.0.0.0", help="Host/IP for the web viewer server.")
    parser.add_argument("--web-port", type=int, default=7860, help="Port for the web viewer server.")
    parser.add_argument("--web-stride", type=int, default=2, help="Pixel stride for the current frame surfels.")
    parser.add_argument("--camera", default="", help="Camera index or path (set to enable live capture).")
    parser.add_argument("--width", type=int, default=640, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=480, help="Requested camera height.")
    parser.add_argument("--fps", type=float, default=30.0, help="Requested camera FPS.")
    parser.add_argument("--img-size", type=int, choices=(224, 512), default=224, help="Resize dimension for camera input.")
    parser.add_argument(
        "--web-keyframe-stride",
        type=int,
        default=8,
        help="Pixel stride for keyframe surfels.",
    )
    parser.add_argument("--imu_dev", default="", help="Serial device for IMU data (e.g. /dev/ttyACM0).")
    parser.add_argument("--imu_baud", type=int, default=115200, help="IMU serial baud rate.")
    parser.add_argument("--anchor_dir", default="", help="Optional directory of anchor images to seed into the retriever.")

    args = parser.parse_args()

    load_config(args.config)
    print(args.dataset)
    print(config)

    manager = mp.Manager()
    main2viz = new_queue(manager, args.no_web)
    viz2main = new_queue(manager, args.no_web)

    use_camera = bool(args.camera)
    cap = None
    camera_intrinsics = None
    dataset = None

    if use_camera:
        if config["use_calib"] and not args.calib:
            print("[Error] Calibration enabled in config but no --calib provided.")
            sys.exit(1)
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
            lietorch.Sim3.Identity(1, device=device),
            img_size=args.img_size,
            device=device,
            use_fp16=args.fp8,
        )
        h, w = temp_frame.img.shape[-2:]
        keyframes = SharedKeyframes(manager, h, w)
        states = SharedStates(manager, h, w)
        if config["use_calib"] and camera_intrinsics is not None:
            K = torch.from_numpy(camera_intrinsics.K_frame).to(
                device, dtype=torch.float32
            )
            keyframes.set_intrinsics(K)
        else:
            K = None
        anchor_img_size = args.img_size
    else:
        dataset = load_dataset(args.dataset)
        dataset.subsample(config["dataset"]["subsample"])
        h, w = dataset.get_img_shape()[0]
        if args.calib:
            with open(args.calib, "r") as f:
                intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
            config["use_calib"] = True
            dataset.use_calibration = True
            dataset.camera_intrinsics = Intrinsics.from_calib(
                dataset.img_size,
                intrinsics["width"],
                intrinsics["height"],
                intrinsics["calibration"],
            )
        keyframes = SharedKeyframes(manager, h, w)
        states = SharedStates(manager, h, w)
        has_calib = dataset.has_calib()
        use_calib = config["use_calib"]
        if use_calib and not has_calib:
            print("[Warning] No calibration provided for this dataset!")
            sys.exit(0)
        K = None
        if use_calib:
            K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
                device, dtype=torch.float32
            )
            keyframes.set_intrinsics(K)
        anchor_img_size = dataset.img_size

    web_viz = None
    if not args.no_web:
        web_viz = mp.Process(
            target=run_web_visualization,
            args=(config, states, keyframes, main2viz, viz2main),
            kwargs={
                "host": args.web_host,
                "port": args.web_port,
                "conf_threshold": config["tracking"]["C_conf"],
                "current_stride": args.web_stride,
                "keyframe_stride": args.web_keyframe_stride,
            },
        )
        web_viz.start()

    model = load_mast3r(device=device, use_fp8=args.fp8)
    model.share_memory()

    use_calib = config["use_calib"]
    seed_anchor_frames(
        args.anchor_dir,
        model,
        keyframes,
        states,
        img_size=anchor_img_size,
        device=device,
        use_fp16=args.fp8,
        K=K if use_calib else None,
    )

    if use_camera:
        tracker = FrameTracker(model, keyframes, device)
        last_msg = WindowMsg()

        backend = mp.Process(target=run_backend, args=(config, model, states, keyframes, K, args.fp8))
        backend.start()

        imu_stop = None
        imu_process = None
        if args.imu_dev:
            imu_stop = mp.Event()
            imu_process = mp.Process(
                target=imu_reader_process,
                args=(args.imu_dev, args.imu_baud, states, imu_stop),
                daemon=True,
            )
            imu_process.start()

        processed_idx = 0
        fps_timer = time.time()

        try:
            while True:
                mode = states.get_mode()
                msg = try_get_msg(viz2main)
                last_msg = msg if msg is not None else last_msg
                if last_msg.is_terminated:
                    states.set_mode(Mode.TERMINATED)
                    break

                if last_msg.is_paused and not last_msg.next:
                    states.pause()
                    time.sleep(0.01)
                    continue

                if not last_msg.is_paused:
                    states.unpause()

                frame_np = grab_frame(cap, config["use_calib"], camera_intrinsics)
                if frame_np is None:
                    print("Failed to capture frame from camera")
                    time.sleep(0.05)
                    continue

                T_WC = (
                    lietorch.Sim3.Identity(1, device=device)
                    if processed_idx == 0
                    else states.get_frame().T_WC
                )
                frame = create_frame(
                    processed_idx,
                    frame_np,
                    T_WC,
                    img_size=args.img_size,
                    device=device,
                    use_fp16=args.fp8,
                )
                if K is not None:
                    frame.K = K

                add_new_kf = False

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
                    prev_pose_data = frame.T_WC.data.clone()
                    add_new_kf, match_info, try_reloc = tracker.track(frame)
                    if not try_reloc:
                        camera_yaw = yaw_from_sim3(frame.T_WC)
                        if states.record_imu_calibration(camera_yaw, required_samples=IMU_CALIBRATION_SAMPLES):
                            bias = states.get_imu_yaw_bias()
                            if bias is not None:
                                print(f"[IMU] Yaw calibration locked (bias {math.degrees(bias):.1f} deg).")
                    else:
                        imu_yaw, _ = states.get_imu_yaw(corrected=True)
                        calibrated = states.is_imu_yaw_calibrated()
                        fallback_pose = build_pose_with_imu_yaw(prev_pose_data, imu_yaw)
                        if imu_yaw is not None:
                            qualifier = 'calibrated' if calibrated else 'uncalibrated'
                            print(f"[IMU] Visual tracking lost; applying {qualifier} yaw {math.degrees(imu_yaw):.1f} deg.")
                        else:
                            print("[IMU] Visual tracking lost; reusing previous pose (no IMU yaw).")
                        frame.T_WC = fallback_pose
                        add_new_kf = False
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
                    raise Exception("Invalid mode")

                if add_new_kf:
                    keyframes.append(frame)
                    states.queue_global_optimization(len(keyframes) - 1)
                    while config["single_thread"]:
                        with states.lock:
                            if len(states.global_optimizer_tasks) == 0:
                                break
                        time.sleep(0.01)

                if processed_idx % 30 == 0 and processed_idx > 0:
                    fps = processed_idx / (time.time() - fps_timer)
                    print(f"FPS: {fps}")
                processed_idx += 1
        except KeyboardInterrupt:
            print("Interrupted by user, shutting down camera loop.")
            states.set_mode(Mode.TERMINATED)
        finally:
            if cap is not None:
                cap.release()

        if imu_stop is not None:
            imu_stop.set()
            if imu_process is not None and imu_process.is_alive():
                imu_process.join(timeout=2.0)

        print("done")
        backend.join()
        if web_viz is not None:
            web_viz.join()
        sys.exit(0)

    tracker = FrameTracker(model, keyframes, device)
    last_msg = WindowMsg()

    backend = mp.Process(target=run_backend, args=(config, model, states, keyframes, K, args.fp8))
    backend.start()

    imu_stop = None
    imu_process = None
    if args.imu_dev:
        imu_stop = mp.Event()
        imu_process = mp.Process(
            target=imu_reader_process,
            args=(args.imu_dev, args.imu_baud, states, imu_stop),
            daemon=True,
        )
        imu_process.start()

    i = 0
    fps_timer = time.time()
    frames = []

    try:
        while True:
            mode = states.get_mode()
            msg = try_get_msg(viz2main)
            last_msg = msg if msg is not None else last_msg
            if last_msg.is_terminated:
                states.set_mode(Mode.TERMINATED)
                break

            if last_msg.is_paused and not last_msg.next:
                states.pause()
                time.sleep(0.01)
                continue

            if not last_msg.is_paused:
                states.unpause()

            if i == len(dataset):
                states.set_mode(Mode.TERMINATED)
                break

            timestamp, img = dataset[i]
            if save_frames:
                frames.append(img)

            T_WC = (
                lietorch.Sim3.Identity(1, device=device)
                if i == 0
                else states.get_frame().T_WC
            )
            frame = create_frame(
                i,
                img,
                T_WC,
                img_size=dataset.img_size,
                device=device,
                use_fp16=args.fp8,
            )

            add_new_kf = False

            if mode == Mode.INIT:
                X_init, C_init = mast3r_inference_mono(model, frame)
                frame.update_pointmap(X_init, C_init)
                keyframes.append(frame)
                states.queue_global_optimization(len(keyframes) - 1)
                states.set_mode(Mode.TRACKING)
                states.set_frame(frame)
                i += 1
                continue

            if mode == Mode.TRACKING:
                prev_pose_data = frame.T_WC.data.clone()
                add_new_kf, match_info, try_reloc = tracker.track(frame)
                if not try_reloc:
                    camera_yaw = yaw_from_sim3(frame.T_WC)
                    if states.record_imu_calibration(camera_yaw, required_samples=IMU_CALIBRATION_SAMPLES):
                        bias = states.get_imu_yaw_bias()
                        if bias is not None:
                            print(f"[IMU] Yaw calibration locked (bias {math.degrees(bias):.1f} deg).")
                else:
                    imu_yaw, _ = states.get_imu_yaw(corrected=True)
                    calibrated = states.is_imu_yaw_calibrated()
                    fallback_pose = build_pose_with_imu_yaw(prev_pose_data, imu_yaw)
                    if imu_yaw is not None:
                        qualifier = 'calibrated' if calibrated else 'uncalibrated'
                        print(f"[IMU] Visual tracking lost; applying {qualifier} yaw {math.degrees(imu_yaw):.1f} deg.")
                    else:
                        print("[IMU] Visual tracking lost; reusing previous pose (no IMU yaw).")
                    frame.T_WC = fallback_pose
                    add_new_kf = False
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
                raise Exception("Invalid mode")

            if add_new_kf:
                keyframes.append(frame)
                states.queue_global_optimization(len(keyframes) - 1)
                while config["single_thread"]:
                    with states.lock:
                        if len(states.global_optimizer_tasks) == 0:
                            break
                    time.sleep(0.01)

            if i % 30 == 0 and i > 0:
                FPS = i / (time.time() - fps_timer)
                print(f"FPS: {FPS}")
            i += 1
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down frontend loop.")
        states.set_mode(Mode.TERMINATED)
    finally:
        if dataset.save_results:
            save_dir, seq_name = eval.prepare_savedir(args, dataset)
            eval.save_traj(save_dir, f"{seq_name}.txt", dataset.timestamps, keyframes)
            eval.save_reconstruction(
                save_dir,
                f"{seq_name}.ply",
                keyframes,
                last_msg.C_conf_threshold,
            )
            eval.save_keyframes(
                save_dir / "keyframes" / seq_name, dataset.timestamps, keyframes
            )
        if save_frames:
            savedir = pathlib.Path(f"logs/frames/{datetime_now}")
            savedir.mkdir(exist_ok=True, parents=True)
            for i, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
                frame = (frame * 255).clip(0, 255)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{savedir}/{i}.png", frame)

    if imu_stop is not None:
        imu_stop.set()
        if imu_process is not None and imu_process.is_alive():
            imu_process.join(timeout=2.0)

    print("done")
    backend.join()
    if web_viz is not None:
        web_viz.join()



# python main_web.py --camera /dev/video4 --imu_dev /dev/ttyACM0