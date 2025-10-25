#!/usr/bin/env python3
"""
Record IMU velocity/yaw samples from a serial port, integrate a 2D trajectory,
and plot/save the result once the session ends (Ctrl+C).

Adds an initial bias calibration, optional fixed 100 Hz timing, and a small
zero-velocity clamp so the trajectory stays stable when the sensor is still.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import serial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record IMU velocity/yaw data and plot the inferred 2D trajectory.")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port providing (vx, vy, yaw) readings.")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate.")
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional CSV file to store the captured samples (time,vx_raw,vy_raw,vx,vy,yaw,x,y).",
    )
    parser.add_argument(
        "--heading-unit",
        choices=("deg", "rad"),
        default="deg",
        help="Unit of the incoming yaw heading values (default: deg).",
    )
    parser.add_argument(
        "--calib-duration",
        type=float,
        default=2.0,
        help="Seconds to average a stationary bias before integrating (set 0 to disable).",
    )
    parser.add_argument(
        "--expected-rate",
        type=float,
        default=100.0,
        help="Expected IMU update rate in Hz; used for diagnostics and optional fixed dt integration.",
    )
    parser.add_argument(
        "--fixed-dt",
        action="store_true",
        help="Use 1/expected-rate as the integration step instead of measured sample spacing.",
    )
    parser.add_argument(
        "--zero-threshold",
        type=float,
        default=0.02,
        help="Clamp corrected body-frame speed below this (m/s) to zero; set <=0 to disable.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip rendering a plot (useful on headless environments).",
    )
    parser.add_argument(
        "--plot-file",
        type=Path,
        help="If provided, save the trajectory plot to this file instead of (or in addition to) showing it.",
    )
    parser.add_argument(
        "--initial-x",
        type=float,
        default=0.0,
        help="Initial x position in meters.",
    )
    parser.add_argument(
        "--initial-y",
        type=float,
        default=0.0,
        help="Initial y position in meters.",
    )
    return parser.parse_args()


def heading_to_rad(heading: float, unit: str) -> float:
    return heading if unit == "rad" else math.radians(heading)


def integrate_trajectory(
    port: str,
    baud: int,
    heading_unit: str,
    initial_x: float,
    initial_y: float,
    calib_duration: float,
    expected_rate: float,
    use_fixed_dt: bool,
    zero_threshold: float,
):
    try:
        ser = serial.Serial(port, baud, timeout=1)
    except serial.SerialException as exc:
        print(f"Serial error: {exc}")
        sys.exit(1)

    print(f"Recording (vx, vy, yaw) from {port} at {baud} baud.")
    print("Press Ctrl+C to stop.\n")

    expected_dt = 1.0 / expected_rate if expected_rate > 0 else None
    if use_fixed_dt and expected_dt is None:
        print("Expected rate must be >0 for fixed-dt integration; falling back to real dt.")
        use_fixed_dt = False

    samples = []
    x, y = initial_x, initial_y
    prev_ts = None
    start_ts = None
    loop_count = 0
    warned_dt = False

    bias_sum_vx = 0.0
    bias_sum_vy = 0.0
    bias_count = 0
    bias_locked = calib_duration <= 0
    bias_vx = 0.0
    bias_vy = 0.0

    zero_threshold_sq = zero_threshold * zero_threshold if zero_threshold > 0 else None

    meta = {
        "expected_dt": expected_dt,
        "use_fixed_dt": use_fixed_dt,
        "zero_threshold": zero_threshold,
        "calib_duration": max(calib_duration, 0.0),
        "calib_samples": 0,
        "calib_time": 0.0,
        "bias_vx": 0.0,
        "bias_vy": 0.0,
    }

    try:
        while True:
            raw_line = ser.readline()
            if not raw_line:
                continue

            try:
                decoded = raw_line.decode("utf-8").strip()
            except UnicodeDecodeError:
                continue

            if not decoded:
                continue

            parts = [p.strip() for p in decoded.split(",")]
            if len(parts) != 3:
                continue

            try:
                vx_body = float(parts[0])
                vy_body = float(parts[1])
                yaw_heading = float(parts[2])
            except ValueError:
                continue

            ts = time.time()
            loop_count += 1
            if start_ts is None:
                start_ts = ts

            if prev_ts is None:
                delta = expected_dt if use_fixed_dt and expected_dt is not None else 0.0
            else:
                delta = expected_dt if use_fixed_dt and expected_dt is not None else ts - prev_ts

            if (
                not use_fixed_dt
                and expected_dt is not None
                and not warned_dt
                and prev_ts is not None
                and delta > expected_dt * 3
            ):
                print(
                    "Warning: observed {dt_ms:.1f} ms between samples; expected about {exp_ms:.1f} ms.".format(
                        dt_ms=delta * 1000.0,
                        exp_ms=expected_dt * 1000.0,
                    )
                )
                warned_dt = True

            prev_ts = ts

            if not bias_locked:
                bias_sum_vx += vx_body
                bias_sum_vy += vy_body
                bias_count += 1
                elapsed = ts - start_ts
                running_vx = bias_sum_vx / bias_count
                running_vy = bias_sum_vy / bias_count
                if elapsed >= calib_duration:
                    bias_locked = True
                    bias_vx = running_vx
                    bias_vy = running_vy
                    meta.update(
                        {
                            "calib_samples": bias_count,
                            "calib_time": elapsed,
                            "bias_vx": bias_vx,
                            "bias_vy": bias_vy,
                        }
                    )
                    print(
                        "Locked velocity bias at ({vx:.5f}, {vy:.5f}) m/s using {n} samples over {t:.2f}s.".format(
                            vx=bias_vx,
                            vy=bias_vy,
                            n=bias_count,
                            t=elapsed,
                        )
                    )
            else:
                elapsed = ts - start_ts if start_ts is not None else 0.0

            if not bias_locked and bias_count > 0:
                bias_vx = bias_sum_vx / bias_count
                bias_vy = bias_sum_vy / bias_count

            adj_vx = vx_body - bias_vx
            adj_vy = vy_body - bias_vy

            if zero_threshold_sq is not None:
                if adj_vx * adj_vx + adj_vy * adj_vy < zero_threshold_sq:
                    adj_vx = 0.0
                    adj_vy = 0.0

            yaw_rad = heading_to_rad(yaw_heading, heading_unit)
            cos_yaw = math.cos(yaw_rad)
            sin_yaw = math.sin(yaw_rad)
            vx_world = cos_yaw * adj_vx - sin_yaw * adj_vy
            vy_world = sin_yaw * adj_vx + cos_yaw * adj_vy

            x += vx_world * delta
            y += vy_world * delta

            samples.append(
                {
                    "time": ts,
                    "vx_raw": vx_body,
                    "vy_raw": vy_body,
                    "vx": adj_vx,
                    "vy": adj_vy,
                    "yaw": yaw_heading,
                    "x": x,
                    "y": y,
                }
            )

            print(
                "(vx={vx: .3f}, vy={vy: .3f}, yaw={yaw: .3f}) -> (x={x: .3f}, y={y: .3f})".format(
                    vx=adj_vx,
                    vy=adj_vy,
                    yaw=yaw_heading,
                    x=x,
                    y=y,
                )
            )

    except KeyboardInterrupt:
        print("\nStopping capture...")
    finally:
        ser.close()

    if bias_count and not meta["calib_samples"]:
        meta.update(
            {
                "calib_samples": bias_count,
                "calib_time": (prev_ts - start_ts) if (prev_ts and start_ts) else 0.0,
                "bias_vx": bias_vx,
                "bias_vy": bias_vy,
            }
        )

    if start_ts and prev_ts and prev_ts > start_ts:
        duration = prev_ts - start_ts
        meta["capture_time"] = duration
        meta["rate_estimate"] = loop_count / duration
    else:
        meta["capture_time"] = 0.0
        meta["rate_estimate"] = 0.0

    meta["samples"] = loop_count

    return samples, meta


def write_csv(path: Path, samples) -> None:
    if not samples:
        print("No samples captured; nothing to write.")
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ("time", "vx_raw", "vy_raw", "vx", "vy", "yaw", "x", "y")
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)

    print(f"Wrote {len(samples)} samples to {path}")


def plot_trajectory(samples, plot_file: Path | None, show_plot: bool) -> None:
    if not samples:
        print("No samples captured; nothing to plot.")
        return

    xs = [row["x"] for row in samples]
    ys = [row["y"] for row in samples]

    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, marker="o", markersize=2, linewidth=1)
    plt.scatter(xs[0], ys[0], color="green", label="Start")
    plt.scatter(xs[-1], ys[-1], color="red", label="End")
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.title("Integrated IMU Trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    if plot_file:
        plot_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_file, bbox_inches="tight")
        print(f"Saved trajectory plot to {plot_file}")

    if show_plot:
        plt.show()

    plt.close()


def main() -> None:
    args = parse_args()

    samples, meta = integrate_trajectory(
        port=args.port,
        baud=args.baud,
        heading_unit=args.heading_unit,
        initial_x=args.initial_x,
        initial_y=args.initial_y,
        calib_duration=args.calib_duration,
        expected_rate=args.expected_rate,
        use_fixed_dt=args.fixed_dt,
        zero_threshold=args.zero_threshold,
    )

    if args.out:
        write_csv(args.out, samples)

    show_plot = not args.no_plot
    if args.plot_file or show_plot:
        plot_trajectory(samples, args.plot_file, show_plot)

    if samples:
        total_time = samples[-1]["time"] - samples[0]["time"]
        total_dist = math.dist((samples[0]["x"], samples[0]["y"]), (samples[-1]["x"], samples[-1]["y"]))
        print(
            "Captured {n} samples over {t:.1f}s (est. {hz:.1f} Hz). Final position: ({x:.3f}, {y:.3f}), net displacement {d:.3f}m.".format(
                n=len(samples),
                t=total_time,
                hz=meta.get("rate_estimate", 0.0),
                x=samples[-1]["x"],
                y=samples[-1]["y"],
                d=total_dist,
            )
        )
        print(
            "Bias removed: ({bx:.5f}, {by:.5f}) m/s; zero threshold {zt:.3f} m/s; fixed_dt={fd}.".format(
                bx=meta.get("bias_vx", 0.0),
                by=meta.get("bias_vy", 0.0),
                zt=args.zero_threshold,
                fd=args.fixed_dt,
            )
        )
    else:
        print("No samples recorded.")


if __name__ == "__main__":
    main()
