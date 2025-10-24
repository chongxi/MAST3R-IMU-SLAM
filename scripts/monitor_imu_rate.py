#!/usr/bin/env python3
"""
IMU serial monitor that reports sampling intervals and estimated frequency.

This utility keeps the original `read_serial_data.py` script untouched
while providing timing diagnostics needed for IMU/vision fusion planning.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import deque
from typing import Deque, Tuple

import serial


def parse_packet(line: bytes) -> Tuple[float, float, float] | None:
    """Parse a single IMU packet of the form "vx,vy,yaw"."""
    try:
        data_str = line.decode("utf-8").strip()
    except UnicodeDecodeError:
        return None
    if not data_str:
        return None
    parts = [p.strip() for p in data_str.split(",")]
    if len(parts) != 3:
        return None
    try:
        vx, vy, yaw = (float(v) for v in parts)
    except ValueError:
        return None
    return vx, vy, yaw


def monitor(port: str, baudrate: int, window: int) -> None:
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
    except serial.SerialException as exc:
        print(f"Serial error: {exc}", file=sys.stderr)
        print("Hint: ensure the device exists and you have permission (dialout group).")
        return

    intervals: Deque[float] = deque(maxlen=window)
    last_time: float | None = None

    print(f"Monitoring IMU packets from {port} at {baudrate} baud.")
    print("Columns: vx  vy  yaw  dt(s)  freq(Hz)  avg_dt  avg_freq")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            line = ser.readline()
            now = time.perf_counter()
            packet = parse_packet(line)
            if packet is None:
                continue

            vx, vy, yaw = packet
            if last_time is not None:
                dt = now - last_time
                intervals.append(dt)
                avg_dt = statistics.mean(intervals)
                freq = 1.0 / dt if dt > 0.0 else float("inf")
                avg_freq = 1.0 / avg_dt if avg_dt > 0.0 else float("inf")
                print(
                    f"(vx={vx:7.4f}, vy={vy:7.4f}, yaw={yaw:7.4f})  "
                    f"dt={dt:6.4f}s  freq={freq:6.1f}Hz  "
                    f"avg_dt={avg_dt:6.4f}s  avg_freq={avg_freq:6.1f}Hz"
                )
            else:
                print(f"(vx={vx:7.4f}, vy={vy:7.4f}, yaw={yaw:7.4f})  dt=--      freq=--")

            last_time = now
    except KeyboardInterrupt:
        print("\nStopping monitor...")
    finally:
        ser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor IMU serial packet timing.")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial device path")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate")
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="Rolling window size for average frequency (in packets)",
    )
    args = parser.parse_args()

    if args.window <= 1:
        print("Window size must be greater than 1 for averaging.", file=sys.stderr)
        sys.exit(1)

    monitor(args.port, args.baudrate, args.window)


if __name__ == "__main__":
    main()
