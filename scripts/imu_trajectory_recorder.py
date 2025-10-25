#!/usr/bin/env python3
"""Read planar velocity + yaw from serial (or simulation) and write IMU trajectory JSON."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import serial  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    serial = None
    SERIAL_IMPORT_ERROR = exc
else:
    SERIAL_IMPORT_ERROR = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    plt = None


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    tmp_path.replace(path)


class PlanarIntegrator:
    def __init__(
        self,
        *,
        max_points: int,
        arrow_length: float,
        vx_bias: float = 0.0,
        vy_bias: float = 0.0,
        zupt_threshold: float = 0.02,
        zupt_samples: int = 5,
        filter_alpha: float = 0.8,
        turn_scale: float = 1.0,
        turn_threshold: float = 10.0,
        yaw_drift_correction: float = 0.0,
    ) -> None:
        self.max_points = max_points
        self.arrow_length = arrow_length
        self.vx_bias = vx_bias
        self.vy_bias = vy_bias
        self.zupt_threshold = zupt_threshold
        self.zupt_samples = zupt_samples
        self.filter_alpha = filter_alpha
        self.turn_scale = turn_scale
        self.turn_threshold = math.radians(turn_threshold)  # Convert to rad/s
        self.yaw_drift_correction = math.radians(yaw_drift_correction)  # deg/s -> rad/s
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._history: list[tuple[float, float]] = []
        self._filtered_vx = 0.0
        self._filtered_vy = 0.0
        self._stationary_count = 0
        self._prev_yaw_rate = 0.0  # Track previous yaw rate to detect turn end
        self._turning = False  # Track if currently in a turn

    def step(self, vx_body: float, vy_body: float, yaw_rad: float, dt: float) -> None:
        dt = max(dt, 0.0)
        if dt == 0.0:
            return
        
        # Calculate yaw rate (angular velocity)
        yaw_rate = abs(yaw_rad - self._yaw) / dt  # rad/s
        
        # Apply yaw drift correction (systematic heading bias)
        yaw_corrected = yaw_rad
        if self.yaw_drift_correction != 0.0:
            # Accumulate correction over time
            yaw_corrected = yaw_rad + self.yaw_drift_correction * dt
        
        # Detect turn end: was turning, now stopped
        was_turning = self._turning
        self._turning = yaw_rate > self.turn_threshold
        
        if was_turning and not self._turning:
            # Just finished a turn - reset filter state to eliminate accumulated errors
            self._filtered_vx = 0.0
            self._filtered_vy = 0.0
        
        # Apply bias correction
        vx_corrected = vx_body - self.vx_bias
        vy_corrected = vy_body - self.vy_bias
        
        # Scale down velocity during turns (sensor is less reliable when rotating)
        if self._turning:
            scale_factor = self.turn_scale
            vx_corrected *= scale_factor
            vy_corrected *= scale_factor
        
        # Zero-velocity update (ZUPT)
        speed = math.sqrt(vx_corrected**2 + vy_corrected**2)
        if speed < self.zupt_threshold:
            self._stationary_count += 1
            if self._stationary_count >= self.zupt_samples:
                vx_corrected = 0.0
                vy_corrected = 0.0
        else:
            self._stationary_count = 0
        
        # Apply low-pass filter
        self._filtered_vx = self.filter_alpha * self._filtered_vx + (1 - self.filter_alpha) * vx_corrected
        self._filtered_vy = self.filter_alpha * self._filtered_vy + (1 - self.filter_alpha) * vy_corrected
        
        # Use midpoint yaw for more accurate rotation during turns
        yaw_mid = (self._yaw + yaw_corrected) / 2.0
        cos_yaw = math.cos(yaw_mid)
        sin_yaw = math.sin(yaw_mid)
        
        # Transform to world frame
        vx_world = cos_yaw * self._filtered_vx - sin_yaw * self._filtered_vy
        vy_world = sin_yaw * self._filtered_vx + cos_yaw * self._filtered_vy
        
        # Integrate position
        self._x += vx_world * dt
        self._y += vy_world * dt
        self._yaw = yaw_corrected
        self._history.append((self._x, self._y))
        if len(self._history) > self.max_points:
            self._history = self._history[-self.max_points :]

    @property
    def history(self) -> list[tuple[float, float]]:
        return self._history

    @property
    def state(self) -> tuple[float, float, float]:
        return self._x, self._y, self._yaw

    def payload(self) -> dict:
        positions: list[float] = []
        for x, y in self._history:
            positions.extend([x, y, 0.0])
        hx = self._x + math.cos(self._yaw) * self.arrow_length
        hy = self._y + math.sin(self._yaw) * self.arrow_length
        heading = [self._x, self._y, 0.0, hx, hy, 0.0]
        return {
            "positions": positions,
            "heading": heading,
            "yaw": math.degrees(self._yaw),
            "yaw_rad": self._yaw,
            "timestamp": time.time(),
        }


class TrajectoryPlotter:
    def __init__(self, arrow_length: float, refresh_interval: float) -> None:
        if plt is None:
            raise RuntimeError("matplotlib is required for --live-plot")
        self.arrow_length = arrow_length
        self.refresh_interval = max(refresh_interval, 0.0)
        self._last_update = 0.0
        plt.ion()
        self.fig, self.ax = plt.subplots()
        (self.traj_line,) = self.ax.plot([], [], color="C0", linewidth=1.5)
        (self.current_point,) = self.ax.plot([], [], marker="o", color="C3")
        from matplotlib.patches import FancyArrowPatch

        self.arrow_patch = FancyArrowPatch((0.0, 0.0), (0.0, 0.0), color="C1", arrowstyle="->", mutation_scale=12, linewidth=1.5)
        self.ax.add_patch(self.arrow_patch)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlim(-1.0, 1.0)
        self.ax.set_ylim(-1.0, 1.0)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.grid(True, alpha=0.3)

    def update(self, history: list[tuple[float, float]], state: tuple[float, float, float], *, force: bool = False) -> None:
        if not history:
            return
        now = time.perf_counter()
        if not force and self.refresh_interval > 0.0 and (now - self._last_update) < self.refresh_interval:
            return
        self._last_update = now
        xs, ys = zip(*history)
        self.traj_line.set_data(xs, ys)
        x, y, yaw = state
        self.current_point.set_data([x], [y])
        dx = math.cos(yaw) * self.arrow_length
        dy = math.sin(yaw) * self.arrow_length
        self.arrow_patch.set_positions((x, y), (x + dx, y + dy))
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def finalize(self, history: list[tuple[float, float]], state: tuple[float, float, float]) -> None:
        if plt is None:
            return
        self.update(history, state, force=True)
        plt.ioff()
        try:
            plt.show()
        except RuntimeError:
            pass


def _open_serial(port: str, baudrate: int):
    if serial is None:
        raise RuntimeError(f"pyserial is not available: {SERIAL_IMPORT_ERROR}")
    return serial.Serial(port, baudrate, timeout=1)


def _parse_line(line: bytes) -> Optional[tuple[float, float, float, Optional[float]]]:
    try:
        data_str = line.decode("utf-8").strip()
    except UnicodeDecodeError:
        return None
    if not data_str:
        return None
    parts = data_str.split(",")
    if len(parts) < 3:
        return None
    try:
        vx = float(parts[0].strip())
        vy = float(parts[1].strip())
        yaw = float(parts[2].strip())
        # Optional: parse timestamp if IMU provides it (in seconds)
        timestamp = float(parts[3].strip()) if len(parts) > 3 else None
    except ValueError:
        return None
    return vx, vy, yaw, timestamp


def run_serial(args: argparse.Namespace) -> None:
    integrator = PlanarIntegrator(
        max_points=args.max_points,
        arrow_length=args.arrow_length,
        vx_bias=args.vx_bias,
        vy_bias=args.vy_bias,
        zupt_threshold=args.zupt_threshold,
        zupt_samples=args.zupt_samples,
        filter_alpha=args.filter_alpha,
        turn_scale=args.turn_scale,
        turn_threshold=args.turn_threshold,
        yaw_drift_correction=args.yaw_drift_correction,
    )
    plotter: Optional[TrajectoryPlotter]
    if args.live_plot:
        plotter = TrajectoryPlotter(args.arrow_length, args.plot_interval)
    else:
        plotter = None
    ser = _open_serial(args.port, args.baudrate)
    print(f"Recording IMU trajectory from {args.port} @ {args.baudrate} baud")
    last_time = time.perf_counter()
    last_imu_timestamp: Optional[float] = None
    
    # dt statistics tracking
    dt_samples: list[float] = []
    sample_count = 0
    stats_print_interval = 100  # Print stats every N samples
    
    # Turn diagnostics
    prev_yaw = 0.0
    yaw_rate_threshold = 5.0  # degrees per second to consider "turning"
    
    try:
        while True:
            line = ser.readline()
            if not line:
                continue
            parsed = _parse_line(line)
            if parsed is None:
                continue
            vx, vy, yaw_deg, imu_timestamp = parsed
            yaw_rad = math.radians(yaw_deg)
            
            # Use IMU timestamp if available, otherwise use system time
            if imu_timestamp is not None:
                if last_imu_timestamp is not None:
                    dt = imu_timestamp - last_imu_timestamp
                else:
                    dt = 0.0
                last_imu_timestamp = imu_timestamp
            else:
                now = time.perf_counter()
                dt = now - last_time
                last_time = now
            
            # Collect dt statistics (skip first sample and zero dt)
            if dt > 0.0:
                dt_samples.append(dt)
                sample_count += 1
                
                # Calculate yaw rate
                yaw_rate = abs(yaw_deg - prev_yaw) / dt  # degrees per second
                prev_yaw = yaw_deg
                
                # Diagnostic: log when turning with significant velocity
                if args.debug_turns:
                    speed = math.sqrt(vx**2 + vy**2)
                    if yaw_rate > yaw_rate_threshold and speed > 0.05:
                        print(f"[TURN] yaw_rate={yaw_rate:.1f}°/s, vx={vx:.3f}, vy={vy:.3f}, speed={speed:.3f}, yaw={yaw_deg:.1f}°")
                
                # Print statistics periodically
                if sample_count % stats_print_interval == 0:
                    avg_dt = sum(dt_samples) / len(dt_samples)
                    min_dt = min(dt_samples)
                    max_dt = max(dt_samples)
                    # Calculate standard deviation
                    variance = sum((x - avg_dt) ** 2 for x in dt_samples) / len(dt_samples)
                    std_dt = variance ** 0.5
                    avg_hz = 1.0 / avg_dt if avg_dt > 0 else 0.0
                    
                    print(f"[{sample_count} samples] dt: avg={avg_dt*1000:.2f}ms ({avg_hz:.1f}Hz), "
                          f"min={min_dt*1000:.2f}ms, max={max_dt*1000:.2f}ms, std={std_dt*1000:.2f}ms")
                    
                    # Keep only recent samples for rolling statistics
                    if len(dt_samples) > 1000:
                        dt_samples = dt_samples[-1000:]
            
            integrator.step(vx, vy, yaw_rad, dt)
            _atomic_write(args.output, integrator.payload())
            if plotter is not None:
                plotter.update(integrator.history, integrator.state)
    except KeyboardInterrupt:
        print("\n\nStopping...")
        # Print final statistics
        if dt_samples:
            avg_dt = sum(dt_samples) / len(dt_samples)
            min_dt = min(dt_samples)
            max_dt = max(dt_samples)
            variance = sum((x - avg_dt) ** 2 for x in dt_samples) / len(dt_samples)
            std_dt = variance ** 0.5
            avg_hz = 1.0 / avg_dt if avg_dt > 0 else 0.0
            
            print(f"\n=== Final dt Statistics ({len(dt_samples)} samples) ===")
            print(f"Average dt: {avg_dt*1000:.3f} ms ({avg_hz:.2f} Hz)")
            print(f"Min dt:     {min_dt*1000:.3f} ms")
            print(f"Max dt:     {max_dt*1000:.3f} ms")
            print(f"Std dev:    {std_dt*1000:.3f} ms")
            print(f"Jitter:     ±{std_dt*1000:.3f} ms ({std_dt/avg_dt*100:.1f}%)")
    finally:
        ser.close()
        if plotter is not None:
            plotter.finalize(integrator.history, integrator.state)


def run_simulated(args: argparse.Namespace) -> None:
    integrator = PlanarIntegrator(max_points=args.max_points, arrow_length=args.arrow_length)
    plotter: Optional[TrajectoryPlotter]
    if args.live_plot:
        plotter = TrajectoryPlotter(args.arrow_length, args.plot_interval)
    else:
        plotter = None
    print("Running IMU trajectory simulator")
    t = 0.0
    dt = 1.0 / args.sim_hz
    try:
        while True:
            yaw = math.fmod(t * args.turn_rate, 2.0 * math.pi)
            vx = args.velocity
            vy = 0.0
            integrator.step(vx, vy, yaw, dt)
            _atomic_write(args.output, integrator.payload())
            if plotter is not None:
                plotter.update(integrator.history, integrator.state)
            time.sleep(dt)
            t += dt
    except KeyboardInterrupt:
        print("\nStopping simulation...")
    finally:
        if plotter is not None:
            plotter.finalize(integrator.history, integrator.state)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Integrate planar IMU velocities into a web-viewable trajectory")
    parser.add_argument("--output", type=Path, default=Path("logs/imu_trajectory.json"), help="Output JSON path")
    parser.add_argument("--max-points", type=int, default=2048, help="Number of samples to keep in history")
    parser.add_argument("--arrow-length", type=float, default=0.3, help="Heading arrow length in meters")
    parser.add_argument("--live-plot", action="store_true", help="Show a real-time trajectory plot with heading arrow")
    parser.add_argument("--plot-interval", type=float, default=0.1, help="Seconds between live plot refreshes")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--simulate", action="store_true", help="Run without hardware and generate a circular trajectory")
    mode.add_argument("--port", default="/dev/ttyACM0", help="Serial port for live IMU data")

    parser.add_argument("--baudrate", type=int, default=921600, help="Serial baud rate")
    
    # Drift reduction parameters
    parser.add_argument("--vx-bias", type=float, default=0.0, help="Velocity X bias correction (m/s)")
    parser.add_argument("--vy-bias", type=float, default=0.0, help="Velocity Y bias correction (m/s)")
    parser.add_argument("--zupt-threshold", type=float, default=0.02, help="Zero-velocity detection threshold (m/s)")
    parser.add_argument("--zupt-samples", type=int, default=5, help="Number of samples before applying ZUPT")
    parser.add_argument("--filter-alpha", type=float, default=0.8, help="Low-pass filter coefficient (0-1, higher = more filtering)")
    parser.add_argument("--turn-scale", type=float, default=0.7, help="Scale velocity during turns (0-1, lower = less trust in turn data)")
    parser.add_argument("--turn-threshold", type=float, default=10.0, help="Yaw rate threshold to detect turns (degrees/s)")
    parser.add_argument("--yaw-drift-correction", type=float, default=0.0, help="Systematic yaw drift correction (degrees/s, + for counter-clockwise drift)")
    parser.add_argument("--debug-turns", action="store_true", help="Print diagnostic info when turning")

    parser.add_argument("--sim-hz", type=float, default=30.0, help="Simulation update rate (Hz)")
    parser.add_argument("--velocity", type=float, default=0.2, help="Forward velocity for simulator (m/s)")
    parser.add_argument("--turn-rate", type=float, default=0.25, help="Yaw rate for simulator (rad/s)")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    args.output = args.output.expanduser().resolve()

    if args.live_plot and plt is None:
        parser.error("matplotlib is required for --live-plot. Install it or drop the flag.")

    if args.simulate:
        run_simulated(args)
    else:
        if not args.port:
            parser.error("--port is required when not running in simulation")
        run_serial(args)


if __name__ == "__main__":
    main(sys.argv[1:])
