# IMU-Backed Fallback Design

## Background
### What a `Frame` Represents
`Frame` objects (`mast3r_slam/frame.py:17-105`) package the RGB observation and all derived state used downstream:
- image tensors (`img`, `img_shape`, `img_true_shape`, `uimg`)
- MASt3R encoder caches (`feat`, `pos`)
- canonical point map and confidences (`X_canon`, `C`, plus `N`, `N_updates` for the fusion policy)
- current pose `T_WC` and optional intrinsics `K`

### Current Tracking Modes & Triggers
The main loop operates the global mode machine (`main.py:242-320`):
- `Mode.INIT` seeds the first keyframe.
- `Mode.TRACKING` runs `FrameTracker.track`, which relies on visual correspondences. When the match fraction drops below `tracking.min_match_frac`, tracking returns `try_reloc=True` and the frontend switches to `Mode.RELOC`.
- `Mode.RELOC` queues the frame for the backend relocalization procedure (`main.py:28-71`).

Tracking success hinges on the Gauss–Newton solvers (`opt_pose_ray_dist_sim3` or `opt_pose_calib_sim3`). When they fail—typically because too few inlier matches survive—we currently have no intermediate fallback before relocalization.

## Goal
Introduce an IMU-driven fallback so the system can propagate poses during visual dropouts, continue accumulating frames for reconstruction, and resume vision tracking when conditions improve.

## Work Plan

1. **Surface IMU Data**
   - Extend dataset interfaces (`mast3r_slam/dataloader.py`) to emit synchronized accelerometer/gyroscope packets.
   - Add IMU containers to `Frame` and to the shared memory structures (`SharedStates`, `SharedKeyframes`) so both the frontend and backend can access measurements.

2. **Expand Mode Handling**
   - Update `Mode` in `mast3r_slam/frame.py` to distinguish `TRACKING_VISION` and `TRACKING_IMU`.
   - Modify the main loop in `main.py` so low match fractions trigger `TRACKING_IMU` instead of immediately entering `Mode.RELOC`, and successful visual matches trigger the reverse transition.

3. **IMU Preintegration Module**
   - Implement a new helper (e.g., `mast3r_slam/imu.py`) that incrementally integrates raw IMU samples into relative Sim(3) deltas with bias/gravity handling.
   - Provide APIs for resetting, accumulating, and querying the preintegrated motion between frame timestamps.

4. **Tracker Adjustments**
   - In `FrameTracker.track` (`mast3r_slam/tracker.py`), consume the IMU preintegration when `match_frac` falls below the threshold: update `frame.T_WC` using the IMU delta and return `try_reloc=False`.
   - When visual matches recover, optionally fuse the IMU prior into the Gauss–Newton solve to smooth the hand-off back to vision.

5. **IMU Backend Worker**
   - Add a dedicated worker (either within `main.py` or as a separate module) to buffer IMU packets, maintain per-interval preintegration, and publish predicted poses while `Mode.TRACKING_IMU` is active.
   - Use shared queues or shared-memory slots to exchange predictions with the main process.

6. **Failure Escalation Rules**
   - Detect IMU anomalies (missing data, huge residuals) inside the IMU worker and escalate to relocalization only when both vision and inertial cues are unreliable.
   - Ensure the existing relocalization workflow still has access to the latest frame/keyframe data after IMU segments.

7. **Scene Reconstruction During IMU Mode**
   - Continue creating frames, updating pointmaps, and storing them in `SharedKeyframes`. Mark IMU-only poses with lower confidence so the backend can down-weight or defer their constraints until vision confirms them.
   - When visual tracking resumes, reconcile the IMU-propagated frames by running the optimiser to re-align them.

## Validation & Next Steps
- Instrument logging/metrics to track mode switches and IMU usage.
- Add regression tests in `performance_test/` that cover clean vision, pure IMU fallback, and vision recovery scenarios.
- Iterate on preintegration fidelity (bias estimation, covariance tracking) once the basic fallback path is functional.
