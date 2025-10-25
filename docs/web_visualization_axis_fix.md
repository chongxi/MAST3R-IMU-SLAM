# Web Visualization Axis Fix

## Background

The desktop viewer (`main.py` → `mast3r_slam/visualization.py`) renders surfels with ModernGL, using pose matrices (`T_WC`) and surfel positions (`X_canon`) that follow the standard computer-vision camera frame:

- right-handed
- `x` to the right, `y` down, `z` forward

`resources/programs/surfelmap.glsl` expects this basis, so the axes appear correct in the native window.

The web viewer (`main_web.py` → `mast3r_slam/web_visualization.py` → `resources/web/app.js`) streams the same data to WebGL. WebGL’s view/projection math uses the classic OpenGL clip-space convention where `y` points up and `z` is toward the viewer (negative forward). Our orbit camera (`mat4LookAt`) assumes that basis, so feeding it OpenCV data mirrored the scene vertically.

## Investigation

1. Added `scripts/dump_frame_state.py` to dump `T_WC` / `X_canon` directly from MASt3R.
2. Added `scripts/fetch_web_state.py` to read `/api/state` and confirm the server sends the raw (unflipped) values.
3. With both dumps matching, the remaining mismatch had to be in the WebGL layer.

## Fix

Implemented a proper CV → WebGL basis swap in `resources/web/app.js`:

- Added helpers (`cvToGlVec3`, `cvToGlBuffer`, `cvToGlPose`) that multiply the `y` and `z` components by −1 while preserving the homogeneous row/column in pose matrices.
- Applied these helpers when uploading:
  - surfel positions and normals
  - camera pose used for axis gizmos
  - keyframe markers, axis triads, trajectory lines, loop-closure edges
  - current-frame position used by the orbit camera target
- Removed the previous `flipZ` calls to avoid double negation.

## Verification

- Reloading the web viewer now shows the same orientation as the desktop visualization.
- `scripts/fetch_web_state.py` still reports the original CV-system values, confirming the conversion is confined to the browser.

## Notes / Next Steps

- Keep server-side data in the CV convention; the browser handles conversion.
- Reuse the helper scripts for future regressions.
- Any new WebGL elements should pass through the same helpers to stay consistent.
