#!/usr/bin/env python3
"""Fetch the current web-visualization state and print key fields.

Useful when `main_web.py` is running and serving the `/api/state` endpoint.
"""

from __future__ import annotations

import argparse
import json
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch the current /api/state payload from the web viewer.")
    parser.add_argument("--host", default="127.0.0.1", help="Web visualization host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=7860, help="Web visualization port (default: 7860).")
    parser.add_argument("--max-print-points", type=int, default=5, help="Number of surfel samples to print.")
    parser.add_argument("--raw", action="store_true", help="Print the full JSON payload instead of a summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    url = f"http://{args.host}:{args.port}/api/state"
    with urllib.request.urlopen(url) as response:  # noqa: S310 (local request)
        payload = json.loads(response.read().decode("utf-8"))

    if args.raw:
        print(json.dumps(payload, indent=2))
        return

    print("=== /api/state summary ===")
    scene = payload.get("scene", {})
    print(f"point count   : {scene.get('count', 0)}")
    print(f"camera pose   : {scene.get('cameraPose', 'n/a')}")

    current = payload.get("currentFrame", {})
    print(f"frame id      : {current.get('id')}")
    print(f"frame pose    : {current.get('pose', 'n/a')}")
    print(f"frame position: {current.get('position')}")

    points = scene.get("points", [])
    normals = scene.get("normals", [])
    colors = scene.get("colors", [])

    n_samples = min(args.max_print_points, len(points) // 3)
    print(f"\nFirst {n_samples} points:")
    for i in range(n_samples):
        base = i * 3
        px, py, pz = points[base : base + 3]
        nx, ny, nz = normals[base : base + 3]
        cr, cg, cb = colors[base : base + 3]
        print(f"  #{i:02d} p=({px:.4f}, {py:.4f}, {pz:.4f}) n=({nx:.4f}, {ny:.4f}, {nz:.4f}) rgb=({cr:.3f}, {cg:.3f}, {cb:.3f})")


if __name__ == "__main__":
    main()
