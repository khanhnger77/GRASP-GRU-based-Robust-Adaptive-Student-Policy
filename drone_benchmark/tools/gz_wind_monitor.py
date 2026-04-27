#!/usr/bin/env python3
"""
Realtime Gazebo wind monitor.

Reads a Gazebo wind topic and prints:
  - wind vector (vx, vy, vz)
  - total speed (m/s)
  - horizontal speed (m/s)
  - horizontal direction angle (deg, from +X axis)
"""

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Tuple


def _run(cmd: Iterable[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(cmd),
        check=False,
        text=True,
        capture_output=True,
    )


def _extract_xyz(msg: Dict) -> Optional[Tuple[float, float, float]]:
    # Different plugins / Gazebo versions can expose slightly different field names.
    candidates = (
        ("linear_velocity",),
        ("velocity",),
        ("wind",),
        ("wind_velocity",),
        ("data",),
    )
    for key_path in candidates:
        node = msg
        ok = True
        for key in key_path:
            if not isinstance(node, dict) or key not in node:
                ok = False
                break
            node = node[key]
        if not ok or not isinstance(node, dict):
            continue
        if all(k in node for k in ("x", "y", "z")):
            try:
                return float(node["x"]), float(node["y"]), float(node["z"])
            except (TypeError, ValueError):
                return None
    return None


def _list_topics() -> Optional[List[str]]:
    proc = _run(["gz", "topic", "-l"])
    if proc.returncode != 0:
        return None
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _find_wind_topic(world: str, topics: List[str]) -> Optional[str]:
    preferred = [
        f"/world/{world}/wind_info",
        f"/world/{world}/wind",
    ]

    for p in preferred:
        if p in topics:
            return p

    for t in topics:
        if f"/world/{world}/" in t and "wind" in t.lower():
            return t
    for t in topics:
        if "wind" in t.lower():
            return t
    return None


def _extract_worlds(topics: List[str]) -> List[str]:
    worlds = set()
    for t in topics:
        m = re.match(r"^/world/([^/]+)/", t)
        if m:
            worlds.add(m.group(1))
    return sorted(worlds)


def _cardinal_from_deg(deg: float) -> str:
    # 8-way compass relative to +X axis.
    dirs = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    idx = int((deg + 22.5) // 45) % 8
    return dirs[idx]


def main() -> int:
    parser = argparse.ArgumentParser(description="Print realtime wind from Gazebo topics.")
    parser.add_argument(
        "--world",
        default="lawn",
        help="Gazebo world name (default: lawn).",
    )
    parser.add_argument(
        "--topic",
        default=None,
        help="Wind topic (auto-detected if omitted).",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=5.0,
        help="Print rate in Hz (default: 5).",
    )
    args = parser.parse_args()

    if shutil.which("gz") is None:
        print("ERROR: 'gz' command not found in PATH.", file=sys.stderr)
        return 1

    topics = _list_topics()
    if topics is None:
        print("ERROR: Failed to query Gazebo topics (`gz topic -l`).", file=sys.stderr)
        return 2

    topic = args.topic or _find_wind_topic(args.world, topics)
    if not topic:
        worlds = _extract_worlds(topics)
        wind_topics = [t for t in topics if "wind" in t.lower()]
        if not topics:
            print(
                "ERROR: No Gazebo topics found. Gazebo is likely not running.",
                file=sys.stderr,
            )
            return 2

        print(
            f"ERROR: Cannot find wind topic for world '{args.world}'. "
            "Start Gazebo first, then retry.",
            file=sys.stderr,
        )
        if worlds:
            print(f"Detected worlds: {', '.join(worlds)}", file=sys.stderr)
        if wind_topics:
            print("Detected wind-like topics:", file=sys.stderr)
            for t in wind_topics[:10]:
                print(f"  {t}", file=sys.stderr)
        else:
            print("No topic containing 'wind' found.", file=sys.stderr)
        return 2

    echo_cmd = ["gz", "topic", "-e", "--json-output", "-t", topic]
    proc = subprocess.Popen(
        echo_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    print(f"Listening topic: {topic}")
    print("Columns: speed[m/s] | horiz[m/s] | dir[deg] | card | vx vy vz")

    min_period = 1.0 / max(args.hz, 0.1)
    last_t = -1.0

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue

            now = math.inf
            if last_t >= 0:
                now = last_t + 0.000001
            # We throttle by simple wall clock via monotonic from perf_counter.
            # Kept local to avoid extra imports at module level.
            import time  # pylint: disable=import-outside-toplevel
            now = time.perf_counter()
            if last_t >= 0 and (now - last_t) < min_period:
                continue
            last_t = now

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            xyz = _extract_xyz(msg)
            if xyz is None:
                continue
            vx, vy, vz = xyz

            speed = math.sqrt(vx * vx + vy * vy + vz * vz)
            horiz = math.hypot(vx, vy)
            deg = (math.degrees(math.atan2(vy, vx)) + 360.0) % 360.0
            card = _cardinal_from_deg(deg)

            print(
                f"{speed:8.3f} | {horiz:8.3f} | {deg:8.2f} | {card:>2} | "
                f"{vx:7.3f} {vy:7.3f} {vz:7.3f}"
            )
            sys.stdout.flush()

    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
