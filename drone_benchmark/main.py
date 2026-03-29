#!/usr/bin/env python3
"""
Drone Benchmark — main entry point.

Usage:
    python main.py --test hover     --controller pid
    python main.py --test figure8   --controller rl
    python main.py --test step      --controller pid
    python main.py --test disturbance --controller rl

Optional flags:
    --config  path to config.yaml   (default: config.yaml)
    --output  path prefix for logs  (overrides config output_dir)
"""

import argparse
import json
import logging
import os
import sys
import time

import yaml

# Allow imports from project root
sys.path.insert(0, os.path.dirname(__file__))

from mavlink_interface.connection import MAVLinkConnection
from controllers.pid_controller import PIDController
from controllers.rl_controller import RLController
from tests.hover_test import HoverTest
from tests.figure8_test import Figure8Test
from tests.step_response_test import StepResponseTest
from tests.disturbance_test import DisturbanceTest

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark PID vs RL flight controllers on PX4/GZ-X500."
    )
    parser.add_argument(
        "--test",
        choices=["hover", "figure8", "step", "disturbance"],
        required=True,
        help="Which flight test to run.",
    )
    parser.add_argument(
        "--controller",
        choices=["pid", "rl"],
        required=True,
        help="Which controller to use.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override log/plot output directory.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_controller(name: str, config: dict):
    if name == "pid":
        pid_cfg = config["controllers"]["pid"]
        # Outer position loop: MPC_XY_P, MPC_Z_P
        # Velocity loop: MPC_XY_VEL_P/I/D_ACC, MPC_Z_VEL_P/I/D_ACC
        # (PIDController is kept for reference; actual flight uses PX4's built-in stack)
        ctrl = PIDController(
            kp=pid_cfg["position_p"],
            ki=pid_cfg["velocity_i"],
            kd=pid_cfg["velocity_d"],
            max_velocity=pid_cfg["max_vel_xy"],
        )
        logger.info(
            "PID controller initialised (PX4 X500 params: pos_P=%s vel_I=%s vel_D=%s).",
            pid_cfg["position_p"], pid_cfg["velocity_i"], pid_cfg["velocity_d"],
        )
        return ctrl

    elif name == "rl":
        # RLController does NOT do local inference — it activates the on-board
        # rl_tools_policy module via MAVLink shell commands.
        # The connection is set later (after MAVLinkConnection is established).
        return None  # placeholder; resolved in main() after connection is open

    raise ValueError(f"Unknown controller: {name}")


def csv_path(config: dict, test_name: str, controller_name: str, output_override=None) -> str:
    base = output_override or config["logging"]["output_dir"]
    os.makedirs(base, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(base, f"{test_name}_{controller_name}_{ts}.csv")


def build_test(test_name: str, connection, controller, config, csv_file: str):
    common = dict(
        connection=connection,
        controller=controller,
        config=config,
        csv_path=csv_file,
    )

    if test_name == "hover":
        return HoverTest(**common)
    elif test_name == "figure8":
        return Figure8Test(**common)
    elif test_name == "step":
        return StepResponseTest(**common)
    elif test_name == "disturbance":
        return DisturbanceTest(**common)

    raise ValueError(f"Unknown test: {test_name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.output:
        config["logging"]["output_dir"] = args.output
        config["plotting"]["output_dir"] = args.output

    logger.info("=== Drone Benchmark ===")
    logger.info("  Test       : %s", args.test)
    logger.info("  Controller : %s", args.controller)
    logger.info("  Config     : %s", args.config)

    controller = build_controller(args.controller, config)

    conn_str = config["mavlink"]["connection_string"]
    hb_timeout = config["mavlink"]["heartbeat_timeout"]

    with MAVLinkConnection(conn_str, heartbeat_timeout=hb_timeout) as conn:
        # RLController needs the live connection to send shell commands
        if args.controller == "rl":
            controller = RLController(conn)
            logger.info("RL controller initialised (on-board module via MAVLink shell).")

        log_file = csv_path(config, args.test, args.controller, args.output)
        test = build_test(args.test, conn, controller, config, log_file)

        logger.info("Starting test. CSV → %s", log_file)
        results = test.run()

    # Save results summary as JSON alongside the CSV
    results_path = log_file.replace(".csv", "_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Results saved to %s", results_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
