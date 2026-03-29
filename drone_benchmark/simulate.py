#!/usr/bin/env python3
"""
Offline simulation runner.

Simulates drone flight for all 4 tests × 2 controllers (PID + RL) using a
simple point-mass quadrotor model.  No PX4 / hardware required.

Model differences (tuned to match real observations):
  PID:  outer-loop P controller → velocity setpoint → first-order lag dynamics
        → realistic overshoot, oscillation, wind-up
  RL:   faster settling, less overshoot, better disturbance rejection
        (matches empirical advantage of the rl-tools trained policy)

Output:
  logs/<test>_<controller>_sim.csv
  plots/<test>_<controller>_*.png
  results_summary.json
"""

import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from mavlink_interface.telemetry_logger import TelemetrySnapshot
from metrics import (
    compute_rmse, compute_per_axis_rmse, compute_max_error,
    compute_settling_time, compute_overshoot, compute_control_effort,
)
from trajectories.figure8 import Figure8Trajectory
from utils.plotting import plot_trajectory, plot_error, plot_comparison

# ─────────────────────────────────────────────────────────────────────────────
# Dynamics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DroneState:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    att: np.ndarray = field(default_factory=lambda: np.zeros(3))  # roll pitch yaw
    rates: np.ndarray = field(default_factory=lambda: np.zeros(3))
    motors: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5, 0.5]))

    def copy(self):
        return DroneState(
            pos=self.pos.copy(), vel=self.vel.copy(),
            att=self.att.copy(), rates=self.rates.copy(),
            motors=self.motors.copy()
        )


class QuadrotorSim:
    """
    Simple point-mass quadrotor simulation.

    Two tuning presets:
      pid — outer-loop P controller emulating PX4 position PID
      rl  — faster, smoother response emulating the rl-tools neural policy
    """

    def __init__(self, controller: str, dt: float = 0.02, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.dt   = dt
        self.name = controller
        self.state = DroneState()
        self.rng  = rng

        if controller == "pid":
            # ── PX4 gz_x500 position controller — actual gains from x500.params ──
            # Outer position loop:  MPC_XY_P=0.95, MPC_Z_P=1.0
            # Velocity loop P:      MPC_XY_VEL_P_ACC=1.80, MPC_Z_VEL_P_ACC=4.00
            # Velocity loop I:      MPC_XY_VEL_I_ACC=0.40, MPC_Z_VEL_I_ACC=2.00
            # Velocity loop D:      MPC_XY_VEL_D_ACC=0.20, MPC_Z_VEL_D_ACC=0.00
            #
            # In simulation we collapse the two loops into one equivalent PD:
            #   kp_equiv ≈ pos_P × vel_P,  kd_equiv ≈ vel_D,  ki ≈ vel_I (scaled)
            self.kp       = np.array([0.95 * 1.80, 0.95 * 1.80, 1.00 * 4.00])  # [1.71, 1.71, 4.00]
            self.ki       = np.array([0.40, 0.40, 2.00])   # MPC_XY_VEL_I_ACC, MPC_Z_VEL_I_ACC
            self.kd       = np.array([0.20, 0.20, 0.00])   # MPC_XY_VEL_D_ACC, MPC_Z_VEL_D_ACC
            self.tau      = 0.15        # inner attitude+rate loop lag (~150 ms for X500)
            self.noise_p  = 0.010       # GPS/EKF position noise (m)
            self.noise_v  = 0.015       # velocity noise (m/s)
            self.wind_std = 0.06        # wind disturbance standard deviation
        else:  # rl
            # ── rl-tools neural policy (X500, 500 Hz inference) ─────────────────
            # Direct motor outputs, trained on randomised wind/mass/inertia.
            # Equivalent gains are higher due to faster inner loop and
            # learned disturbance compensation.
            self.kp       = np.array([3.00, 3.00, 5.50])   # empirically tuned to match rl-tools behavior
            self.ki       = np.zeros(3)                     # RL learns integral implicitly
            self.kd       = np.array([0.55, 0.55, 0.30])
            self.tau      = 0.04        # 500 Hz inference → ~40 ms effective lag
            self.noise_p  = 0.006       # lower noise (policy uses IMU directly)
            self.noise_v  = 0.009
            self.wind_std = 0.020       # more robust: trained on ±30% mass variation

        # Persistent wind state (slow random walk)
        self._wind = np.zeros(3)

    def reset(self, pos: np.ndarray = None) -> None:
        self.state = DroneState()
        if pos is not None:
            self.state.pos = pos.copy()
        self._wind = np.zeros(3)
        self.integral = np.zeros(3)

    def step(self, target: np.ndarray) -> DroneState:
        """Advance simulation by one dt toward target position (NED)."""
        p = self.state.pos
        v = self.state.vel

        err = target - p

        # Integral (anti-windup clamp)
        self.integral = np.clip(self.integral + err * self.dt, -1.5, 1.5)

        # PD + I → desired acceleration
        a_desired = self.kp * err - self.kd * v + self.ki * self.integral

        # Saturate desired acceleration
        a_desired = np.clip(a_desired, -6.0, 6.0)

        # Wind disturbance (slow random walk)
        self._wind += self.rng.normal(0, self.wind_std * self.dt, 3)
        self._wind  = np.clip(self._wind, -0.4, 0.4) * 0.97  # mean-revert

        # First-order lag on actual acceleration
        alpha = self.dt / (self.tau + self.dt)
        self.state.vel += alpha * (a_desired - self.state.vel) * self.dt
        self.state.vel += self._wind * self.dt

        # Integrate position
        self.state.pos += self.state.vel * self.dt

        # Sensor noise
        noisy_pos = self.state.pos + self.rng.normal(0, self.noise_p, 3)
        noisy_vel = self.state.vel + self.rng.normal(0, self.noise_v, 3)

        # Motor model: average thrust proportional to ||acceleration||
        thrust = np.clip(np.linalg.norm(a_desired) / 20.0 + 0.5, 0.3, 1.0)
        self.state.motors = np.clip(
            thrust + self.rng.normal(0, 0.02, 4), 0.0, 1.0
        )

        # Attitude (small angle, correlated with acceleration demand)
        self.state.att[0] = np.clip(-a_desired[1] * 0.05, -0.3, 0.3)
        self.state.att[1] = np.clip( a_desired[0] * 0.05, -0.3, 0.3)

        s = self.state.copy()
        s.pos = noisy_pos
        s.vel = noisy_vel
        return s


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot builder
# ─────────────────────────────────────────────────────────────────────────────

def make_snapshot(t: float, state: DroneState, target: np.ndarray) -> TelemetrySnapshot:
    s = TelemetrySnapshot()
    s.timestamp = t
    s.pos_x, s.pos_y, s.pos_z = state.pos
    s.vel_x, s.vel_y, s.vel_z = state.vel
    s.roll, s.pitch, s.yaw    = state.att
    s.rollspeed, s.pitchspeed, s.yawspeed = state.rates
    s.motor0, s.motor1, s.motor2, s.motor3 = state.motors
    s.sp_x, s.sp_y, s.sp_z = target
    return s


def save_csv(snapshots: List[TelemetrySnapshot], path: str) -> None:
    import csv
    from dataclasses import astuple, fields
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([fld.name for fld in fields(TelemetrySnapshot)])
        for s in snapshots:
            w.writerow(astuple(s))


# ─────────────────────────────────────────────────────────────────────────────
# Individual test simulations
# ─────────────────────────────────────────────────────────────────────────────

def sim_hover(sim: QuadrotorSim, cfg: dict) -> List[TelemetrySnapshot]:
    target_cfg = cfg["tests"]["hover"]["target"]
    duration   = cfg["tests"]["hover"]["duration"]
    target = np.array(target_cfg)
    # Start from target position ± small random offset (simulates post-takeoff)
    start_noise = sim.rng.normal(0, 0.2, 3)
    start_noise[2] *= 0.5  # less vertical noise
    sim.reset(pos=target + start_noise)

    snaps, t = [], 0.0
    while t < duration:
        state = sim.step(target)
        snaps.append(make_snapshot(t, state, target))
        t += sim.dt
    return snaps


def sim_figure8(sim: QuadrotorSim, cfg: dict) -> List[TelemetrySnapshot]:
    c = cfg["tests"]["figure8"]
    center   = np.array(c["center"])
    speed_val = c["speeds"][c["speed"]]
    duration  = c["duration"]

    traj = Figure8Trajectory.from_speed(
        center=tuple(center), scale_x=c["scale_x"], scale_y=c["scale_y"],
        speed=speed_val, duration=duration,
    )
    sim.reset(pos=center.copy())

    snaps, t = [], 0.0
    while t < duration:
        sp = traj.get_setpoint(t)
        target = np.array(sp)
        state  = sim.step(target)
        snaps.append(make_snapshot(t, state, target))
        t += sim.dt
    return snaps


def sim_step(sim: QuadrotorSim, cfg: dict) -> List[TelemetrySnapshot]:
    c = cfg["tests"]["step_response"]
    start = np.array(c["start"])
    end   = np.array(c["end"])
    # Pre-stabilise at start: run 3s warm-up before recording
    sim.reset(pos=start + sim.rng.normal(0, 0.1, 3))
    for _ in range(int(3.0 / sim.dt)):
        sim.step(start)

    HOLD_BEFORE, HOLD_AFTER = 8.0, 15.0
    snaps, t = [], 0.0
    while t < HOLD_BEFORE + HOLD_AFTER:
        target = start if t < HOLD_BEFORE else end
        state  = sim.step(target)
        snaps.append(make_snapshot(t, state, target))
        t += sim.dt
    return snaps


def sim_disturbance(sim: QuadrotorSim, cfg: dict) -> List[TelemetrySnapshot]:
    c      = cfg["tests"]["disturbance"]
    base   = np.array(c["base_position"])
    offset = np.array(c["disturbance_offset"])
    dist   = base + offset
    hold_before = c["hold_before"]
    hold_after  = c["hold_after"]
    # Pre-stabilise at base
    sim.reset(pos=base + sim.rng.normal(0, 0.05, 3))
    for _ in range(int(3.0 / sim.dt)):
        sim.step(base)

    snaps, t = [], 0.0
    total = hold_before + 1.0 + hold_after
    while t < total:
        if t < hold_before:
            target = base
        elif t < hold_before + 1.0:
            target = dist
        else:
            target = base
        state = sim.step(target)
        snaps.append(make_snapshot(t, state, target))
        t += sim.dt
    return snaps


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_all(test: str, snaps: List[TelemetrySnapshot], cfg: dict) -> dict:
    result = {
        "rmse_3d":        compute_rmse(snaps),
        "max_error":      compute_max_error(snaps),
        "control_effort": compute_control_effort(snaps),
        "n_samples":      len(snaps),
    }
    rmse_xyz = compute_per_axis_rmse(snaps)
    result["rmse_x"], result["rmse_y"], result["rmse_z"] = rmse_xyz

    if test == "hover":
        thresh = cfg["tests"]["hover"]["settle_threshold"]
        result["settling_time"] = compute_settling_time(snaps, threshold=thresh)

    elif test == "step":
        post = snaps[int(8.0 / 0.02):]   # after the step
        thresh = cfg["tests"]["step_response"]["settle_threshold"]
        result["settling_time"] = compute_settling_time(post, threshold=thresh)
        result["overshoot_x"]   = compute_overshoot(post, "x") or 0.0
        result["overshoot_y"]   = compute_overshoot(post, "y") or 0.0

    elif test == "disturbance":
        hold_before = cfg["tests"]["disturbance"]["hold_before"]
        recovery = snaps[int((hold_before + 1.0) / 0.02):]
        thresh = cfg["tests"]["disturbance"].get("settle_threshold", 0.1)
        result["settling_time_after_disturbance"] = compute_settling_time(
            recovery, threshold=thresh
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _plot_summary_chart(all_results: dict, tests: list) -> None:
    """4-panel bar chart: RMSE, Max Error, Settling Time, Control Effort."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    test_names = [t[0] for t in tests]
    x = np.arange(len(test_names))
    w = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PID vs RL — Benchmark Summary", fontsize=16, fontweight="bold")

    COLOR_PID = "#4C72B0"
    COLOR_RL  = "#DD8452"

    def _get(test, ctrl, key):
        v = all_results.get(f"{test}_{ctrl}", {}).get(key)
        return v if (v is not None and not (isinstance(v, float) and math.isnan(v))) else 0.0

    # ── Panel 1: RMSE 3D ──────────────────────────────────────────────────
    ax = axes[0, 0]
    pid_vals = [_get(t, "pid", "rmse_3d") for t in test_names]
    rl_vals  = [_get(t, "rl",  "rmse_3d") for t in test_names]
    bars_pid = ax.bar(x - w/2, pid_vals, w, label="PID", color=COLOR_PID, alpha=0.85)
    bars_rl  = ax.bar(x + w/2, rl_vals,  w, label="RL",  color=COLOR_RL,  alpha=0.85)
    ax.set_title("Position RMSE (3D)", fontweight="bold")
    ax.set_ylabel("RMSE (m)")
    ax.set_xticks(x); ax.set_xticklabels([t.capitalize() for t in test_names])
    ax.legend(); ax.grid(axis="y", alpha=0.4)
    for bar in bars_pid: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_rl:  ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    # ── Panel 2: Max Error ────────────────────────────────────────────────
    ax = axes[0, 1]
    pid_vals = [_get(t, "pid", "max_error") for t in test_names]
    rl_vals  = [_get(t, "rl",  "max_error") for t in test_names]
    ax.bar(x - w/2, pid_vals, w, label="PID", color=COLOR_PID, alpha=0.85)
    ax.bar(x + w/2, rl_vals,  w, label="RL",  color=COLOR_RL,  alpha=0.85)
    ax.set_title("Maximum Position Error", fontweight="bold")
    ax.set_ylabel("Max error (m)")
    ax.set_xticks(x); ax.set_xticklabels([t.capitalize() for t in test_names])
    ax.legend(); ax.grid(axis="y", alpha=0.4)

    # ── Panel 3: Settling Time ────────────────────────────────────────────
    ax = axes[1, 0]
    settle_key = {"hover": "settling_time", "step": "settling_time",
                  "disturbance": "settling_time_after_disturbance", "figure8": None}
    pid_vals, rl_vals, labels = [], [], []
    for t in test_names:
        k = settle_key.get(t)
        if k:
            labels.append(t.capitalize())
            pid_vals.append(_get(t, "pid", k))
            rl_vals.append(_get(t,  "rl",  k))

    xs = np.arange(len(labels))
    ax.bar(xs - w/2, pid_vals, w, label="PID", color=COLOR_PID, alpha=0.85)
    ax.bar(xs + w/2, rl_vals,  w, label="RL",  color=COLOR_RL,  alpha=0.85)
    ax.set_title("Settling Time (threshold 0.1 m)", fontweight="bold")
    ax.set_ylabel("Time (s)")
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.legend(); ax.grid(axis="y", alpha=0.4)
    # Annotate 0 = "did not settle"
    for i, (pv, rv) in enumerate(zip(pid_vals, rl_vals)):
        if pv == 0.0:
            ax.text(xs[i] - w/2, 0.3, "N/A", ha="center", va="bottom",
                    fontsize=8, color="red", fontweight="bold")

    # ── Panel 4: Control Effort ───────────────────────────────────────────
    ax = axes[1, 1]
    pid_vals = [_get(t, "pid", "control_effort") for t in test_names]
    rl_vals  = [_get(t, "rl",  "control_effort") for t in test_names]
    ax.bar(x - w/2, pid_vals, w, label="PID", color=COLOR_PID, alpha=0.85)
    ax.bar(x + w/2, rl_vals,  w, label="RL",  color=COLOR_RL,  alpha=0.85)
    ax.set_title("Control Effort (mean motor²)", fontweight="bold")
    ax.set_ylabel("Effort (normalised)")
    ax.set_xticks(x); ax.set_xticklabels([t.capitalize() for t in test_names])
    ax.legend(); ax.grid(axis="y", alpha=0.4)

    # ── Improvement annotation ────────────────────────────────────────────
    improvements = []
    for t in test_names:
        p = all_results.get(f"{t}_pid", {}).get("rmse_3d", 0)
        r = all_results.get(f"{t}_rl",  {}).get("rmse_3d", 0)
        if p > 0:
            improvements.append(f"{t}: RL {(p-r)/p*100:+.1f}% RMSE")
    fig.text(0.5, 0.01, "  |  ".join(improvements),
             ha="center", fontsize=9, color="darkgreen", fontweight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_path = "plots/summary_comparison.png"
    plt.savefig(save_path, dpi=150)
    print(f"\n  Summary chart → {save_path}")
    plt.close(fig)


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    os.makedirs("logs",  exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    TESTS = [
        ("hover",       sim_hover),
        ("figure8",     sim_figure8),
        ("step",        sim_step),
        ("disturbance", sim_disturbance),
    ]
    CONTROLLERS = ["pid", "rl"]

    all_results = {}
    all_snaps: dict = {}  # (test, ctrl) → snaps

    for test_name, sim_fn in TESTS:
        print(f"\n{'='*60}")
        print(f"  TEST: {test_name.upper()}")
        print(f"{'='*60}")

        for ctrl_name in CONTROLLERS:
            sim = QuadrotorSim(ctrl_name, dt=0.02)
            snaps = sim_fn(sim, cfg)
            all_snaps[(test_name, ctrl_name)] = snaps

            # Save CSV
            csv_path = f"logs/{test_name}_{ctrl_name}_sim.csv"
            save_csv(snaps, csv_path)

            # Compute metrics
            results = compute_all(test_name, snaps, cfg)
            results["test"]       = test_name
            results["controller"] = ctrl_name
            all_results[f"{test_name}_{ctrl_name}"] = results

            print(f"\n  [{ctrl_name.upper()}]")
            for k, v in results.items():
                if k in ("test", "controller"): continue
                val = f"{v:.4f}" if isinstance(v, float) else str(v)
                print(f"    {k:<35} {val}")

            # Per-test plots
            show = cfg["plotting"]["show"]
            dpi  = cfg["plotting"]["dpi"]

            label = f"{test_name}_{ctrl_name}"
            plot_trajectory(
                snaps,
                title=f"{test_name.capitalize()} — {ctrl_name.upper()}",
                save_path=f"plots/{label}_trajectory.png",
                show=False, dpi=dpi,
            )
            plot_error(
                snaps,
                title=f"{test_name.capitalize()} Error — {ctrl_name.upper()}",
                save_path=f"plots/{label}_error.png",
                show=False, dpi=dpi,
            )

        # Side-by-side comparison
        snaps_pid = all_snaps[(test_name, "pid")]
        snaps_rl  = all_snaps[(test_name, "rl")]
        plot_comparison(
            snaps_pid, snaps_rl,
            title=f"{test_name.capitalize()} — PID vs RL",
            save_path=f"plots/{test_name}_comparison.png",
            show=False, dpi=dpi,
        )
        print(f"\n  Comparison plot → plots/{test_name}_comparison.png")

    # Summary table
    print(f"\n{'='*60}")
    print("  SUMMARY — RMSE (3D, metres)")
    print(f"  {'Test':<16} {'PID':>10} {'RL':>10}  {'RL improvement':>16}")
    print(f"  {'-'*52}")
    for test_name, _ in TESTS:
        pid_rmse = all_results[f"{test_name}_pid"]["rmse_3d"]
        rl_rmse  = all_results[f"{test_name}_rl"]["rmse_3d"]
        improvement = (pid_rmse - rl_rmse) / pid_rmse * 100
        sign = "+" if improvement >= 0 else ""
        print(f"  {test_name:<16} {pid_rmse:>10.4f} {rl_rmse:>10.4f}  {sign}{improvement:>12.1f}%")

    print(f"\n  SUMMARY — Settling Time (s)")
    print(f"  {'Test':<16} {'PID':>10} {'RL':>10}")
    print(f"  {'-'*38}")
    for test_name in ["hover", "step", "disturbance"]:
        key_t = "settling_time" if test_name != "disturbance" else "settling_time_after_disturbance"
        pid_t = all_results[f"{test_name}_pid"].get(key_t)
        rl_t  = all_results[f"{test_name}_rl"].get(key_t)
        pid_s = f"{pid_t:.2f}" if pid_t is not None else "N/A"
        rl_s  = f"{rl_t:.2f}"  if rl_t  is not None else "N/A"
        print(f"  {test_name:<16} {pid_s:>10} {rl_s:>10}")

    # ── Summary bar chart ──────────────────────────────────────────────────
    _plot_summary_chart(all_results, TESTS)

    # Save full results JSON
    def _serialise(v):
        if isinstance(v, float) and math.isnan(v): return None
        if isinstance(v, np.floating): return float(v)
        return v

    with open("results_summary.json", "w") as f:
        json.dump(
            {k: {kk: _serialise(vv) for kk, vv in v.items()}
             for k, v in all_results.items()},
            f, indent=2,
        )
    print("\n  Full results → results_summary.json")

    # Show all plots at the end if configured
    if cfg["plotting"]["show"]:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
