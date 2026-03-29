#!/usr/bin/env python3
"""
PID vs RL comparison plots.

For each test (hover, step, figure8, disturbance):
  - 3D trajectory overlay: PID path, RL path, setpoint path on the same axes
  - Per-axis time series: X/Y/Z actual vs setpoint for both controllers
  - Error magnitude over time: PID vs RL on the same axes

Plus a summary bar-chart comparing key metrics across all tests.

Usage:
    python compare.py                     # auto-select latest logs
    python compare.py --logs-dir logs     # custom log directory
    python compare.py --out-dir plots     # custom output directory
    python compare.py --no-show           # save only, don't display
"""

import argparse
import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TESTS = ["hover", "step", "figure8", "disturbance"]

PID_COLOR   = "#2979FF"   # blue
RL_COLOR    = "#FF6D00"   # orange
SP_COLOR    = "#757575"   # grey
PID_LABEL   = "PID"
RL_LABEL    = "RL"
SP_LABEL    = "Setpoint"

# ---------------------------------------------------------------------------
# Log discovery
# ---------------------------------------------------------------------------

def find_latest(logs_dir: str, test: str, ctrl: str) -> str | None:
    """Return path of the most-recently timestamped CSV for (test, ctrl)."""
    pattern = os.path.join(logs_dir, f"{test}_{ctrl}_2*.csv")
    candidates = sorted(glob.glob(pattern))
    return candidates[-1] if candidates else None


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["t"] = df["timestamp"] - df["timestamp"].iloc[0]
    return df


def error_magnitude(df: pd.DataFrame) -> np.ndarray:
    ex = df["pos_x"] - df["sp_x"]
    ey = df["pos_y"] - df["sp_y"]
    ez = df["pos_z"] - df["sp_z"]
    return np.sqrt(ex**2 + ey**2 + ez**2).to_numpy()


def rmse(df: pd.DataFrame) -> float:
    return float(np.sqrt(np.mean(error_magnitude(df) ** 2)))


def max_error(df: pd.DataFrame) -> float:
    return float(error_magnitude(df).max())


# ---------------------------------------------------------------------------
# Per-test comparison plot
# ---------------------------------------------------------------------------

def plot_test_comparison(
    df_pid: pd.DataFrame,
    df_rl: pd.DataFrame,
    test_name: str,
    save_path: str,
    show: bool,
    dpi: int,
) -> None:
    """
    One figure per test with 4 panels:
      [0] 3D trajectory overlay
      [1] Error magnitude over time (PID vs RL)
      [2] X/Y top-down path overlay
      [3] Altitude (Z) over time
    """
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(f"{test_name.upper()}  —  PID vs RL Comparison", fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

    ax3d  = fig.add_subplot(gs[0, 0], projection="3d")
    ax_err = fig.add_subplot(gs[0, 1])
    ax_xy  = fig.add_subplot(gs[0, 2])
    ax_x   = fig.add_subplot(gs[1, 0])
    ax_y   = fig.add_subplot(gs[1, 1])
    ax_z   = fig.add_subplot(gs[1, 2])

    # ── helpers ──────────────────────────────────────────────────────────
    def _pos(df):
        return (df["pos_x"].to_numpy(),
                df["pos_y"].to_numpy(),
                -df["pos_z"].to_numpy())   # NED → altitude up

    def _sp(df):
        return (df["sp_x"].to_numpy(),
                df["sp_y"].to_numpy(),
                -df["sp_z"].to_numpy())

    pid_x, pid_y, pid_z = _pos(df_pid)
    rl_x,  rl_y,  rl_z  = _pos(df_rl)
    sp_x, sp_y, sp_z     = _sp(df_pid)   # setpoint is same for both

    # ── 3D trajectory ────────────────────────────────────────────────────
    ax3d.plot(sp_x,  sp_y,  sp_z,  "--", color=SP_COLOR,  lw=1.2, alpha=0.7, label=SP_LABEL)
    ax3d.plot(pid_x, pid_y, pid_z, "-",  color=PID_COLOR, lw=1.5, label=PID_LABEL)
    ax3d.plot(rl_x,  rl_y,  rl_z,  "-",  color=RL_COLOR,  lw=1.5, label=RL_LABEL)
    # mark start
    ax3d.scatter(*[v[0:1] for v in (pid_x, pid_y, pid_z)], color=PID_COLOR, s=40, zorder=5)
    ax3d.scatter(*[v[0:1] for v in (rl_x,  rl_y,  rl_z)],  color=RL_COLOR,  s=40, zorder=5)
    ax3d.set_xlabel("X (m)", fontsize=8)
    ax3d.set_ylabel("Y (m)", fontsize=8)
    ax3d.set_zlabel("Alt (m)", fontsize=8)
    ax3d.legend(fontsize=8)
    ax3d.set_title("3D Path", fontsize=10)

    # ── Error magnitude over time ─────────────────────────────────────────
    pid_err = error_magnitude(df_pid)
    rl_err  = error_magnitude(df_rl)
    ax_err.plot(df_pid["t"], pid_err, color=PID_COLOR, lw=1.4,
                label=f"{PID_LABEL}  RMSE={rmse(df_pid):.3f} m")
    ax_err.plot(df_rl["t"],  rl_err,  color=RL_COLOR,  lw=1.4,
                label=f"{RL_LABEL}  RMSE={rmse(df_rl):.3f} m")
    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("3D Error (m)")
    ax_err.legend(fontsize=8)
    ax_err.grid(True, alpha=0.4)
    ax_err.set_title("Error Magnitude", fontsize=10)

    # ── Top-down X/Y path ────────────────────────────────────────────────
    ax_xy.plot(sp_x,  sp_y,  "--", color=SP_COLOR,  lw=1.2, alpha=0.7, label=SP_LABEL)
    ax_xy.plot(pid_x, pid_y, "-",  color=PID_COLOR, lw=1.4, label=PID_LABEL)
    ax_xy.plot(rl_x,  rl_y,  "-",  color=RL_COLOR,  lw=1.4, label=RL_LABEL)
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.legend(fontsize=8)
    ax_xy.grid(True, alpha=0.4)
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.set_title("Top-down (X/Y)", fontsize=10)

    # ── X over time ──────────────────────────────────────────────────────
    ax_x.plot(df_pid["t"], df_pid["sp_x"], "--", color=SP_COLOR,  lw=1.0, alpha=0.7)
    ax_x.plot(df_pid["t"], pid_x,          "-",  color=PID_COLOR, lw=1.3, label=PID_LABEL)
    ax_x.plot(df_rl["t"],  rl_x,           "-",  color=RL_COLOR,  lw=1.3, label=RL_LABEL)
    ax_x.set_xlabel("Time (s)")
    ax_x.set_ylabel("X (m)")
    ax_x.legend(fontsize=8)
    ax_x.grid(True, alpha=0.4)
    ax_x.set_title("X axis", fontsize=10)

    # ── Y over time ──────────────────────────────────────────────────────
    ax_y.plot(df_pid["t"], df_pid["sp_y"], "--", color=SP_COLOR,  lw=1.0, alpha=0.7)
    ax_y.plot(df_pid["t"], pid_y,          "-",  color=PID_COLOR, lw=1.3, label=PID_LABEL)
    ax_y.plot(df_rl["t"],  rl_y,           "-",  color=RL_COLOR,  lw=1.3, label=RL_LABEL)
    ax_y.set_xlabel("Time (s)")
    ax_y.set_ylabel("Y (m)")
    ax_y.legend(fontsize=8)
    ax_y.grid(True, alpha=0.4)
    ax_y.set_title("Y axis", fontsize=10)

    # ── Altitude over time ───────────────────────────────────────────────
    ax_z.plot(df_pid["t"], sp_z,   "--", color=SP_COLOR,  lw=1.0, alpha=0.7, label=SP_LABEL)
    ax_z.plot(df_pid["t"], pid_z,  "-",  color=PID_COLOR, lw=1.3, label=PID_LABEL)
    ax_z.plot(df_rl["t"],  rl_z,   "-",  color=RL_COLOR,  lw=1.3, label=RL_LABEL)
    ax_z.set_xlabel("Time (s)")
    ax_z.set_ylabel("Altitude (m)")
    ax_z.legend(fontsize=8)
    ax_z.grid(True, alpha=0.4)
    ax_z.set_title("Altitude (Z)", fontsize=10)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary bar-chart
# ---------------------------------------------------------------------------

def plot_summary(metrics: dict, save_path: str, show: bool, dpi: int) -> None:
    """
    metrics = {
        "hover":       {"pid_rmse": .., "rl_rmse": .., "pid_max": .., "rl_max": ..},
        "step":        {...},
        ...
    }
    """
    tests  = list(metrics.keys())
    n      = len(tests)
    x      = np.arange(n)
    width  = 0.35

    pid_rmse = [metrics[t]["pid_rmse"] for t in tests]
    rl_rmse  = [metrics[t]["rl_rmse"]  for t in tests]
    pid_max  = [metrics[t]["pid_max"]  for t in tests]
    rl_max   = [metrics[t]["rl_max"]   for t in tests]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("PID vs RL — Summary Metrics", fontsize=14, fontweight="bold")

    # RMSE
    bars1 = ax1.bar(x - width/2, pid_rmse, width, label=PID_LABEL, color=PID_COLOR, alpha=0.85)
    bars2 = ax1.bar(x + width/2, rl_rmse,  width, label=RL_LABEL,  color=RL_COLOR,  alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([t.capitalize() for t in tests])
    ax1.set_ylabel("RMSE (m)")
    ax1.set_title("3D RMSE")
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.4)
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    # Max error
    bars3 = ax2.bar(x - width/2, pid_max, width, label=PID_LABEL, color=PID_COLOR, alpha=0.85)
    bars4 = ax2.bar(x + width/2, rl_max,  width, label=RL_LABEL,  color=RL_COLOR,  alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([t.capitalize() for t in tests])
    ax2.set_ylabel("Max Error (m)")
    ax2.set_title("Max 3D Error")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.4)
    for bar in bars3:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars4:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="PID vs RL comparison plots.")
    p.add_argument("--logs-dir", default="logs")
    p.add_argument("--out-dir",  default="plots")
    p.add_argument("--dpi",      type=int, default=150)
    p.add_argument("--no-show",  action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    show = not args.no_show

    # Change to script directory so relative paths work
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    summary_metrics = {}
    missing = []

    for test in TESTS:
        pid_path = find_latest(args.logs_dir, test, "pid")
        rl_path  = find_latest(args.logs_dir, test, "rl")

        if pid_path is None or rl_path is None:
            missing.append(test)
            print(f"  [SKIP] {test}: missing {'PID' if pid_path is None else 'RL'} log")
            continue

        print(f"\n[{test.upper()}]")
        print(f"  PID → {pid_path}")
        print(f"  RL  → {rl_path}")

        df_pid = load_csv(pid_path)
        df_rl  = load_csv(rl_path)

        save_path = os.path.join(args.out_dir, f"comparison_{test}.png")
        plot_test_comparison(df_pid, df_rl, test, save_path, show, args.dpi)

        summary_metrics[test] = {
            "pid_rmse": rmse(df_pid),
            "rl_rmse":  rmse(df_rl),
            "pid_max":  max_error(df_pid),
            "rl_max":   max_error(df_rl),
        }

    if summary_metrics:
        print("\n[SUMMARY]")
        print(f"  {'Test':<14} {'PID RMSE':>10} {'RL RMSE':>10} {'PID Max':>10} {'RL Max':>10}")
        print("  " + "-" * 54)
        for test, m in summary_metrics.items():
            print(f"  {test:<14} {m['pid_rmse']:>10.4f} {m['rl_rmse']:>10.4f} "
                  f"{m['pid_max']:>10.4f} {m['rl_max']:>10.4f}")

        summary_path = os.path.join(args.out_dir, "comparison_summary.png")
        plot_summary(summary_metrics, summary_path, show, args.dpi)

    if missing:
        print(f"\nSkipped tests (no matching logs): {missing}")


if __name__ == "__main__":
    main()
