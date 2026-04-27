#!/usr/bin/env python3
"""
PID vs RL comparison plots — baseline + payload tests.

Baseline tests (no payload):
    hover, step, figure8, disturbance

Payload tests (gz_x500_payload_drop model):
    payload_hover  — stability while carrying 0.5 kg payload
    payload_drop   — stability after dropping the payload mid-flight

For each test the script generates:
  - Per-test comparison figure: 3D path, error over time, X/Y top-down, altitude
  - Payload drop figure:        pre-drop vs post-drop error overlay
  - Summary bar-chart:          RMSE + max error across all tests

Usage:
    python compare.py                           # auto-select latest logs
    python compare.py --logs-dir logs           # custom log directory
    python compare.py --out-dir plots           # custom output directory
    python compare.py --no-show                 # save only, don't display
    python compare.py --payload-only            # skip baseline tests
    python compare.py --baseline-only           # skip payload tests
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASELINE_TESTS = ["hover", "step", "figure8", "disturbance"]
PAYLOAD_TESTS  = ["payload_hover", "payload_drop"]
ALL_TESTS      = BASELINE_TESTS + PAYLOAD_TESTS

PID_COLOR  = "#2979FF"   # blue
RL_COLOR   = "#FF6D00"   # orange
SP_COLOR   = "#757575"   # grey
DROP_COLOR = "#E53935"   # red — marks the drop event

# ---------------------------------------------------------------------------
# Log discovery helpers
# ---------------------------------------------------------------------------

def find_latest(logs_dir: str, test: str, ctrl: str) -> str | None:
    pattern    = os.path.join(logs_dir, f"{test}_{ctrl}_2*.csv")
    candidates = sorted(glob.glob(pattern))
    return candidates[-1] if candidates else None


def find_latest_json(logs_dir: str, test: str, ctrl: str) -> dict | None:
    pattern    = os.path.join(logs_dir, f"{test}_{ctrl}_2*_results.json")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        return None
    with open(candidates[-1]) as f:
        return json.load(f)


def load_csv(path: str) -> pd.DataFrame:
    df    = pd.read_csv(path)
    df["t"] = df["timestamp"] - df["timestamp"].iloc[0]
    return df


def error_magnitude(df: pd.DataFrame) -> np.ndarray:
    return np.sqrt(
        (df["pos_x"] - df["sp_x"]) ** 2 +
        (df["pos_y"] - df["sp_y"]) ** 2 +
        (df["pos_z"] - df["sp_z"]) ** 2
    ).to_numpy()


def rmse(df: pd.DataFrame) -> float:
    return float(np.sqrt(np.mean(error_magnitude(df) ** 2)))


def max_error(df: pd.DataFrame) -> float:
    return float(error_magnitude(df).max())


# ---------------------------------------------------------------------------
# Per-test comparison figure  (shared by baseline + payload_hover)
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
    4-panel comparison figure:
      [0] 3D trajectory overlay
      [1] Error magnitude over time
      [2] X/Y top-down path
      [3] Altitude (Z) over time
    """
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        f"{test_name.upper().replace('_', ' ')}  —  PID vs RL",
        fontsize=15, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

    ax3d   = fig.add_subplot(gs[0, 0], projection="3d")
    ax_err = fig.add_subplot(gs[0, 1])
    ax_xy  = fig.add_subplot(gs[0, 2])
    ax_x   = fig.add_subplot(gs[1, 0])
    ax_y   = fig.add_subplot(gs[1, 1])
    ax_z   = fig.add_subplot(gs[1, 2])

    def _pos(df):
        return df["pos_x"].to_numpy(), df["pos_y"].to_numpy(), -df["pos_z"].to_numpy()

    def _sp(df):
        return df["sp_x"].to_numpy(), df["sp_y"].to_numpy(), -df["sp_z"].to_numpy()

    pid_x, pid_y, pid_z = _pos(df_pid)
    rl_x,  rl_y,  rl_z  = _pos(df_rl)
    sp_x,  sp_y,  sp_z  = _sp(df_pid)

    # 3D
    ax3d.plot(sp_x,  sp_y,  sp_z,  "--", color=SP_COLOR,  lw=1.2, alpha=0.7, label="Setpoint")
    ax3d.plot(pid_x, pid_y, pid_z, "-",  color=PID_COLOR, lw=1.5, label="PID")
    ax3d.plot(rl_x,  rl_y,  rl_z,  "-",  color=RL_COLOR,  lw=1.5, label="RL")
    ax3d.set_xlabel("X (m)", fontsize=8); ax3d.set_ylabel("Y (m)", fontsize=8)
    ax3d.set_zlabel("Alt (m)", fontsize=8); ax3d.legend(fontsize=8)
    ax3d.set_title("3D Path", fontsize=10)

    # Error magnitude
    pid_err = error_magnitude(df_pid)
    rl_err  = error_magnitude(df_rl)
    ax_err.plot(df_pid["t"], pid_err, color=PID_COLOR, lw=1.4,
                label=f"PID  RMSE={rmse(df_pid):.3f} m")
    ax_err.plot(df_rl["t"],  rl_err,  color=RL_COLOR,  lw=1.4,
                label=f"RL   RMSE={rmse(df_rl):.3f} m")
    ax_err.set_xlabel("Time (s)"); ax_err.set_ylabel("3D Error (m)")
    ax_err.legend(fontsize=8); ax_err.grid(True, alpha=0.4)
    ax_err.set_title("Error Magnitude", fontsize=10)

    # X/Y top-down
    ax_xy.plot(sp_x,  sp_y,  "--", color=SP_COLOR,  lw=1.2, alpha=0.7, label="Setpoint")
    ax_xy.plot(pid_x, pid_y, "-",  color=PID_COLOR, lw=1.4, label="PID")
    ax_xy.plot(rl_x,  rl_y,  "-",  color=RL_COLOR,  lw=1.4, label="RL")
    ax_xy.set_xlabel("X (m)"); ax_xy.set_ylabel("Y (m)")
    ax_xy.legend(fontsize=8); ax_xy.grid(True, alpha=0.4)
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.set_title("Top-down (X/Y)", fontsize=10)

    # X axis
    ax_x.plot(df_pid["t"], df_pid["sp_x"], "--", color=SP_COLOR,  lw=1.0, alpha=0.7)
    ax_x.plot(df_pid["t"], pid_x,          "-",  color=PID_COLOR, lw=1.3, label="PID")
    ax_x.plot(df_rl["t"],  rl_x,           "-",  color=RL_COLOR,  lw=1.3, label="RL")
    ax_x.set_xlabel("Time (s)"); ax_x.set_ylabel("X (m)")
    ax_x.legend(fontsize=8); ax_x.grid(True, alpha=0.4); ax_x.set_title("X axis", fontsize=10)

    # Y axis
    ax_y.plot(df_pid["t"], df_pid["sp_y"], "--", color=SP_COLOR,  lw=1.0, alpha=0.7)
    ax_y.plot(df_pid["t"], pid_y,          "-",  color=PID_COLOR, lw=1.3, label="PID")
    ax_y.plot(df_rl["t"],  rl_y,           "-",  color=RL_COLOR,  lw=1.3, label="RL")
    ax_y.set_xlabel("Time (s)"); ax_y.set_ylabel("Y (m)")
    ax_y.legend(fontsize=8); ax_y.grid(True, alpha=0.4); ax_y.set_title("Y axis", fontsize=10)

    # Altitude
    ax_z.plot(df_pid["t"], sp_z,  "--", color=SP_COLOR,  lw=1.0, alpha=0.7, label="Setpoint")
    ax_z.plot(df_pid["t"], pid_z, "-",  color=PID_COLOR, lw=1.3, label="PID")
    ax_z.plot(df_rl["t"],  rl_z,  "-",  color=RL_COLOR,  lw=1.3, label="RL")
    ax_z.set_xlabel("Time (s)"); ax_z.set_ylabel("Altitude (m)")
    ax_z.legend(fontsize=8); ax_z.grid(True, alpha=0.4); ax_z.set_title("Altitude (Z)", fontsize=10)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Payload drop comparison figure
# ---------------------------------------------------------------------------

def plot_payload_drop_comparison(
    df_pid: pd.DataFrame,
    df_rl: pd.DataFrame,
    drop_t_pid: float,
    drop_t_rl: float,
    save_path: str,
    show: bool,
    dpi: int,
) -> None:
    """
    3-panel payload drop figure:
      [top-left]  Error magnitude: PID vs RL — full test, drop line marked
      [top-right] Attitude: roll + pitch — PID vs RL around drop moment
      [bottom]    Altitude: drone Z — shows upward jerk at drop
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "PAYLOAD DROP  —  PID vs RL  (pre-drop + drop jerk + recovery)",
        fontsize=14, fontweight="bold",
    )

    ax_err   = axes[0, 0]
    ax_att   = axes[0, 1]
    ax_z     = axes[1, 0]
    ax_spike = axes[1, 1]

    pid_err = error_magnitude(df_pid)
    rl_err  = error_magnitude(df_rl)

    # ── Error magnitude — full test ──────────────────────────────────────
    ax_err.plot(df_pid["t"], pid_err, color=PID_COLOR, lw=1.4, label=f"PID  RMSE={rmse(df_pid):.3f} m")
    ax_err.plot(df_rl["t"],  rl_err,  color=RL_COLOR,  lw=1.4, label=f"RL   RMSE={rmse(df_rl):.3f} m")
    ax_err.axvline(x=drop_t_pid, color=DROP_COLOR, ls="--", lw=1.2, alpha=0.7, label="PID drop")
    ax_err.axvline(x=drop_t_rl,  color=DROP_COLOR, ls=":",  lw=1.2, alpha=0.7, label="RL drop")
    ax_err.set_xlabel("Time (s)"); ax_err.set_ylabel("3D Error (m)")
    ax_err.legend(fontsize=8); ax_err.grid(True, alpha=0.4)
    ax_err.set_title("Error Magnitude (drop ↓ marked)", fontsize=10)

    # ── Altitude over time — shows upward jerk ───────────────────────────
    pid_alt = -df_pid["pos_z"].to_numpy()
    rl_alt  = -df_rl["pos_z"].to_numpy()
    sp_alt  = -df_pid["sp_z"].to_numpy()
    ax_z.plot(df_pid["t"], sp_alt,  "--", color=SP_COLOR,  lw=1.0, alpha=0.7, label="Setpoint")
    ax_z.plot(df_pid["t"], pid_alt, "-",  color=PID_COLOR, lw=1.4, label="PID")
    ax_z.plot(df_rl["t"],  rl_alt,  "-",  color=RL_COLOR,  lw=1.4, label="RL")
    ax_z.axvline(x=drop_t_pid, color=DROP_COLOR, ls="--", lw=1.2, alpha=0.7)
    ax_z.axvline(x=drop_t_rl,  color=DROP_COLOR, ls=":",  lw=1.2, alpha=0.7)
    ax_z.set_xlabel("Time (s)"); ax_z.set_ylabel("Altitude (m)")
    ax_z.legend(fontsize=8); ax_z.grid(True, alpha=0.4)
    ax_z.set_title("Altitude (upward jerk at drop)", fontsize=10)

    # ── Attitude: roll angle around drop ─────────────────────────────────
    pid_roll = np.degrees(df_pid["roll"].to_numpy())
    rl_roll  = np.degrees(df_rl["roll"].to_numpy())
    ax_att.plot(df_pid["t"], pid_roll, color=PID_COLOR, lw=1.3, label="PID roll")
    ax_att.plot(df_rl["t"],  rl_roll,  color=RL_COLOR,  lw=1.3, label="RL roll")
    ax_att.axvline(x=drop_t_pid, color=DROP_COLOR, ls="--", lw=1.2, alpha=0.7)
    ax_att.axvline(x=drop_t_rl,  color=DROP_COLOR, ls=":",  lw=1.2, alpha=0.7)
    ax_att.set_xlabel("Time (s)"); ax_att.set_ylabel("Roll (deg)")
    ax_att.legend(fontsize=8); ax_att.grid(True, alpha=0.4)
    ax_att.set_title("Roll Attitude (attitude coupling)", fontsize=10)

    # ── Post-drop error comparison (bar chart) ───────────────────────────
    post_pid = df_pid[df_pid["t"] >= drop_t_pid]
    post_rl  = df_rl[df_rl["t"]  >= drop_t_rl]

    categories = ["Post-drop RMSE (m)", "Post-drop Max Err (m)"]
    pid_vals = [rmse(post_pid) if len(post_pid) > 0 else 0,
                max_error(post_pid) if len(post_pid) > 0 else 0]
    rl_vals  = [rmse(post_rl)  if len(post_rl)  > 0 else 0,
                max_error(post_rl)  if len(post_rl)  > 0 else 0]

    x = np.arange(len(categories))
    w = 0.35
    bars_pid = ax_spike.bar(x - w/2, pid_vals, w, label="PID", color=PID_COLOR, alpha=0.85)
    bars_rl  = ax_spike.bar(x + w/2, rl_vals,  w, label="RL",  color=RL_COLOR,  alpha=0.85)
    ax_spike.set_xticks(x); ax_spike.set_xticklabels(categories, fontsize=8)
    ax_spike.set_ylabel("Error (m)"); ax_spike.legend(fontsize=8)
    ax_spike.grid(True, axis="y", alpha=0.4)
    ax_spike.set_title("Post-Drop Recovery Quality", fontsize=10)
    for bar in list(bars_pid) + list(bars_rl):
        ax_spike.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
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
    Side-by-side RMSE and Max Error bar charts across all tests.

    metrics = {
        "hover": {"pid_rmse": .., "rl_rmse": .., "pid_max": .., "rl_max": ..},
        ...
    }
    """
    tests    = list(metrics.keys())
    n        = len(tests)
    x        = np.arange(n)
    width    = 0.35
    labels   = [t.replace("_", "\n") for t in tests]

    pid_rmse = [metrics[t]["pid_rmse"] for t in tests]
    rl_rmse  = [metrics[t]["rl_rmse"]  for t in tests]
    pid_max  = [metrics[t]["pid_max"]  for t in tests]
    rl_max   = [metrics[t]["rl_max"]   for t in tests]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("PID vs RL — Summary (Baseline + Payload)", fontsize=14, fontweight="bold")

    for ax, pid_vals, rl_vals, ylabel, title in [
        (ax1, pid_rmse, rl_rmse, "RMSE (m)",      "3D RMSE"),
        (ax2, pid_max,  rl_max,  "Max Error (m)", "Max 3D Error"),
    ]:
        b1 = ax.bar(x - width/2, pid_vals, width, label="PID", color=PID_COLOR, alpha=0.85)
        b2 = ax.bar(x + width/2, rl_vals,  width, label="RL",  color=RL_COLOR,  alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(); ax.grid(True, axis="y", alpha=0.4)
        for bar in list(b1) + list(b2):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8,
            )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Payload drop: extract drop time from JSON results
# ---------------------------------------------------------------------------

def get_drop_time_from_results(results_json: dict | None, hold_before_default: float = 15.0) -> float:
    """
    Returns the elapsed time (seconds from test start) when the payload was dropped.
    Falls back to hold_before_default if results JSON is not available.
    """
    if results_json is None:
        return hold_before_default
    # Settling time is measured from drop — hold_before is not stored directly,
    # but we can use hold_before from config.  As a safe fallback use the default.
    return hold_before_default


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="PID vs RL comparison plots.")
    p.add_argument("--logs-dir",      default="logs")
    p.add_argument("--out-dir",       default="plots")
    p.add_argument("--dpi",           type=int, default=150)
    p.add_argument("--no-show",       action="store_true")
    p.add_argument("--payload-only",  action="store_true", help="Skip baseline tests")
    p.add_argument("--baseline-only", action="store_true", help="Skip payload tests")
    p.add_argument(
        "--payload-drop-t",
        type=float, default=15.0,
        help="Seconds from test start when payload was dropped (default: 15.0)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    show = not args.no_show

    # Change to script directory so relative paths work
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    tests_to_run = []
    if not args.payload_only:
        tests_to_run += BASELINE_TESTS
    if not args.baseline_only:
        tests_to_run += PAYLOAD_TESTS

    summary_metrics = {}
    missing = []

    for test in tests_to_run:
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

        # ── Payload drop: special figure ─────────────────────────────────
        if test == "payload_drop":
            drop_t = args.payload_drop_t
            save_path = os.path.join(args.out_dir, "comparison_payload_drop.png")
            plot_payload_drop_comparison(
                df_pid, df_rl,
                drop_t_pid=drop_t, drop_t_rl=drop_t,
                save_path=save_path, show=show, dpi=args.dpi,
            )
            # For summary, use post-drop RMSE
            post_pid = df_pid[df_pid["t"] >= drop_t]
            post_rl  = df_rl[df_rl["t"]  >= drop_t]
            summary_metrics[test] = {
                "pid_rmse": rmse(post_pid) if len(post_pid) > 0 else float("nan"),
                "rl_rmse":  rmse(post_rl)  if len(post_rl)  > 0 else float("nan"),
                "pid_max":  max_error(post_pid) if len(post_pid) > 0 else float("nan"),
                "rl_max":   max_error(post_rl)  if len(post_rl)  > 0 else float("nan"),
            }
        else:
            # Baseline + payload_hover: standard comparison figure
            save_path = os.path.join(args.out_dir, f"comparison_{test}.png")
            plot_test_comparison(df_pid, df_rl, test, save_path, show, args.dpi)
            summary_metrics[test] = {
                "pid_rmse": rmse(df_pid),
                "rl_rmse":  rmse(df_rl),
                "pid_max":  max_error(df_pid),
                "rl_max":   max_error(df_rl),
            }

    # ── Summary ───────────────────────────────────────────────────────────
    if summary_metrics:
        print(f"\n[SUMMARY]")
        print(f"  {'Test':<18} {'PID RMSE':>10} {'RL RMSE':>10} {'PID Max':>10} {'RL Max':>10}")
        print("  " + "-" * 62)
        for test, m in summary_metrics.items():
            pid_r = f"{m['pid_rmse']:.4f}" if not np.isnan(m['pid_rmse']) else "  n/a  "
            rl_r  = f"{m['rl_rmse']:.4f}"  if not np.isnan(m['rl_rmse'])  else "  n/a  "
            pid_x = f"{m['pid_max']:.4f}"  if not np.isnan(m['pid_max'])  else "  n/a  "
            rl_x  = f"{m['rl_max']:.4f}"   if not np.isnan(m['rl_max'])   else "  n/a  "
            print(f"  {test:<18} {pid_r:>10} {rl_r:>10} {pid_x:>10} {rl_x:>10}")

        summary_path = os.path.join(args.out_dir, "comparison_summary.png")
        plot_summary(summary_metrics, summary_path, show, args.dpi)

    if missing:
        print(f"\nSkipped tests (no matching logs): {missing}")


if __name__ == "__main__":
    main()
