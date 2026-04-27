#!/usr/bin/env python3
"""
make_video.py — Generate an animated MP4 from a benchmark CSV log.

The video shows a moving dot (actual position) chasing a moving cross
(setpoint) on a top-down XY view, with altitude and 3D error plotted
alongside in real time.

Usage:
    python make_video.py logs/payload_drop_rl_20260415_140403.csv
    python make_video.py logs/figure8_pid_20260415_143556.csv --fps 30
    python make_video.py logs/          # batch: process all CSVs in a folder
    python make_video.py logs/ --out videos/

Output:
    Same directory as the CSV (or --out), with .mp4 extension.
    Example: logs/payload_drop_rl_20260415_140403.mp4
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"timestamp", "pos_x", "pos_y", "pos_z", "sp_x", "sp_y", "sp_z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df["t"] = df["timestamp"] - df["timestamp"].iloc[0]
    # Convert NED z → up-positive for display
    df["alt"]    = -df["pos_z"]
    df["sp_alt"] = -df["sp_z"]
    df["err3d"]  = np.sqrt(
        (df["pos_x"] - df["sp_x"])**2 +
        (df["pos_y"] - df["sp_y"])**2 +
        (df["pos_z"] - df["sp_z"])**2
    )
    return df


def _title_from_path(csv_path: str) -> str:
    name = os.path.splitext(os.path.basename(csv_path))[0]
    # e.g. payload_drop_rl_20260415_140403 → "payload drop rl  20260415 140403"
    parts = name.split("_")
    # rejoin: test name (everything before the last timestamp parts)
    return name.replace("_", "  ")


# ---------------------------------------------------------------------------
# Core animation builder
# ---------------------------------------------------------------------------

def make_video(csv_path: str, out_path: str, fps: int = 25) -> None:
    df = _load_csv(csv_path)
    title = _title_from_path(csv_path)

    # Down-sample to target fps so large logs don't produce huge files.
    # Real sample rate ≈ 50 Hz; take every nth row to match desired fps.
    duration_s = df["t"].iloc[-1]
    n_frames   = max(int(duration_s * fps), 1)
    step       = max(len(df) // n_frames, 1)
    df_ds      = df.iloc[::step].reset_index(drop=True)
    n          = len(df_ds)

    # ── Layout ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 7), facecolor="#1a1a2e")
    fig.suptitle(title, color="white", fontsize=11, y=0.98)

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35,
                          left=0.06, right=0.97, top=0.93, bottom=0.08)

    ax_xy  = fig.add_subplot(gs[:, 0])   # top-down XY (tall, left)
    ax_alt = fig.add_subplot(gs[0, 1:])  # altitude timeline (top-right)
    ax_err = fig.add_subplot(gs[1, 1:])  # 3D error timeline (bottom-right)

    dark = "#1a1a2e"
    for ax in (ax_xy, ax_alt, ax_err):
        ax.set_facecolor("#0f3460")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.grid(True, color="#334466", linewidth=0.5, alpha=0.6)

    # XY axis setup
    pad = 0.5
    all_x = np.concatenate([df_ds["pos_x"], df_ds["sp_x"]])
    all_y = np.concatenate([df_ds["pos_y"], df_ds["sp_y"]])
    x_min, x_max = all_x.min() - pad, all_x.max() + pad
    y_min, y_max = all_y.min() - pad, all_y.max() + pad
    ax_xy.set_xlim(x_min, x_max)
    ax_xy.set_ylim(y_min, y_max)
    ax_xy.set_aspect("equal")
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("Top-down (XY)")

    # Ghost trail (full path, faint)
    ax_xy.plot(df_ds["pos_x"], df_ds["pos_y"],
               color="#4488ff", alpha=0.15, linewidth=0.8, zorder=1)
    ax_xy.plot(df_ds["sp_x"],  df_ds["sp_y"],
               color="#ff6666", alpha=0.15, linewidth=0.8, linestyle="--", zorder=1)

    # Live trail lines
    trail_actual, = ax_xy.plot([], [], color="#4488ff", linewidth=1.2,
                                alpha=0.7, zorder=2, label="Actual")
    trail_sp,     = ax_xy.plot([], [], color="#ff6666", linewidth=1.0,
                                alpha=0.7, linestyle="--", zorder=2, label="Setpoint")

    # Moving dots
    dot_actual, = ax_xy.plot([], [], "o", color="#88bbff", markersize=8,
                              zorder=5, label="_")
    dot_sp,     = ax_xy.plot([], [], "x", color="#ff4444", markersize=10,
                              markeredgewidth=2, zorder=5, label="_")
    ax_xy.legend(loc="upper right", fontsize=7,
                 facecolor="#0f3460", labelcolor="white", framealpha=0.8)

    # Time text
    time_text = ax_xy.text(0.02, 0.97, "", transform=ax_xy.transAxes,
                            color="white", fontsize=8, va="top",
                            fontfamily="monospace")

    # Altitude panel
    t_full  = df_ds["t"].values
    ax_alt.plot(t_full, df_ds["alt"],    color="#88bbff", linewidth=0.8, alpha=0.4)
    ax_alt.plot(t_full, df_ds["sp_alt"], color="#ff6666", linewidth=0.8,
                linestyle="--", alpha=0.4)
    alt_line_a, = ax_alt.plot([], [], color="#88bbff", linewidth=1.5, label="Actual")
    alt_line_s, = ax_alt.plot([], [], color="#ff6666", linewidth=1.2,
                               linestyle="--", label="Setpoint")
    vline_alt = ax_alt.axvline(0, color="white", linewidth=0.8, alpha=0.6)
    ax_alt.set_xlim(0, t_full[-1])
    alt_vals = np.concatenate([df_ds["alt"], df_ds["sp_alt"]])
    ax_alt.set_ylim(alt_vals.min() - 0.3, alt_vals.max() + 0.3)
    ax_alt.set_ylabel("Altitude (m)")
    ax_alt.set_title("Altitude vs Setpoint")
    ax_alt.legend(loc="upper right", fontsize=7,
                  facecolor="#0f3460", labelcolor="white", framealpha=0.8)

    # Error panel
    ax_err.plot(t_full, df_ds["err3d"], color="#ffcc44", linewidth=0.8, alpha=0.4)
    err_line, = ax_err.plot([], [], color="#ffcc44", linewidth=1.5, label="3D error")
    vline_err = ax_err.axvline(0, color="white", linewidth=0.8, alpha=0.6)
    ax_err.set_xlim(0, t_full[-1])
    ax_err.set_ylim(0, df_ds["err3d"].max() * 1.15 + 0.01)
    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("Error (m)")
    ax_err.set_title("3D Position Error")
    ax_err.legend(loc="upper right", fontsize=7,
                  facecolor="#0f3460", labelcolor="white", framealpha=0.8)

    # ── Trail length (show last N frames) ─────────────────────────────────
    TRAIL = min(100, n)

    def update(frame_idx):
        i = frame_idx
        t_now = df_ds["t"].iloc[i]

        # Trail window
        t_start = max(0, i - TRAIL)
        tx = df_ds["pos_x"].iloc[t_start:i+1].values
        ty = df_ds["pos_y"].iloc[t_start:i+1].values
        sx = df_ds["sp_x"].iloc[t_start:i+1].values
        sy = df_ds["sp_y"].iloc[t_start:i+1].values

        trail_actual.set_data(tx, ty)
        trail_sp.set_data(sx, sy)
        dot_actual.set_data([df_ds["pos_x"].iloc[i]], [df_ds["pos_y"].iloc[i]])
        dot_sp.set_data([df_ds["sp_x"].iloc[i]],     [df_ds["sp_y"].iloc[i]])
        time_text.set_text(f"t = {t_now:6.1f} s")

        # Altitude lines
        alt_line_a.set_data(t_full[:i+1], df_ds["alt"].iloc[:i+1])
        alt_line_s.set_data(t_full[:i+1], df_ds["sp_alt"].iloc[:i+1])
        vline_alt.set_xdata([t_now, t_now])

        # Error line
        err_line.set_data(t_full[:i+1], df_ds["err3d"].iloc[:i+1])
        vline_err.set_xdata([t_now, t_now])

        return (trail_actual, trail_sp, dot_actual, dot_sp, time_text,
                alt_line_a, alt_line_s, vline_alt, err_line, vline_err)

    ani = animation.FuncAnimation(
        fig, update, frames=n,
        interval=1000 // fps,
        blit=True,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Use ffmpeg (MP4) if available, otherwise fall back to Pillow (GIF).
    import shutil
    if shutil.which("ffmpeg"):
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800,
                                        extra_args=["-vcodec", "libx264",
                                                    "-pix_fmt", "yuv420p"])
        ani.save(out_path, writer=writer)
    else:
        gif_path = os.path.splitext(out_path)[0] + ".gif"
        writer = animation.PillowWriter(fps=fps)
        ani.save(gif_path, writer=writer)
        out_path = gif_path
        print("  (ffmpeg not found — saved as GIF. Install ffmpeg for MP4: sudo apt install ffmpeg)")

    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate animated MP4 from a drone benchmark CSV log."
    )
    parser.add_argument(
        "input",
        help="Path to a CSV file or a directory of CSV files.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output directory for MP4 files (default: same as input).",
    )
    parser.add_argument(
        "--fps", type=int, default=25,
        help="Frames per second for the output video (default: 25).",
    )
    args = parser.parse_args()

    # Collect CSV files
    if os.path.isdir(args.input):
        csv_files = sorted(
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.endswith(".csv")
        )
        if not csv_files:
            print(f"No CSV files found in {args.input}")
            sys.exit(1)
    else:
        csv_files = [args.input]

    out_dir = args.out

    for csv_path in csv_files:
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            mp4_path = os.path.join(out_dir, stem + ".mp4")
        else:
            mp4_path = os.path.join(os.path.dirname(csv_path), stem + ".mp4")

        print(f"Processing {os.path.basename(csv_path)} ...")
        try:
            make_video(csv_path, mp4_path, fps=args.fps)
        except Exception as exc:
            plt.close("all")
            print(f"  ERROR: {exc}")


if __name__ == "__main__":
    main()
