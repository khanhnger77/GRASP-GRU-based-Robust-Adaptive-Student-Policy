#!/usr/bin/env python3
"""Render README demo GIFs (PID vs GRASP) from benchmark CSV logs.

Usage:
    python make_readme_gifs.py                 # writes ../assets/demo_*.gif
    python make_readme_gifs.py --only figure8

Requires matplotlib, pandas, numpy and ffmpeg on PATH.
"""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyBboxPatch

HERE = Path(__file__).resolve().parent
LOGS = HERE / "logs_ok"
OUT = HERE.parent / "assets"

BG = "#ffffff"
PANEL = "#ffffff"
GRID = "#d0d7de"
TEXT = "#1f2328"
MUTED = "#57606a"
PID_C = "#1f77b4"
RL_C = "#ff7f0e"
SP_C = "#24292f"
GOLD = "#2ca02c"  # payload-release marker

LOG_FILES = {
    "figure8": ("figure8_pid_20260510_015844.csv", "figure8_rl_20260511_145657.csv"),
    "payload_drop": ("payload_drop_pid_20260511_174223.csv", "payload_drop_rl_20260511_170512.csv"),
}
DROP_T = 15.0  # payload release time in the payload_drop runs (s after log start)


def load(name: str, fps: float, speed: float) -> pd.DataFrame:
    """Load a log, resample it onto the GIF frame clock and add error columns."""
    d = pd.read_csv(LOGS / name)
    t = (d.timestamp - d.timestamp.iloc[0]).to_numpy()
    tf = np.arange(0.0, t[-1], speed / fps)
    out = {"t": tf}
    for c in ["pos_x", "pos_y", "pos_z", "sp_x", "sp_y", "sp_z", "roll", "pitch"]:
        out[c] = np.interp(tf, t, d[c].to_numpy())
    r = pd.DataFrame(out)
    r["ex"] = r.pos_x - r.sp_x
    r["ey"] = r.pos_y - r.sp_y
    r["ez"] = -(r.pos_z - r.sp_z)  # NED -> up-positive
    r["err"] = np.sqrt(r.ex**2 + r.ey**2 + r.ez**2)
    r["rmse"] = np.sqrt(np.cumsum(r.err**2) / np.arange(1, len(r) + 1))
    return r


def style_ax(ax):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color("#8c959f")
        s.set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, lw=0.5, ls=":")


def glow_trail(ax, x, y, color, n_tail):
    """Fading trail with a soft glow: older points thinner and more transparent."""
    x, y = x[-n_tail:], y[-n_tail:]
    if len(x) < 2:
        return
    pts = np.stack([x, y], 1).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], 1)
    a = np.linspace(0.0, 1.0, len(segs)) ** 1.5
    rgba = np.array(matplotlib.colors.to_rgba(color))
    for lw, scale in [(6, 0.10), (2.0, 1.0)]:
        cols = np.tile(rgba, (len(segs), 1))
        cols[:, 3] = a * scale
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw, capstyle="round"))


def drone(ax, x, y, color, size):
    ax.add_patch(Circle((x, y), size * 2.2, color=color, alpha=0.15, lw=0))
    ax.add_patch(Circle((x, y), size, color=color, lw=0, zorder=5))
    ax.add_patch(Circle((x, y), size * 0.45, color=BG, lw=0, zorder=6))


def header(fig, title, subtitle):
    fig.text(0.03, 0.935, title, color=TEXT, fontsize=19, weight="bold", va="center")
    fig.text(0.03, 0.875, subtitle, color=MUTED, fontsize=11, va="center")
    fig.text(0.97, 0.935, "GRASP · Payload-Raptor", color=TEXT, fontsize=11,
             weight="bold", ha="right", va="center")
    fig.text(0.97, 0.875, "PX4 SITL · Gazebo X500 · real logs", color=MUTED,
             fontsize=9.5, ha="right", va="center")


def badge(ax, label, color, err, rmse):
    ax.text(0.03, 0.95, label, transform=ax.transAxes, color=color, fontsize=13,
            weight="bold", va="top")
    ax.text(0.97, 0.95, f"error  {err:5.2f} m", transform=ax.transAxes, color=TEXT,
            fontsize=10.5, family="DejaVu Sans Mono", ha="right", va="top")
    ax.text(0.97, 0.86, f"RMSE   {rmse:5.2f} m", transform=ax.transAxes, color=color,
            fontsize=10.5, family="DejaVu Sans Mono", ha="right", va="top", weight="bold")


# ── Scene 1: figure-8 tracking, top-down race ────────────────────────────────
def render_figure8(frames_dir: Path, fps: float, speed: float, dpi: int):
    pid, rl = (load(f, fps, speed) for f in LOG_FILES["figure8"])
    n = min(len(pid), len(rl))
    pad = 0.4
    xlim = (min(pid.pos_x.min(), rl.pos_x.min()) - pad, max(pid.pos_x.max(), rl.pos_x.max()) + pad)
    ylim = (min(pid.pos_y.min(), rl.pos_y.min()) - pad, max(pid.pos_y.max(), rl.pos_y.max()) + pad)
    emax = max(pid.err.max(), rl.err.max()) * 1.1
    tail = int(fps * 2.5)

    for i in range(n):
        fig = plt.figure(figsize=(10, 5.6), facecolor=BG)
        header(fig, "Figure-8 tracking in wind", f"PID vs GRASP  ·  {speed:.0f}× speed")
        axes = [fig.add_axes([0.04 + k * 0.48, 0.30, 0.44, 0.52]) for k in range(2)]
        for ax, d, lab, c in zip(axes, (pid, rl), ("PID", "GRASP (RL)"), (PID_C, RL_C)):
            style_ax(ax)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.plot(d.sp_x[:n], d.sp_y[:n], color=SP_C, lw=1, ls=(0, (4, 4)), alpha=0.35)
            glow_trail(ax, d.pos_x[: i + 1].to_numpy(), d.pos_y[: i + 1].to_numpy(), c, tail)
            ax.plot(d.sp_x[i], d.sp_y[i], "x", color=SP_C, ms=7, mew=1.6, alpha=0.8)
            drone(ax, d.pos_x[i], d.pos_y[i], c, 0.07)
            badge(ax, lab, c, d.err[i], d.rmse[i])

        ax = fig.add_axes([0.06, 0.07, 0.88, 0.16])
        style_ax(ax)
        ax.set_xlim(0, pid.t[n - 1])
        ax.set_ylim(0, emax)
        ax.set_ylabel("error (m)", color=MUTED, fontsize=9)
        for d, c in ((pid, PID_C), (rl, RL_C)):
            ax.fill_between(d.t[: i + 1], 0, d.err[: i + 1], color=c, alpha=0.15, lw=0)
            ax.plot(d.t[: i + 1], d.err[: i + 1], color=c, lw=1.6)
        ax.axvline(pid.t[i], color=MUTED, lw=0.8, alpha=0.6)
        ax.text(0.995, 0.88, f"t = {pid.t[i]:4.1f} s", transform=ax.transAxes,
                color=MUTED, fontsize=9, ha="right", va="top", family="DejaVu Sans Mono")
        fig.savefig(frames_dir / f"{i:05d}.png", dpi=dpi, facecolor=BG)
        plt.close(fig)
    return n


# ── Scene 2: payload drop, top-down bullseye + 3D error ──────────────────────
def render_payload_drop(frames_dir: Path, fps: float, speed: float, dpi: int):
    pid, rl = (load(f, fps, speed) for f in LOG_FILES["payload_drop"])
    n = min(len(pid), len(rl))
    R = max(np.hypot(pid.ex, pid.ey).max(), np.hypot(rl.ex, rl.ey).max()) * 1.1
    emax = max(pid.err.max(), rl.err.max()) * 1.1
    tail = int(fps * 3)

    for i in range(n):
        t = pid.t[i]
        dropped = t >= DROP_T
        flash = dropped and t < DROP_T + 1.2
        fig = plt.figure(figsize=(10, 5.6), facecolor=BG)
        header(fig, "Mid-air payload drop", f"Hover with payload, release at t = {DROP_T:.0f} s  ·  {speed:.0f}× speed")

        for k, (d, lab, c) in enumerate(((pid, "PID", PID_C), (rl, "GRASP", RL_C))):
            ax = fig.add_axes([0.03 + k * 0.235, 0.36, 0.22, 0.44])
            style_ax(ax)
            ax.grid(False)
            ax.set_xlim(-R, R)
            ax.set_ylim(-R, R)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for r in np.arange(0.25, R, 0.25):
                ax.add_patch(Circle((0, 0), r, fill=False, color=GRID, lw=0.8, ls=":"))
            ax.plot([-R, R], [0, 0], color=GRID, lw=0.6)
            ax.plot([0, 0], [-R, R], color=GRID, lw=0.6)
            ax.plot(0, 0, "x", color=SP_C, ms=7, mew=1.6)
            glow_trail(ax, d.ex[: i + 1].to_numpy(), d.ey[: i + 1].to_numpy(), c, tail)
            drone(ax, d.ex[i], d.ey[i], c, R * 0.035)
            ax.text(0.05, 0.95, lab, transform=ax.transAxes, color=c, fontsize=12, weight="bold", va="top")
            status, sc = ("payload released", GOLD) if dropped else ("payload attached", MUTED)
            ax.text(0.5, -0.06, status, transform=ax.transAxes, color=sc, fontsize=9, ha="center", va="top")
            if flash:
                for s in ax.spines.values():
                    s.set_color(GOLD)
                    s.set_linewidth(2)
        fig.text(0.26, 0.255, "top-down position error  (rings = 0.25 m)", color=MUTED, fontsize=9, ha="center")

        def post(d):
            m = (d.t >= DROP_T) & (d.t <= t)
            return np.sqrt(np.mean(d.err[m] ** 2)) if t > DROP_T + 0.5 else None
        p, r = post(pid), post(rl)
        card = fig.add_axes([0.03, 0.06, 0.455, 0.15])
        card.set_xlim(0, 1)
        card.set_ylim(0, 1)
        card.axis("off")
        card.add_patch(FancyBboxPatch((0.005, 0.05), 0.99, 0.9, boxstyle="round,pad=0,rounding_size=0.05",
                                      facecolor="#f6f8fa", edgecolor=GRID))
        card.text(0.04, 0.72, "post-drop RMSE", color=MUTED, fontsize=9.5, va="center")
        if p is None:
            card.text(0.04, 0.32, "waiting for release…", color=MUTED, fontsize=11, va="center")
        else:
            card.text(0.04, 0.32, f"PID {p:4.2f} m", color=PID_C, fontsize=14, weight="bold",
                      va="center", family="DejaVu Sans Mono")
            card.text(0.45, 0.32, f"GRASP {r:4.2f} m", color=RL_C, fontsize=14, weight="bold",
                      va="center", family="DejaVu Sans Mono")
            card.text(0.96, 0.72, f"{p / r:3.1f}× lower", color=TEXT, fontsize=11, weight="bold",
                      ha="right", va="center")

        ax2 = fig.add_axes([0.55, 0.10, 0.42, 0.70])
        style_ax(ax2)
        ax2.set_xlim(0, pid.t[n - 1])
        ax2.set_ylim(0, emax)
        ax2.set_xlabel("time (s)", color=MUTED, fontsize=9)
        ax2.set_ylabel("3D position error (m)", color=MUTED, fontsize=9)
        ax2.axvline(DROP_T, color=GOLD, lw=1.2, ls="--")
        ax2.text(DROP_T + 0.4, emax * 0.97, "payload\nreleased", color=GOLD, fontsize=9, va="top")
        for d, lab, c in ((pid, "PID", PID_C), (rl, "GRASP", RL_C)):
            ax2.fill_between(d.t[: i + 1], 0, d.err[: i + 1], color=c, alpha=0.10, lw=0)
            ax2.plot(d.t[: i + 1], d.err[: i + 1], color=c, lw=1.6, label=lab)
            ax2.plot(d.t[i], d.err[i], "o", color=c, ms=5)
        ax2.legend(loc="upper left", frameon=False, labelcolor=TEXT, fontsize=9.5)
        fig.savefig(frames_dir / f"{i:05d}.png", dpi=dpi, facecolor=BG)
        plt.close(fig)
    return n


def encode_gif(frames_dir: Path, out: Path, fps: float, width: int, hold_s: float):
    """Two-pass ffmpeg palette encode; holds the last frame so the result is readable."""
    vf = (f"fps={fps},scale={width}:-1:flags=lanczos,tpad=stop_mode=clone:stop_duration={hold_s},"
          "split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];"
          "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps),
                    "-i", str(frames_dir / "%05d.png"), "-vf", vf, "-loop", "0", str(out)], check=True)


SCENES = {
    "figure8": (render_figure8, 6.0),
    "payload_drop": (render_payload_drop, 3.0),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", choices=list(SCENES))
    p.add_argument("--fps", type=float, default=15)
    p.add_argument("--width", type=int, default=860)
    p.add_argument("--dpi", type=int, default=100)
    a = p.parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH")
    OUT.mkdir(exist_ok=True)
    for name, (fn, speed) in SCENES.items():
        if a.only and name != a.only:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            n = fn(Path(tmp), a.fps, speed, a.dpi)
            out = OUT / f"demo_{name}.gif"
            encode_gif(Path(tmp), out, a.fps, a.width, hold_s=2.0)
        print(f"{out}  ({n} frames, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
