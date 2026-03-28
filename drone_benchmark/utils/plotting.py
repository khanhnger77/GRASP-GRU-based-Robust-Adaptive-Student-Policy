"""
Plotting utilities for benchmark results.

All plots are saved to the configured output directory and optionally shown
interactively.
"""

import logging
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from mavlink_interface.telemetry_logger import TelemetrySnapshot

logger = logging.getLogger(__name__)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_trajectory(
    snapshots: List[TelemetrySnapshot],
    title: str = "Trajectory",
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = 150,
) -> None:
    """3D trajectory plot: actual path vs setpoint path."""
    if not snapshots:
        logger.warning("No data to plot.")
        return

    t0 = snapshots[0].timestamp
    times = [s.timestamp - t0 for s in snapshots]
    actual_x = [s.pos_x for s in snapshots]
    actual_y = [s.pos_y for s in snapshots]
    actual_z = [-s.pos_z for s in snapshots]  # NED→up positive for display
    sp_x = [s.sp_x for s in snapshots]
    sp_y = [s.sp_y for s in snapshots]
    sp_z = [-s.sp_z for s in snapshots]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=14)

    # -- 3D trajectory
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.plot(actual_x, actual_y, actual_z, "b-", label="Actual", linewidth=1.5)
    ax3d.plot(sp_x, sp_y, sp_z, "r--", label="Setpoint", linewidth=1.0, alpha=0.7)
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z up (m)")
    ax3d.legend()
    ax3d.set_title("3D Path")

    # -- X over time
    ax_x = fig.add_subplot(2, 2, 2)
    ax_x.plot(times, actual_x, "b-", label="Actual X")
    ax_x.plot(times, sp_x, "r--", label="Setpoint X")
    ax_x.set_xlabel("Time (s)")
    ax_x.set_ylabel("X (m)")
    ax_x.legend()
    ax_x.grid(True)
    ax_x.set_title("X axis")

    # -- Y over time
    ax_y = fig.add_subplot(2, 2, 3)
    ax_y.plot(times, actual_y, "g-", label="Actual Y")
    ax_y.plot(times, sp_y, "r--", label="Setpoint Y")
    ax_y.set_xlabel("Time (s)")
    ax_y.set_ylabel("Y (m)")
    ax_y.legend()
    ax_y.grid(True)
    ax_y.set_title("Y axis")

    # -- Z (altitude) over time
    ax_z = fig.add_subplot(2, 2, 4)
    ax_z.plot(times, actual_z, "m-", label="Actual Z (up)")
    ax_z.plot(times, sp_z, "r--", label="Setpoint Z (up)")
    ax_z.set_xlabel("Time (s)")
    ax_z.set_ylabel("Z (m, up positive)")
    ax_z.legend()
    ax_z.grid(True)
    ax_z.set_title("Altitude")

    plt.tight_layout()

    if save_path:
        _ensure_dir(os.path.dirname(save_path) or ".")
        plt.savefig(save_path, dpi=dpi)
        logger.info("Trajectory plot saved to %s", save_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_error(
    snapshots: List[TelemetrySnapshot],
    title: str = "Position Error",
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = 150,
) -> None:
    """Plot position error magnitude and per-axis over time."""
    if not snapshots:
        return

    t0 = snapshots[0].timestamp
    times = [s.timestamp - t0 for s in snapshots]
    ex = [s.pos_x - s.sp_x for s in snapshots]
    ey = [s.pos_y - s.sp_y for s in snapshots]
    ez = [s.pos_z - s.sp_z for s in snapshots]
    mag = [np.sqrt(x**2 + y**2 + z**2) for x, y, z in zip(ex, ey, ez)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(title)

    axes[0].plot(times, mag, "k-", linewidth=1.5, label="3D error magnitude")
    axes[0].set_ylabel("Error (m)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(times, ex, "b-", label="Ex")
    axes[1].plot(times, ey, "g-", label="Ey")
    axes[1].plot(times, ez, "r-", label="Ez")
    axes[1].axhline(0, color="k", linewidth=0.5)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Error (m)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        _ensure_dir(os.path.dirname(save_path) or ".")
        plt.savefig(save_path, dpi=dpi)
        logger.info("Error plot saved to %s", save_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_comparison(
    snapshots_pid: List[TelemetrySnapshot],
    snapshots_rl: List[TelemetrySnapshot],
    title: str = "PID vs RL Comparison",
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = 150,
) -> None:
    """Side-by-side error comparison between two controllers."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title)

    for ax, snapshots, label, color in [
        (axes[0], snapshots_pid, "PID", "steelblue"),
        (axes[1], snapshots_rl,  "RL",  "tomato"),
    ]:
        if not snapshots:
            ax.set_title(f"{label} — no data")
            continue
        t0 = snapshots[0].timestamp
        times = [s.timestamp - t0 for s in snapshots]
        mag = [
            np.sqrt((s.pos_x - s.sp_x)**2 + (s.pos_y - s.sp_y)**2 + (s.pos_z - s.sp_z)**2)
            for s in snapshots
        ]
        ax.plot(times, mag, color=color, linewidth=1.5)
        ax.set_title(f"{label} — RMSE={np.sqrt(np.mean(np.array(mag)**2)):.3f} m")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("3D Error (m)")
        ax.grid(True)

    plt.tight_layout()
    if save_path:
        _ensure_dir(os.path.dirname(save_path) or ".")
        plt.savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
