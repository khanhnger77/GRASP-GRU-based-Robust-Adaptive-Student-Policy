#!/usr/bin/env python3
"""
Benchmark plot: Return & Feasibility across init_deg experiments
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# Experiment configuration
EXPERIMENTS = [
    {"init_deg": 0,  "exp_id": "2026-03-29_03-42-07"},
    {"init_deg": 60, "exp_id": "2026-03-29_03-43-25"},
]

BASE_DIR = Path(__file__).parent.parent / "experiments"
COMMIT = "49cd544_payload-policy-pre-training_dynamics-id"

COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]


def load_experiment(exp_id):
    path = BASE_DIR / exp_id / COMMIT / "0" / "0000" / "return.json"
    with open(path) as f:
        data = json.load(f)
    steps = [d["step"] for d in data]
    returns_mean = [d["returns_mean"] for d in data]
    returns_std = [d["returns_std"] for d in data]
    episode_len_mean = [d["episode_length_mean"] for d in data]
    episode_len_std = [d["episode_length_std"] for d in data]
    # share_terminated = fraction of episodes that crashed/failed
    # feasibility = 1 - share_terminated
    feasibility = [1.0 - d["share_terminated"] for d in data]
    return {
        "steps": np.array(steps),
        "returns_mean": np.array(returns_mean),
        "returns_std": np.array(returns_std),
        "episode_len_mean": np.array(episode_len_mean),
        "episode_len_std": np.array(episode_len_std),
        "feasibility": np.array(feasibility),
    }


def smooth(values, window=5):
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


# ── Load all data ────────────────────────────────────────────────────────────
all_data = {}
for exp in EXPERIMENTS:
    all_data[exp["init_deg"]] = load_experiment(exp["exp_id"])

# ── Figure layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.suptitle(
    "Benchmark: Pre-training across Initial Tilt Angles (init_deg)\n"
    "Payload Policy — Comparison of Return & Feasibility",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
ax_ret   = fig.add_subplot(gs[0, :])   # Return over training (full width)
ax_feas  = fig.add_subplot(gs[1, 0])   # Feasibility over training
ax_final = fig.add_subplot(gs[1, 1])   # Final-step bar comparison

# ── Plot 1: Return over training steps ──────────────────────────────────────
for (deg, data), color in zip(all_data.items(), COLORS):
    steps_m = data["steps"] / 1e6
    ret_s   = smooth(data["returns_mean"], window=5)
    std_s   = smooth(data["returns_std"],  window=5)

    ax_ret.plot(steps_m, ret_s, color=color, linewidth=2,
                label=f"init_deg={deg}°")
    ax_ret.fill_between(steps_m, ret_s - std_s, ret_s + std_s,
                         alpha=0.15, color=color)

ax_ret.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax_ret.set_xlabel("Training Steps (×10⁶)", fontsize=11)
ax_ret.set_ylabel("Episode Return", fontsize=11)
ax_ret.set_title("Training Return vs Steps  (shaded = ±1 std)", fontsize=12)
ax_ret.legend(fontsize=10, loc="lower right")
ax_ret.grid(True, alpha=0.3)

# ── Plot 2: Feasibility over training steps ──────────────────────────────────
for (deg, data), color in zip(all_data.items(), COLORS):
    steps_m  = data["steps"] / 1e6
    feas_s   = smooth(data["feasibility"], window=5)
    ax_feas.plot(steps_m, feas_s * 100, color=color, linewidth=2,
                 label=f"init_deg={deg}°")

ax_feas.axhline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax_feas.set_xlabel("Training Steps (×10⁶)", fontsize=11)
ax_feas.set_ylabel("Feasibility  (%)", fontsize=11)
ax_feas.set_title("Feasibility = 1 − share_terminated", fontsize=12)
ax_feas.set_ylim(-5, 105)
ax_feas.legend(fontsize=10)
ax_feas.grid(True, alpha=0.3)

# ── Plot 3: Final-step bar chart ─────────────────────────────────────────────
degs         = [exp["init_deg"] for exp in EXPERIMENTS]
final_ret    = [all_data[d]["returns_mean"][-1]  for d in degs]
final_feas   = [all_data[d]["feasibility"][-1] * 100 for d in degs]

x      = np.arange(len(degs))
width  = 0.35

bars1 = ax_final.bar(x - width / 2, final_ret,  width, color=COLORS,
                     alpha=0.85, label="Final Return")
ax_final2 = ax_final.twinx()
bars2 = ax_final2.bar(x + width / 2, final_feas, width, color=COLORS,
                      alpha=0.45, hatch="//", label="Feasibility (%)")

# Annotate values
for bar, val in zip(bars1, final_ret):
    ax_final.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                  f"{val:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
for bar, val in zip(bars2, final_feas):
    ax_final2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                   f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

ax_final.set_xlabel("Initial Tilt Angle (deg)", fontsize=11)
ax_final.set_ylabel("Final Episode Return", fontsize=11)
ax_final2.set_ylabel("Final Feasibility (%)", fontsize=11)
ax_final.set_title("Final Checkpoint Comparison", fontsize=12)
ax_final.set_xticks(x)
ax_final.set_xticklabels([f"{d}°" for d in degs])
ax_final.grid(True, alpha=0.3, axis="y")

# Combined legend
lines1, labels1 = ax_final.get_legend_handles_labels()
lines2, labels2 = ax_final2.get_legend_handles_labels()
ax_final.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="lower right")

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = Path(__file__).parent.parent / "benchmark_init_deg.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
