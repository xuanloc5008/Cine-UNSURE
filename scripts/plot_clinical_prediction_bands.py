#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_band(ax, times, bands, title: str, ylabel: str) -> None:
    mean = np.asarray([row["mean"] for row in bands], dtype=float)
    lower = np.asarray([row["lower"] for row in bands], dtype=float)
    upper = np.asarray([row["upper"] for row in bands], dtype=float)
    ax.plot(times, mean, color="#1769aa", linewidth=2, label="Mean")
    ax.fill_between(times, lower, upper, color="#5aa9e6", alpha=0.32, label="Model uncertainty band")
    ax.set_title(title)
    ax.set_xlabel("Normalized cardiac time")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    row = json.loads(Path(args.input).read_text(encoding="utf-8"))
    bands = row["prediction_bands"]
    times = np.asarray(row["times"], dtype=float)
    if times.max() > times.min():
        times = (times - times.min()) / (times.max() - times.min())

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    plot_band(axes[0, 0], times, bands["volume_curve"], "LV volume", "Volume (ml)")
    plot_band(axes[0, 1], times, bands["wall_motion"], "Mean wall motion", "Motion (mm)")
    strain_keys = ("strain_xx", "strain_yy", "strain_zz")
    for ax, key in zip((axes[1, 0], axes[1, 1], axes[1, 2]), strain_keys, strict=True):
        plot_band(ax, times, [frame[key] for frame in bands["strain"]], key, "Strain")

    ef = bands["ef"]
    axes[0, 2].errorbar(
        [0],
        [ef["mean"]],
        yerr=[[ef["mean"] - ef["lower"]], [ef["upper"] - ef["mean"]]],
        fmt="o",
        color="#1769aa",
        capsize=7,
    )
    axes[0, 2].set_xlim(-1, 1)
    axes[0, 2].set_xticks([])
    axes[0, 2].set_title("Ejection fraction over the cycle")
    axes[0, 2].set_ylabel("EF")
    axes[0, 2].grid(axis="y", alpha=0.25)
    axes[0, 0].legend(loc="best")
    fig.suptitle(
        f"Clinical trajectories with {100 * float(bands['coverage']):.0f}% model-derived uncertainty bands",
        fontsize=14,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(json.dumps({"output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
