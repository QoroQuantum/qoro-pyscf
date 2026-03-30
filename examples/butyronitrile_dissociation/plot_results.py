#!/usr/bin/env python3
# Copyright 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Plot Butyronitrile C≡N Dissociation Results
=============================================

Reads JSON results produced by ``run_cn_dissociation.py`` and generates
publication-quality plots:

  1. **Dissociation PES** — Energy vs C≡N distance for HF, MP2, FCI, VQE
  2. **Accuracy plot** — VQE error vs FCI at each geometry (mHa)
  3. **Timing plot** — Wall time per geometry point (+ GPU speedup)
  4. **Cross-stage overlay** — Compare Stage 1 and Stage 2 on one figure

Usage
-----
  # Plot a single result file
  python plot_results.py results/butyronitrile_stage1_sto-3g_cas8.json

  # Compare two stages
  python plot_results.py results/butyronitrile_stage1_*.json results/butyronitrile_stage2_*.json

  # Save to PNG instead of displaying
  python plot_results.py results/*.json --save

  # Specify output directory
  python plot_results.py results/*.json --save --outdir figures/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator


# ─────────────────────────────────────────────
# Colour palette (dark theme)
# ─────────────────────────────────────────────
DARK_BG = "#0f1117"
PANEL_BG = "#0f1117"
GRID_COLOR = "#1e2130"
SPINE_COLOR = "#333"
TEXT_COLOR = "white"

COLORS = {
    "hf":       "#6b7280",  # grey
    "mp2":      "#f59e0b",  # amber
    "fci":      "#10b981",  # emerald
    "vqe_cpu":  "#60a5fa",  # blue
    "vqe_gpu":  "#a78bfa",  # violet
    "error":    "#ef4444",  # red
    "chem_acc": "#22d3ee",  # cyan
}

LABELS = {
    "hf":      "HF",
    "mp2":     "MP2",
    "fci":     "FCI (CAS)",
    "vqe_cpu": "VQE (CPU)",
    "vqe_gpu": "VQE (GPU)",
}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def load_results(path: str) -> dict:
    """Load a JSON results file."""
    with open(path) as f:
        return json.load(f)


def _style_axis(ax):
    """Apply the dark-theme styling to an axis."""
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COLOR, which="both")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COLOR)
    ax.yaxis.grid(True, color=GRID_COLOR, zorder=0, alpha=0.5)
    ax.xaxis.grid(True, color=GRID_COLOR, zorder=0, alpha=0.3)


def _safe_array(values):
    """Convert a list with possible None entries to a masked array."""
    arr = np.array([v if v is not None else np.nan for v in values], dtype=float)
    return arr


# ─────────────────────────────────────────────
# Plot 1: Dissociation PES
# ─────────────────────────────────────────────
def plot_pes(data: dict, ax=None, stage_label: str = ""):
    """
    Plot the potential energy surface: energy vs C≡N distance.

    Shows HF, MP2, FCI(CAS), and VQE curves on the same axes.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(DARK_BG)

    _style_axis(ax)

    d = np.array(data["cn_distances"])
    meta = data["metadata"]

    # Reference curves
    hf = _safe_array(data["hf_energies"])
    mp2 = _safe_array(data["mp2_energies"])
    fci = _safe_array(data["fci_energies"])

    ax.plot(d, hf, "o--", color=COLORS["hf"], ms=5, lw=1.5,
            label=LABELS["hf"], zorder=2, alpha=0.7)
    if not np.all(np.isnan(mp2)):
        ax.plot(d, mp2, "s--", color=COLORS["mp2"], ms=5, lw=1.5,
                label=LABELS["mp2"], zorder=2, alpha=0.7)
    if not np.all(np.isnan(fci)):
        ax.plot(d, fci, "^-", color=COLORS["fci"], ms=6, lw=2,
                label=LABELS["fci"], zorder=3)

    # VQE curves
    for backend, color_key in [("cpu", "vqe_cpu"), ("gpu", "vqe_gpu")]:
        key = f"vqe_{backend}_energies"
        if key in data:
            vqe = _safe_array(data[key])
            if not np.all(np.isnan(vqe)):
                ax.plot(d, vqe, "D-", color=COLORS[color_key], ms=6, lw=2.2,
                        label=LABELS[color_key], zorder=4,
                        markeredgecolor="white", markeredgewidth=0.5)

    ax.set_xlabel("C≡N Distance (Å)", color=TEXT_COLOR, fontsize=12)
    ax.set_ylabel("Energy (Ha)", color=TEXT_COLOR, fontsize=12)

    title = "Butyronitrile C≡N Dissociation"
    if stage_label:
        title += f" — {stage_label}"
    title += f"\nCAS({meta['nelec']},{meta['norb']})/{meta['basis']} | {meta['n_qubits']} qubits"
    ax.set_title(title, color=TEXT_COLOR, fontsize=13, pad=12)

    ax.legend(
        facecolor="#1e2130", edgecolor="#444", labelcolor=TEXT_COLOR,
        fontsize=10, loc="upper right",
    )

    if standalone:
        plt.tight_layout()
    return ax


# ─────────────────────────────────────────────
# Plot 2: Accuracy (error vs FCI)
# ─────────────────────────────────────────────
def plot_accuracy(data: dict, ax=None, stage_label: str = ""):
    """
    Bar chart of VQE error relative to FCI(CAS) at each geometry.

    Shows a dashed line at 1.6 mHa (chemical accuracy threshold).
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(DARK_BG)

    _style_axis(ax)

    d = np.array(data["cn_distances"])
    fci = _safe_array(data["fci_energies"])
    meta = data["metadata"]

    bar_width = 0.06
    backends_plotted = 0

    for backend, color_key in [("cpu", "vqe_cpu"), ("gpu", "vqe_gpu")]:
        key = f"vqe_{backend}_energies"
        if key not in data:
            continue
        vqe = _safe_array(data[key])

        # Compute errors (mHa) only where both VQE and FCI are available
        mask = ~np.isnan(vqe) & ~np.isnan(fci)
        if not np.any(mask):
            continue

        errors_mha = np.abs(vqe[mask] - fci[mask]) * 1000
        positions = d[mask] + (backends_plotted - 0.5) * bar_width

        bars = ax.bar(
            positions, errors_mha, width=bar_width * 0.9,
            color=COLORS[color_key], edgecolor="white", linewidth=0.5,
            label=LABELS[color_key], zorder=3, alpha=0.85,
        )
        # Value labels on bars
        for bar, err in zip(bars, errors_mha):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{err:.1f}", ha="center", va="bottom",
                color=TEXT_COLOR, fontsize=8, fontweight="bold",
            )
        backends_plotted += 1

    # Chemical accuracy line
    ax.axhline(1.6, color=COLORS["chem_acc"], linestyle="--", lw=2,
               alpha=0.8, label="Chemical accuracy (1.6 mHa)", zorder=2)

    ax.set_xlabel("C≡N Distance (Å)", color=TEXT_COLOR, fontsize=12)
    ax.set_ylabel("Error vs FCI (mHa)", color=TEXT_COLOR, fontsize=12)

    title = "VQE Accuracy"
    if stage_label:
        title += f" — {stage_label}"
    ax.set_title(title, color=TEXT_COLOR, fontsize=13, pad=12)

    ax.set_xticks(d)
    ax.set_xticklabels([f"{x:.2f}" for x in d], fontsize=9)
    ax.legend(
        facecolor="#1e2130", edgecolor="#444", labelcolor=TEXT_COLOR,
        fontsize=10,
    )

    if standalone:
        plt.tight_layout()
    return ax


# ─────────────────────────────────────────────
# Plot 3: Timing
# ─────────────────────────────────────────────
def plot_timing(data: dict, ax=None, stage_label: str = ""):
    """
    Bar chart of wall time per geometry point.

    If both CPU and GPU data are present, shows a speedup annotation.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(DARK_BG)

    _style_axis(ax)

    d = np.array(data["cn_distances"])
    meta = data["metadata"]
    bar_width = 0.06

    all_times = {}
    backends_plotted = 0

    for backend, color_key in [("cpu", "vqe_cpu"), ("gpu", "vqe_gpu")]:
        key = f"vqe_{backend}_times"
        if key not in data:
            continue
        times = _safe_array(data[key])
        mask = ~np.isnan(times)
        if not np.any(mask):
            continue

        all_times[backend] = times
        positions = d[mask] + (backends_plotted - 0.5) * bar_width

        ax.bar(
            positions, times[mask], width=bar_width * 0.9,
            color=COLORS[color_key], edgecolor="white", linewidth=0.5,
            label=LABELS[color_key], zorder=3, alpha=0.85,
        )
        backends_plotted += 1

    # Speedup annotation if both backends present
    if "cpu" in all_times and "gpu" in all_times:
        cpu_t = all_times["cpu"]
        gpu_t = all_times["gpu"]
        valid = ~np.isnan(cpu_t) & ~np.isnan(gpu_t) & (gpu_t > 0)
        if np.any(valid):
            avg_speedup = np.mean(cpu_t[valid] / gpu_t[valid])
            ax.text(
                0.98, 0.95,
                f"Avg GPU speedup: {avg_speedup:.1f}×",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=12, fontweight="bold", color=COLORS["vqe_gpu"],
                bbox=dict(facecolor="#1e2130", edgecolor=COLORS["vqe_gpu"],
                          boxstyle="round,pad=0.4", alpha=0.9),
            )

    ax.set_xlabel("C≡N Distance (Å)", color=TEXT_COLOR, fontsize=12)
    ax.set_ylabel("Wall Time (s)", color=TEXT_COLOR, fontsize=12)

    title = "VQE Wall Time per Geometry"
    if stage_label:
        title += f" — {stage_label}"
    ax.set_title(title, color=TEXT_COLOR, fontsize=13, pad=12)

    ax.set_xticks(d)
    ax.set_xticklabels([f"{x:.2f}" for x in d], fontsize=9)
    ax.legend(
        facecolor="#1e2130", edgecolor="#444", labelcolor=TEXT_COLOR,
        fontsize=10,
    )

    if standalone:
        plt.tight_layout()
    return ax


# ─────────────────────────────────────────────
# Combined dashboard for one stage
# ─────────────────────────────────────────────
def plot_dashboard(data: dict, save_path: str | None = None):
    """
    Three-panel dashboard: PES + accuracy + timing.
    """
    meta = data["metadata"]
    stage_label = meta["label"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor(DARK_BG)

    plot_pes(data, ax=axes[0], stage_label=stage_label)
    plot_accuracy(data, ax=axes[1], stage_label=stage_label)
    plot_timing(data, ax=axes[2], stage_label=stage_label)

    fig.suptitle(
        f"Butyronitrile C≡N Dissociation — {stage_label}",
        color=TEXT_COLOR, fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        print(f"  Saved: {save_path}")
    else:
        plt.show()


# ─────────────────────────────────────────────
# Cross-stage comparison
# ─────────────────────────────────────────────
def plot_cross_stage(datasets: list[dict], save_path: str | None = None):
    """
    Compare multiple stages on the same figure.

    Top row:   PES for each stage side by side
    Bottom:    Accuracy comparison overlay
    """
    n_stages = len(datasets)

    fig = plt.figure(figsize=(8 * n_stages, 10))
    fig.patch.set_facecolor(DARK_BG)

    gs = fig.add_gridspec(2, n_stages, hspace=0.35, wspace=0.3)

    # Top row: PES per stage
    for i, data in enumerate(datasets):
        ax = fig.add_subplot(gs[0, i])
        plot_pes(data, ax=ax, stage_label=data["metadata"]["label"])

    # Bottom row: accuracy overlay
    ax_acc = fig.add_subplot(gs[1, :])
    _style_axis(ax_acc)

    stage_colors = ["#60a5fa", "#a78bfa", "#34d399", "#f59e0b"]

    for i, data in enumerate(datasets):
        meta = data["metadata"]
        d = np.array(data["cn_distances"])
        fci = _safe_array(data["fci_energies"])
        color = stage_colors[i % len(stage_colors)]

        # Use the first available VQE backend
        for backend in ("gpu", "cpu"):
            key = f"vqe_{backend}_energies"
            if key in data:
                vqe = _safe_array(data[key])
                mask = ~np.isnan(vqe) & ~np.isnan(fci)
                if np.any(mask):
                    errors = np.abs(vqe[mask] - fci[mask]) * 1000
                    ax_acc.plot(
                        d[mask], errors, "D-", color=color, ms=7, lw=2,
                        label=f"{meta['label']} ({backend.upper()})",
                        markeredgecolor="white", markeredgewidth=0.5,
                        zorder=3,
                    )
                break

    ax_acc.axhline(1.6, color=COLORS["chem_acc"], linestyle="--", lw=2,
                   alpha=0.8, label="Chemical accuracy (1.6 mHa)", zorder=2)

    ax_acc.set_xlabel("C≡N Distance (Å)", color=TEXT_COLOR, fontsize=12)
    ax_acc.set_ylabel("Error vs FCI (mHa)", color=TEXT_COLOR, fontsize=12)
    ax_acc.set_title("Cross-Stage Accuracy Comparison", color=TEXT_COLOR,
                     fontsize=13, pad=12)
    ax_acc.legend(
        facecolor="#1e2130", edgecolor="#444", labelcolor=TEXT_COLOR,
        fontsize=10,
    )

    fig.suptitle(
        "Butyronitrile C≡N Dissociation — Stage Comparison",
        color=TEXT_COLOR, fontsize=16, fontweight="bold", y=1.01,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        print(f"  Saved: {save_path}")
    else:
        plt.show()


# ─────────────────────────────────────────────
# Print summary table
# ─────────────────────────────────────────────
def print_summary(data: dict):
    """Print a text summary table of the results."""
    meta = data["metadata"]
    d = data["cn_distances"]
    fci = _safe_array(data["fci_energies"])

    print()
    print("=" * 78)
    print(f"  {meta['label']}")
    print(f"  {meta.get('description', '')}")
    print(f"  Ansatz: {meta['ansatz']}  |  Optimizer: {meta['optimizer']}")
    print(f"  Qubits: {meta['n_qubits']}  |  Basis: {meta['basis']}")
    print("=" * 78)

    header = f"  {'d(C≡N)':>8s}  {'HF':>14s}  {'MP2':>14s}  {'FCI':>14s}"
    backends = []
    for backend in ("cpu", "gpu"):
        if f"vqe_{backend}_energies" in data:
            backends.append(backend)
            header += f"  {'VQE('+backend.upper()+')':>14s}  {'Err(mHa)':>9s}  {'Time':>7s}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for i, r in enumerate(d):
        hf = data["hf_energies"][i]
        mp2_v = data["mp2_energies"][i]
        fci_v = data["fci_energies"][i]

        line = f"  {r:8.3f}  {hf:+14.8f}"
        line += f"  {mp2_v:+14.8f}" if mp2_v is not None else "           N/A  "
        line += f"  {fci_v:+14.8f}" if fci_v is not None else "           N/A  "

        for backend in backends:
            vqe_v = data[f"vqe_{backend}_energies"][i]
            time_v = data[f"vqe_{backend}_times"][i]
            if vqe_v is not None:
                err = abs(vqe_v - fci_v) * 1000 if fci_v is not None else float("nan")
                err_str = f"{err:9.2f}" if not np.isnan(err) else "      N/A"
                time_str = f"{time_v:7.1f}" if time_v is not None else "    N/A"
                line += f"  {vqe_v:+14.8f}  {err_str}  {time_str}"
            else:
                line += "            N/A        N/A      N/A"

        print(line)

    # Overall stats
    for backend in backends:
        vqe = _safe_array(data[f"vqe_{backend}_energies"])
        mask = ~np.isnan(vqe) & ~np.isnan(fci)
        if np.any(mask):
            errors = np.abs(vqe[mask] - fci[mask]) * 1000
            times = _safe_array(data[f"vqe_{backend}_times"])
            valid_times = times[~np.isnan(times)]
            chem = "✓" if np.max(errors) < 1.6 else "✗"
            print(f"\n  {backend.upper()}: max err = {np.max(errors):.2f} mHa, "
                  f"mean err = {np.mean(errors):.2f} mHa  "
                  f"| chem. accuracy: {chem}")
            if len(valid_times) > 0:
                print(f"       total time = {np.sum(valid_times):.1f}s, "
                      f"avg = {np.mean(valid_times):.1f}s/point")
    print("=" * 78)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Plot butyronitrile C≡N dissociation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python plot_results.py results/butyronitrile_stage1_*.json
  python plot_results.py results/*.json --save
  python plot_results.py results/*.json --save --outdir figures/
""",
    )
    parser.add_argument("files", nargs="+", help="JSON result file(s)")
    parser.add_argument("--save", action="store_true",
                        help="Save plots as PNG instead of displaying")
    parser.add_argument("--outdir", type=str, default="figures",
                        help="Output directory for saved plots (default: figures/)")
    parser.add_argument("--no-summary", action="store_true",
                        help="Skip printing text summary table")

    args = parser.parse_args()

    # Load all result files
    datasets = []
    for f in args.files:
        try:
            datasets.append(load_results(f))
            print(f"  Loaded: {f}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"  ⚠️  Skipping {f}: {e}", file=sys.stderr)

    if not datasets:
        print("No valid result files found.", file=sys.stderr)
        sys.exit(1)

    # Setup output directory
    outdir = Path(args.outdir)
    if args.save:
        outdir.mkdir(parents=True, exist_ok=True)

    # Print summaries
    if not args.no_summary:
        for data in datasets:
            print_summary(data)

    # Plot each dataset individually
    for data in datasets:
        meta = data["metadata"]
        stage = meta["stage"]
        basis = meta["basis"].replace("*", "star")
        filename = f"butyronitrile_stage{stage}_{basis}_cas{meta['norb']}"

        save_path = str(outdir / f"{filename}_dashboard.png") if args.save else None
        plot_dashboard(data, save_path=save_path)

    # Cross-stage comparison if multiple files
    if len(datasets) > 1:
        save_path = str(outdir / "butyronitrile_cross_stage.png") if args.save else None
        plot_cross_stage(datasets, save_path=save_path)

    if args.save:
        print(f"\n  All plots saved to {outdir}/")


if __name__ == "__main__":
    main()
