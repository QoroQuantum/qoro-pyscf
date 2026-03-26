#!/usr/bin/env python3
# Copyright 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Publication-quality plots for the N₂ dissociation blog post.

Reads results from results/*.json (produced by run_pes_scan.py) and generates:

  1. pes_comparison.png   — PES curves: HF, CCSD(T), VQE, FCI (the hero plot)
  2. timing_comparison.png — CPU vs GPU wall-time per geometry
  3. error_analysis.png    — VQE error vs FCI across the PES
  4. speedup_bar.png       — Speedup factor at each geometry

Usage
-----
    python plot_results.py                          # auto-detect latest results
    python plot_results.py results/n2_cc-pvdz_cas10.json
"""

import json
import sys
from pathlib import Path

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# Style configuration (publication quality)
# ═══════════════════════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Publication-quality defaults
rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2,
    "lines.markersize": 7,
})

# Colour palette
COLORS = {
    "hf":      "#bdbdbd",     # grey
    "ccsd_t":  "#e74c3c",     # red
    "fci":     "#2c3e50",     # dark navy
    "vqe_gpu": "#2ecc71",     # green
    "vqe_cpu": "#3498db",     # blue
    "error":   "#e67e22",     # orange
    "speedup": "#9b59b6",     # purple
    "qoro":    "#00c9a7",     # Qoro teal
}


def load_results(path: str | None = None) -> dict:
    """Load results JSON, auto-detecting the latest if no path given."""
    if path:
        with open(path) as f:
            return json.load(f)

    results_dir = Path(__file__).parent / "results"
    jsons = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not jsons:
        raise FileNotFoundError("No results found in results/. Run run_pes_scan.py first.")
    print(f"  Loading: {jsons[-1]}")
    with open(jsons[-1]) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1: Potential Energy Surface comparison (the "hero" plot)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pes(data: dict, out_dir: Path):
    """
    The main plot: HF, CCSD(T), VQE-GPU, and FCI potential energy curves.
    This is the plot that tells the story.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))

    r = data["distances"]
    meta = data["metadata"]

    # HF — the single-reference baseline
    ax.plot(r, data["hf_energies"], ":", color=COLORS["hf"],
            label="RHF", linewidth=1.5)

    # CCSD(T) — breaks at stretched geometries
    ccsd_t = data.get("ccsd_t_energies", [])
    if ccsd_t:
        r_cc = [r[i] for i, e in enumerate(ccsd_t) if e is not None]
        e_cc = [e for e in ccsd_t if e is not None]
        ax.plot(r_cc, e_cc, "s--", color=COLORS["ccsd_t"],
                label="CCSD(T)", markersize=5, linewidth=1.5)

        # Mark where CCSD(T) fails
        for i, e in enumerate(ccsd_t):
            if e is None:
                ax.axvline(r[i], color=COLORS["ccsd_t"], alpha=0.15, linewidth=8)

    # FCI reference (exact within active space)
    ax.plot(r, data["fci_energies"], "k-", color=COLORS["fci"],
            label=f'FCI — CAS({meta["nelec"]},{meta["norb"]})',
            linewidth=2.5)

    # VQE on GPU
    vqe_gpu = data.get("vqe_gpu_energies", [])
    if vqe_gpu:
        r_gpu = [r[i] for i, e in enumerate(vqe_gpu) if e is not None]
        e_gpu = [e for e in vqe_gpu if e is not None]
        ax.plot(r_gpu, e_gpu, "o-", color=COLORS["vqe_gpu"],
                label="UCCSD-VQE (Qoro GPU)", markersize=6,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5)

    # VQE on CPU
    vqe_cpu = data.get("vqe_cpu_energies", [])
    if vqe_cpu:
        r_cpu = [r[i] for i, e in enumerate(vqe_cpu) if e is not None]
        e_cpu = [e for e in vqe_cpu if e is not None]
        ax.plot(r_cpu, e_cpu, "^--", color=COLORS["vqe_cpu"],
                label="UCCSD-VQE (CPU)", markersize=5, alpha=0.7)

    ax.set_xlabel("N–N bond length (Å)")
    ax.set_ylabel("Total energy (Hartree)")
    ax.set_title("N₂ Dissociation: Where Single-Reference Methods Fail")
    ax.legend(loc="upper right", framealpha=0.9)

    # Annotation: mark the failure region
    if ccsd_t:
        fail_start = min((r[i] for i, e in enumerate(ccsd_t) if e is None), default=None)
        if fail_start:
            ax.annotate(
                "CCSD(T) fails\n(multireference\nregion)",
                xy=(fail_start, ax.get_ylim()[0] + 0.3 * (ax.get_ylim()[1] - ax.get_ylim()[0])),
                fontsize=10, color=COLORS["ccsd_t"], ha="center",
                fontstyle="italic",
            )

    # Equilibrium marker
    r_eq_idx = np.argmin(data["fci_energies"])
    ax.annotate(
        f"  r_eq = {r[r_eq_idx]:.2f} Å",
        xy=(r[r_eq_idx], data["fci_energies"][r_eq_idx]),
        xytext=(r[r_eq_idx] + 0.3, data["fci_energies"][r_eq_idx] - 0.05),
        arrowprops=dict(arrowstyle="->", color=COLORS["fci"]),
        fontsize=10, color=COLORS["fci"],
    )

    fig.tight_layout()
    path = out_dir / "pes_comparison.png"
    fig.savefig(path)
    print(f"  Saved: {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2: GPU vs CPU wall-time
# ═══════════════════════════════════════════════════════════════════════════════

def plot_timing(data: dict, out_dir: Path):
    """CPU vs GPU wall-time per geometry point."""
    cpu_times = data.get("vqe_cpu_times", [])
    gpu_times = data.get("vqe_gpu_times", [])
    if not cpu_times or not gpu_times:
        print("  ⚠ Skipping timing plot (need --both data)")
        return

    r = data["distances"]
    meta = data["metadata"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]})

    # Left: timing curves
    valid = [(r[i], c, g) for i, (c, g) in enumerate(zip(cpu_times, gpu_times))
             if c is not None and g is not None]
    if not valid:
        plt.close(fig)
        return

    rv, ct, gt = zip(*valid)

    ax1.fill_between(rv, ct, gt, alpha=0.15, color=COLORS["speedup"],
                     label="Time saved")
    ax1.plot(rv, ct, "s-", color=COLORS["vqe_cpu"], label="CPU", markersize=6)
    ax1.plot(rv, gt, "o-", color=COLORS["vqe_gpu"], label="GPU (Qoro)",
             markersize=6, markeredgecolor="white", markeredgewidth=0.8)
    ax1.set_xlabel("N–N bond length (Å)")
    ax1.set_ylabel("Wall time (seconds)")
    ax1.set_title(f'VQE Wall Time — {meta["n_qubits"]} Qubits')
    ax1.legend(loc="upper left")

    # Right: speedup bar chart
    speedups = [c / g for c, g in zip(ct, gt)]
    ax2.barh(range(len(rv)), speedups, color=COLORS["speedup"], alpha=0.8,
             edgecolor="white")
    ax2.set_yticks(range(len(rv)))
    ax2.set_yticklabels([f"{x:.1f} Å" for x in rv])
    ax2.set_xlabel("Speedup (CPU / GPU)")
    ax2.set_title("GPU Speedup Factor")
    ax2.axvline(1.0, color="grey", linestyle="--", alpha=0.5)

    # Average speedup annotation
    avg_speedup = np.mean(speedups)
    ax2.annotate(
        f"Average: {avg_speedup:.1f}×",
        xy=(avg_speedup, len(rv) - 1),
        fontsize=12, fontweight="bold", color=COLORS["speedup"],
        ha="center", va="bottom",
    )

    fig.suptitle("Why GPU Matters for Quantum Chemistry", fontsize=15, y=1.02)
    fig.tight_layout()
    path = out_dir / "timing_comparison.png"
    fig.savefig(path)
    print(f"  Saved: {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3: VQE error analysis
# ═══════════════════════════════════════════════════════════════════════════════

def plot_error(data: dict, out_dir: Path):
    """VQE error compared to exact FCI across the PES."""
    gpu = data.get("vqe_gpu_energies", data.get("vqe_cpu_energies", []))
    if not gpu:
        print("  ⚠ Skipping error plot (no VQE data)")
        return

    r = data["distances"]
    fci = data["fci_energies"]

    valid = [(r[i], abs(g - fci[i]) * 1000) for i, g in enumerate(gpu) if g is not None]
    if not valid:
        return

    rv, errors = zip(*valid)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(rv, errors, width=0.08, color=COLORS["qoro"], alpha=0.8,
           edgecolor="white", label="|E_VQE − E_FCI|")
    ax.axhline(1.6, color=COLORS["ccsd_t"], linestyle="--", linewidth=1.5,
               label="Chemical accuracy (1.6 mHa)")

    ax.set_xlabel("N–N bond length (Å)")
    ax.set_ylabel("Error (mHa)")
    ax.set_title("VQE Accuracy Across the Dissociation Curve")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    path = out_dir / "error_analysis.png"
    fig.savefig(path)
    print(f"  Saved: {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 4: Combined summary figure (for blog header)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_summary(data: dict, out_dir: Path):
    """Single-panel summary suitable for a blog header image."""
    r = data["distances"]
    meta = data["metadata"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # FCI
    ax.plot(r, data["fci_energies"], "-", color=COLORS["fci"],
            linewidth=3, label="Exact (FCI)", zorder=3)

    # VQE GPU
    gpu = data.get("vqe_gpu_energies", [])
    if gpu:
        r_gpu = [r[i] for i, e in enumerate(gpu) if e is not None]
        e_gpu = [e for e in gpu if e is not None]
        ax.plot(r_gpu, e_gpu, "o", color=COLORS["vqe_gpu"], markersize=10,
                markeredgecolor="white", markeredgewidth=1.5,
                label="VQE (Qoro GPU)", zorder=5)

    # CCSD(T)
    ccsd_t = data.get("ccsd_t_energies", [])
    if ccsd_t:
        r_cc = [r[i] for i, e in enumerate(ccsd_t) if e is not None]
        e_cc = [e for e in ccsd_t if e is not None]
        ax.plot(r_cc, e_cc, "x-", color=COLORS["ccsd_t"],
                label="CCSD(T)", markersize=8, linewidth=1.5, alpha=0.7)

    ax.set_xlabel("N–N distance (Å)", fontsize=15)
    ax.set_ylabel("Energy (Hartree)", fontsize=15)
    ax.legend(fontsize=13, loc="upper right", framealpha=0.9)

    # Title with Qoro branding
    ax.set_title(
        f"N₂ Dissociation — UCCSD-VQE on {meta['n_qubits']} Qubits\n"
        f"Powered by Qoro GPU",
        fontsize=16, fontweight="bold",
    )

    fig.tight_layout()
    path = out_dir / "summary_hero.png"
    fig.savefig(path)
    print(f"  Saved: {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    data = load_results(path)

    out_dir = Path(__file__).parent / "figures"
    out_dir.mkdir(exist_ok=True)

    print("\n  Generating publication-quality figures...")
    plot_pes(data, out_dir)
    plot_timing(data, out_dir)
    plot_error(data, out_dir)
    plot_summary(data, out_dir)
    print(f"\n  All figures saved to: {out_dir}/")


if __name__ == "__main__":
    main()
