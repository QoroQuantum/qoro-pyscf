"""
Predicting the UV-Vis Absorption of Ethylene with Maestro
==========================================================
Ethylene (C₂H₄) absorbs UV light at ~162 nm via a π→π* transition.
This script calculates S₀ (VQE) and S₁ (VQD) energies, converts the
excitation energy to a wavelength, and plots the predicted spectrum.

Three progressively larger active spaces:
  Stage 1 – CPU,          (4e, 4o) →  8 qubits  — fast prototype
  Stage 2 – GPU exact,   (6e, 6o) → 12 qubits  — precision + speed
  Stage 3 – GPU + MPS,  (12e,12o) → 24 qubits  — Maestro hero feature

Usage — run all stages:
  python 14_ethylene_uv_vis.py

Usage — run a single stage interactively:
  from examples.14_ethylene_uv_vis import setup_molecule, run_stage1
  hf = setup_molecule()
  result = run_stage1(hf)
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
from pyscf import gto, mcscf, scf

from qoro_pyscf import (
    QoroSolver,
    VQDSolver,
    suggest_active_space_from_mp2,
)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
HARTREE_TO_EV = 27.211386245988   # 1 Ha in eV
EV_TO_NM      = 1239.84193        # hc in eV·nm
LAMBDA_EXP    = 162.0             # nm, experimental π→π* maximum


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def excitation_to_nm(delta_e_hartree: float) -> float:
    """Convert a vertical excitation energy (Hartree) to wavelength (nm)."""
    if delta_e_hartree <= 0:
        return float("inf")
    return EV_TO_NM / (delta_e_hartree * HARTREE_TO_EV)


def check_vqd_converged(E0: float, E1: float, tol: float = 1e-4) -> bool:
    """Warn if the VQD S₁ collapsed back to S₀."""
    if abs(E1 - E0) < tol:
        print(
            f"\n⚠️  WARNING: |E(S₁) - E(S₀)| = {abs(E1-E0):.2e} Ha — VQD may have collapsed.\n"
            "   Try increasing penalty_weights (e.g. 200.0) or maxiter.\n"
        )
        return False
    return True


def print_stage_results(label: str, E0: float, E1: float, lam: float, elapsed: float) -> None:
    delta_e = E1 - E0
    err_pct = abs(lam - LAMBDA_EXP) / LAMBDA_EXP * 100 if np.isfinite(lam) else float("nan")
    print(f"\n{'═' * 60}")
    print(f"  {label}")
    print(f"{'═' * 60}")
    print(f"  E(S₀)  = {E0:+.10f} Ha")
    print(f"  E(S₁)  = {E1:+.10f} Ha")
    print(f"  ΔE     = {delta_e:.6f} Ha  ({delta_e * HARTREE_TO_EV:.2f} eV)")
    print(f"  λ_max  = {lam:.1f} nm   [expt: {LAMBDA_EXP:.0f} nm, error: {err_pct:.0f}%]")
    print(f"  Time   = {elapsed:.1f} s")
    print(f"{'═' * 60}")


def plot_stage_summary(label: str, E0: float, E1: float, lam: float, elapsed: float) -> None:
    """Quick two-panel visual after a single stage."""
    delta_e = E1 - E0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor("#0f1117")
    for ax in axes:
        ax.set_facecolor("#0f1117")

    # Energy level diagram
    ax = axes[0]
    for state_label, E, color in [("S₀", E0, "#4ade80"), ("S₁", E1, "#60a5fa")]:
        ax.hlines(E, 0.2, 0.8, color=color, linewidth=3, zorder=3)
        ax.text(0.85, E, f"{state_label}  {E:+.4f} Ha", va="center",
                color=color, fontsize=10, fontweight="bold")
    ax.annotate("", xy=(0.5, E1), xytext=(0.5, E0),
                arrowprops=dict(arrowstyle="->", color="#f59e0b", lw=2.5, mutation_scale=18))
    ax.text(0.55, (E0 + E1) / 2,
            f"ΔE = {delta_e * HARTREE_TO_EV:.2f} eV\n({lam:.0f} nm)",
            color="#f59e0b", fontsize=10, va="center", fontweight="bold")
    span = abs(E1 - E0)
    ax.set_xlim(0, 1.5)
    ax.set_ylim(E0 - span * 0.3, E1 + span * 0.3)
    ax.set_ylabel("Energy (Ha)", color="white", fontsize=11)
    ax.set_title(f"{label} — Energy Levels", color="white", fontsize=12, pad=12)
    ax.tick_params(colors="white")
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    # Bar chart
    ax = axes[1]
    display_lam = lam if np.isfinite(lam) and lam < 5000 else LAMBDA_EXP * 5
    error_pct = abs(lam - LAMBDA_EXP) / LAMBDA_EXP * 100 if np.isfinite(lam) else float("nan")
    bars = ax.bar(["VQE/VQD\n(predicted)", "Experiment"],
                  [display_lam, LAMBDA_EXP],
                  color=["#60a5fa", "#f59e0b"], width=0.4,
                  edgecolor="white", linewidth=0.8, zorder=3)
    for bar, val in zip(bars, [lam, LAMBDA_EXP]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(display_lam, LAMBDA_EXP) * 0.02,
                f"{val:.0f} nm", ha="center", color="white",
                fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(display_lam, LAMBDA_EXP) * 1.3)
    ax.set_ylabel("λ_max (nm)", color="white", fontsize=11)
    ax.set_title(f"Predicted vs Experimental λ_max\n(error: {error_pct:.0f}%)",
                 color="white", fontsize=12, pad=12)
    ax.tick_params(colors="white")
    ax.axhline(LAMBDA_EXP, color="#f59e0b", linestyle="--", lw=1.2, alpha=0.5, zorder=2)
    ax.yaxis.grid(True, color="#333", zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    plt.suptitle(f"{label} Summary  |  Time: {elapsed:.1f} s",
                 color="white", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# Molecule setup
# ─────────────────────────────────────────────
def setup_molecule():
    """Build ethylene and run Hartree-Fock. Returns the converged HF object."""
    mol = gto.M(
        atom="""
            C   0.0000   0.0000   0.6695
            C   0.0000   0.0000  -0.6695
            H   0.0000   0.9289   1.2321
            H   0.0000  -0.9289   1.2321
            H   0.0000   0.9289  -1.2321
            H   0.0000  -0.9289  -1.2321
        """,
        basis="sto-3g",
        symmetry=False,
        verbose=3,
    )
    hf = scf.RHF(mol)
    hf.kernel()
    return hf


# ─────────────────────────────────────────────
# Stage 1: CPU — (4e, 4o) → 8 qubits
# ─────────────────────────────────────────────
def run_stage1(hf, penalty_weights: float = 100.0, maxiter: int = 300, show_plot: bool = True) -> dict:
    """
    Stage 1: CPU statevector, minimal (4e, 4o) active space → 8 qubits.

    Fast proof-of-concept. Good for verifying the workflow locally.

    Parameters
    ----------
    hf : pyscf RHF object
        Converged Hartree-Fock result from setup_molecule().
    penalty_weights : float
        VQD overlap penalty strength. Higher = better orthogonality, slower convergence.
    maxiter : int
        Maximum VQE/VQD optimizer iterations.
    show_plot : bool
        Whether to display the Stage 1 quick-look visual.

    Returns
    -------
    dict with keys: E0, E1, delta_E, lambda_nm, time_s, norb, nelec
    """
    norb, nelec, mo = suggest_active_space_from_mp2(hf, max_orbitals=4)
    print(f"\nStage 1 active space: ({sum(nelec)}e, {norb}o) → {2*norb} qubits")

    t0 = time.perf_counter()

    # S₀ via VQE
    cas_gs = mcscf.CASCI(hf, norb, nelec)
    cas_gs.fcisolver = QoroSolver(
        ansatz="uccsd", backend="cpu", simulation="statevector",
        maxiter=maxiter, verbose=True,
    )
    E0 = cas_gs.kernel(mo_coeff=mo)[0]

    # S₁ via VQD
    cas_ex = mcscf.CASCI(hf, norb, nelec)
    cas_ex.fcisolver = VQDSolver(
        solver=QoroSolver(
            ansatz="uccsd", backend="cpu", simulation="statevector",
            maxiter=maxiter, verbose=True,
        ),
        num_states=2,
        penalty_weights=penalty_weights,
    )
    energies = cas_ex.kernel(mo_coeff=mo)[0]
    E1 = energies[1]

    elapsed = time.perf_counter() - t0
    lam = excitation_to_nm(E1 - E0)
    check_vqd_converged(E0, E1)
    print_stage_results("STAGE 1 — CPU, (4e, 4o)", E0, E1, lam, elapsed)

    if show_plot:
        plot_stage_summary("Stage 1 (CPU, 4e,4o)", E0, E1, lam, elapsed)

    return dict(E0=E0, E1=E1, delta_E=E1-E0, lambda_nm=lam, time_s=elapsed, norb=norb, nelec=nelec)


# ─────────────────────────────────────────────
# Stage 2: GPU Exact — (6e, 6o) → 12 qubits
# ─────────────────────────────────────────────
def run_stage2(hf, license_key: str = "YOUR_KEY", penalty_weights: float = 100.0,
               maxiter: int = 400, show_plot: bool = True) -> dict:
    """
    Stage 2: GPU statevector, medium (6e, 6o) active space → 12 qubits.

    Requires a Maestro GPU license key.

    Parameters
    ----------
    hf : pyscf RHF object
    license_key : str
        Your Maestro GPU license key.
    penalty_weights : float
        VQD overlap penalty strength.
    maxiter : int
        Maximum VQE/VQD optimizer iterations.
    show_plot : bool
        Whether to display the Stage 2 quick-look visual.

    Returns
    -------
    dict with keys: E0, E1, delta_E, lambda_nm, time_s, norb, nelec
    """
    norb, nelec, mo = suggest_active_space_from_mp2(hf, max_orbitals=6)
    print(f"\nStage 2 active space: ({sum(nelec)}e, {norb}o) → {2*norb} qubits")

    t0 = time.perf_counter()

    cas_gs = mcscf.CASCI(hf, norb, nelec)
    cas_gs.fcisolver = QoroSolver(
        ansatz="uccsd", backend="gpu", simulation="statevector",
        maxiter=maxiter, verbose=True, license_key=license_key,
    )
    E0 = cas_gs.kernel(mo_coeff=mo)[0]

    cas_ex = mcscf.CASCI(hf, norb, nelec)
    cas_ex.fcisolver = VQDSolver(
        solver=QoroSolver(
            ansatz="uccsd", backend="gpu", simulation="statevector",
            maxiter=maxiter, verbose=True, license_key=license_key,
        ),
        num_states=2,
        penalty_weights=penalty_weights,
    )
    energies = cas_ex.kernel(mo_coeff=mo)[0]
    E1 = energies[1]

    elapsed = time.perf_counter() - t0
    lam = excitation_to_nm(E1 - E0)
    check_vqd_converged(E0, E1)
    print_stage_results("STAGE 2 — GPU Exact, (6e, 6o)", E0, E1, lam, elapsed)

    if show_plot:
        plot_stage_summary("Stage 2 (GPU, 6e,6o)", E0, E1, lam, elapsed)

    return dict(E0=E0, E1=E1, delta_E=E1-E0, lambda_nm=lam, time_s=elapsed, norb=norb, nelec=nelec)


# ─────────────────────────────────────────────
# Stage 3: GPU + MPS — (12e, 12o) → 24 qubits
# ─────────────────────────────────────────────
def run_stage3(hf, license_key: str = "YOUR_KEY", penalty_weights: float = 100.0,
               maxiter: int = 500, mps_bond_dim: int = 256, show_plot: bool = True) -> dict:
    """
    Stage 3: GPU + MPS, large (12e, 12o) active space → 24 qubits.

    Maestro's hero feature — simulates 24-qubit UCCSD on a single GPU
    using Matrix Product State (MPS) compression.

    Parameters
    ----------
    hf : pyscf RHF object
    license_key : str
        Your Maestro GPU license key.
    penalty_weights : float
        VQD overlap penalty strength.
    maxiter : int
        Maximum VQE/VQD optimizer iterations.
    mps_bond_dim : int
        MPS bond dimension (higher = more accurate, slower).
    show_plot : bool
        Whether to display the Stage 3 quick-look visual.

    Returns
    -------
    dict with keys: E0, E1, delta_E, lambda_nm, time_s, norb, nelec
    """
    norb, nelec, mo = suggest_active_space_from_mp2(hf, max_orbitals=12)
    print(f"\nStage 3 active space: ({sum(nelec)}e, {norb}o) → {2*norb} qubits")

    t0 = time.perf_counter()

    cas_gs = mcscf.CASCI(hf, norb, nelec)
    cas_gs.fcisolver = QoroSolver(
        ansatz="uccsd", backend="gpu", simulation="mps",
        mps_bond_dim=mps_bond_dim, maxiter=maxiter,
        verbose=True, license_key=license_key,
    )
    E0 = cas_gs.kernel(mo_coeff=mo)[0]

    cas_ex = mcscf.CASCI(hf, norb, nelec)
    cas_ex.fcisolver = VQDSolver(
        solver=QoroSolver(
            ansatz="uccsd", backend="gpu", simulation="mps",
            mps_bond_dim=mps_bond_dim, maxiter=maxiter,
            verbose=True, license_key=license_key,
        ),
        num_states=2,
        penalty_weights=penalty_weights,
    )
    energies = cas_ex.kernel(mo_coeff=mo)[0]
    E1 = energies[1]

    elapsed = time.perf_counter() - t0
    lam = excitation_to_nm(E1 - E0)
    check_vqd_converged(E0, E1)
    print_stage_results("STAGE 3 — GPU+MPS, (12e, 12o)", E0, E1, lam, elapsed)

    if show_plot:
        plot_stage_summary("Stage 3 (GPU+MPS, 12e,12o)", E0, E1, lam, elapsed)

    return dict(E0=E0, E1=E1, delta_E=E1-E0, lambda_nm=lam, time_s=elapsed, norb=norb, nelec=nelec)


# ─────────────────────────────────────────────
# Final spectrum plot
# ─────────────────────────────────────────────
def plot_spectrum(*results: dict, labels: list[str] | None = None, save_path: str | None = None) -> None:
    """
    Plot overlaid Gaussian UV-Vis peaks for each stage result.

    Parameters
    ----------
    *results : dicts returned by run_stage1/2/3
    labels : optional list of legend labels (one per result)
    save_path : optional file path to save the figure (e.g. 'uv_vis.png')
    """
    default_labels = [
        "Stage 1 — CPU (4e,4o)",
        "Stage 2 — GPU (6e,6o)",
        "Stage 3 — GPU+MPS (12e,12o)",
    ]
    colors = ["#60a5fa", "#a78bfa", "#34d399"]
    wl = np.linspace(100, 400, 3000)

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    def gaussian(x, center, sigma=8.0):
        return np.exp(-0.5 * ((x - center) / sigma) ** 2)

    for i, result in enumerate(results):
        lam = result["lambda_nm"]
        label = (labels[i] if labels else default_labels[i] if i < len(default_labels) else f"Stage {i+1}")
        color = colors[i % len(colors)]
        if np.isfinite(lam) and 100 < lam < 400:
            ax.plot(wl, gaussian(wl, lam), color=color, lw=2,
                    label=f"{label}  ({lam:.0f} nm)")
            ax.axvline(lam, color=color, linestyle=":", lw=1, alpha=0.6)
        else:
            print(f"⚠️  Skipping {label}: λ = {lam:.0f} nm is out of plot range (VQD may have collapsed)")

    ax.axvline(LAMBDA_EXP, color="white", linestyle="--", lw=2,
               label=f"Experiment ({LAMBDA_EXP:.0f} nm)")

    ax.set_xlabel("Wavelength (nm)", color="white", fontsize=13)
    ax.set_ylabel("Intensity (arb. units)", color="white", fontsize=13)
    ax.set_title("Ethylene UV-Vis Absorption — Maestro VQE/VQD Predictions",
                 color="white", fontsize=14, pad=14)
    ax.tick_params(colors="white")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(facecolor="#1e2130", edgecolor="#444", labelcolor="white", fontsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.yaxis.grid(True, color="#1e2130", zorder=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
        print(f"Saved spectrum to {save_path}")
    plt.show()


# ─────────────────────────────────────────────
# Main — run all three stages
# ─────────────────────────────────────────────
if __name__ == "__main__":
    LICENSE_KEY = "YOUR_KEY"   # ← Replace with your Maestro GPU license key

    hf = setup_molecule()

    r1 = run_stage1(hf)
    # r2 = run_stage2(hf, license_key=LICENSE_KEY)
    # r3 = run_stage3(hf, license_key=LICENSE_KEY)

    exit()

    plot_spectrum(r1, r2, r3, save_path="ethylene_uv_vis.png")

    # Summary table
    print(f"\n{'═' * 68}")
    print(f"  {'Stage':<32} {'λ_max (nm)':>12} {'Error':>10} {'Time':>8}")
    print(f"{'─' * 68}")
    for label, r in [
        ("Stage 1 — CPU (4e,4o)",        r1),
        ("Stage 2 — GPU exact (6e,6o)",  r2),
        ("Stage 3 — GPU+MPS (12e,12o)",  r3),
    ]:
        lam = r["lambda_nm"]
        err = abs(lam - LAMBDA_EXP) / LAMBDA_EXP * 100 if np.isfinite(lam) else float("nan")
        print(f"  {label:<32} {lam:>12.1f} {err:>9.0f}% {r['time_s']:>7.1f}s")
    print(f"  {'Experiment':<32} {LAMBDA_EXP:>12.0f}")
    print(f"{'═' * 68}")
