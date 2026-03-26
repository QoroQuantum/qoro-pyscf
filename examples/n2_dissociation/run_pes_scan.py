#!/usr/bin/env python3
# Copyright 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Breaking the Triple Bond: N₂ Dissociation with VQE on Qoro
==============================================================

This script computes the potential energy surface of N₂ along the N–N
bond stretch using VQE on Qoro. It demonstrates:

  1. AVAS-based active-space auto-selection of the 2p valence orbitals
  2. UCCSD-VQE with the Adam optimiser (parameter-shift gradients)
  3. GPU vs CPU wall-time comparison at 20 qubits
  4. Optional Z₂ tapering to reduce from 20 → 18 qubits
  5. FCI reference for accuracy verification

The N₂ triple bond is the textbook multireference problem: CCSD(T)
famously fails at stretched geometries because the wavefunction
develops strong multiconfigurational character as the three bonding
pairs break simultaneously.

Active space
------------
  CAS(6,6) → 12 qubits (minimal, fast prototyping)
  CAS(10,10) → 20 qubits (full valence, GPU advantage)

Usage
-----
  # Quick run (CAS(6,6), 12 qubits, ~5 min per geometry)
  python run_pes_scan.py --cas 6 --basis sto-3g --gpu

  # Full benchmark (CAS(10,10), 20 qubits, GPU vs CPU)
  python run_pes_scan.py --cas 10 --basis cc-pvdz --both

  # With tapering (saves 2 qubits)
  python run_pes_scan.py --cas 10 --basis cc-pvdz --gpu --taper

  # Control number of geometry points
  python run_pes_scan.py --cas 6 --gpu --npoints 20

Results are saved to results/ as JSON for plotting.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from pyscf import gto, scf, mcscf, fci

from qoro_pyscf import QoroSolver


# ═══════════════════════════════════════════════════════════════════════════════
# Molecule builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_n2(bond_length: float, basis: str = "cc-pvdz") -> gto.Mole:
    """Build N₂ at a given bond length (in Ångström)."""
    return gto.M(
        atom=f"N 0 0 0; N 0 0 {bond_length}",
        basis=basis,
        symmetry=False,
        verbose=0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Reference calculations
# ═══════════════════════════════════════════════════════════════════════════════

def compute_hf_energy(mol: gto.Mole) -> float:
    """Restricted Hartree-Fock energy."""
    mf = scf.RHF(mol)
    mf.verbose = 0
    return mf.kernel()


def compute_fci_casci(mol: gto.Mole, norb: int, nelec: int) -> float:
    """Exact FCI within the active space (PySCF CASCI)."""
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    cas = mcscf.CASCI(mf, norb, nelec)
    cas.verbose = 0
    return cas.kernel()[0]


def compute_ccsd_t(mol: gto.Mole) -> float:
    """CCSD(T) energy — the 'gold standard' that breaks for stretched N₂."""
    from pyscf import cc
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    mycc = cc.CCSD(mf)
    mycc.verbose = 0
    mycc.kernel()
    et = mycc.ccsd_t()
    return mycc.e_tot + et


# ═══════════════════════════════════════════════════════════════════════════════
# VQE runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_vqe_point(
    mol: gto.Mole,
    norb: int,
    nelec: int,
    backend: str = "gpu",
    taper: bool = False,
    maxiter: int = 300,
    optimizer: str = "adam",
    ansatz: str = "upccd",
    learning_rate: float = 0.002,
    previous_params: np.ndarray | None = None,
) -> dict:
    """
    Run VQE at one geometry and return results dict.

    Uses the previous geometry's optimal parameters as the initial point
    (warm-starting) for faster convergence along the PES scan.
    """
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    cas = mcscf.CASCI(mf, norb, nelec)
    cas.verbose = 0

    solver = QoroSolver(
        ansatz=ansatz,
        optimizer=optimizer,
        learning_rate=learning_rate,
        maxiter=maxiter,
        backend=backend,
        taper=taper,
        verbose=False,
    )

    # Warm-start from previous geometry (crucial for PES scans)
    if previous_params is not None:
        solver.initial_point = previous_params

    cas.fcisolver = solver

    t0 = time.perf_counter()
    energy = cas.kernel()[0]
    wall_time = time.perf_counter() - t0

    return {
        "energy": float(energy),
        "wall_time": wall_time,
        "converged": solver.converged,
        "n_params": len(solver.optimal_params) if solver.optimal_params is not None else 0,
        "optimal_params": solver.optimal_params.tolist() if solver.optimal_params is not None else [],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PES scan
# ═══════════════════════════════════════════════════════════════════════════════

def run_pes_scan(args):
    """Run the full potential energy surface scan."""
    # Bond lengths: from near-equilibrium (1.0 Å) to stretched (3.0 Å)
    distances = np.linspace(
        args.r_min, args.r_max, args.npoints
    ).tolist()

    norb = args.cas
    nelec = norb  # CAS(n,n): n electrons in n orbitals
    n_qubits = 2 * norb

    backends = []
    if args.both:
        backends = ["cpu", "gpu"]
    elif args.gpu:
        backends = ["gpu"]
    else:
        backends = ["cpu"]

    print("=" * 78)
    print("  N₂ DISSOCIATION — VQE ON QORO")
    print(f"  Basis: {args.basis}  |  CAS({nelec},{norb})  |  {n_qubits} qubits")
    if args.taper:
        print(f"  Z₂ tapering: ON (saves ~2 qubits → {n_qubits - 2})")
    print(f"  Optimizer: {args.optimizer.upper()}  |  maxiter: {args.maxiter}")
    print(f"  Distances: {args.r_min:.2f} → {args.r_max:.2f} Å ({args.npoints} points)")
    print(f"  Backends: {', '.join(b.upper() for b in backends)}")
    print("=" * 78)

    # Collect results
    results = {
        "metadata": {
            "basis": args.basis,
            "norb": norb,
            "nelec": nelec,
            "n_qubits": n_qubits,
            "taper": args.taper,
            "optimizer": args.optimizer,
            "maxiter": args.maxiter,
        },
        "distances": distances,
        "hf_energies": [],
        "fci_energies": [],
        "ccsd_t_energies": [],
    }
    for backend in backends:
        results[f"vqe_{backend}_energies"] = []
        results[f"vqe_{backend}_times"] = []

    # ─── Reference energies ──────────────────────────────────────────────
    print("\n  Computing reference energies...")
    for i, r in enumerate(distances):
        mol = build_n2(r, args.basis)

        # HF
        e_hf = compute_hf_energy(mol)
        results["hf_energies"].append(e_hf)

        # FCI within CAS
        e_fci = compute_fci_casci(mol, norb, nelec)
        results["fci_energies"].append(e_fci)

        # CCSD(T) — may fail at stretched geometries
        try:
            e_ccsd_t = compute_ccsd_t(mol)
            results["ccsd_t_energies"].append(e_ccsd_t)
        except Exception:
            results["ccsd_t_energies"].append(None)

        marker = "✓" if results["ccsd_t_energies"][-1] is not None else "✗ (CCSD(T) failed)"
        print(f"    r = {r:.2f} Å  |  HF = {e_hf:+.8f}  |  FCI = {e_fci:+.8f}  |  CCSD(T) {marker}")

    # ─── VQE on each backend ─────────────────────────────────────────────
    for backend in backends:
        print(f"\n  ─── VQE on {backend.upper()} ───")
        prev_params = None

        for i, r in enumerate(distances):
            mol = build_n2(r, args.basis)

            print(f"    r = {r:.2f} Å  ... ", end="", flush=True)
            try:
                out = run_vqe_point(
                    mol, norb, nelec,
                    backend=backend,
                    taper=args.taper,
                    maxiter=args.maxiter,
                    optimizer=args.optimizer,
                    ansatz=args.ansatz,
                    learning_rate=args.learning_rate,
                    previous_params=prev_params,
                )
                results[f"vqe_{backend}_energies"].append(out["energy"])
                results[f"vqe_{backend}_times"].append(out["wall_time"])
                prev_params = np.array(out["optimal_params"]) if out["optimal_params"] else None

                error_mha = abs(out["energy"] - results["fci_energies"][i]) * 1000
                status = "✓" if out["converged"] else "✗"
                print(f"E = {out['energy']:+.8f}  |  err = {error_mha:.2f} mHa  |  "
                      f"{out['wall_time']:.1f}s  {status}")
            except Exception as e:
                results[f"vqe_{backend}_energies"].append(None)
                results[f"vqe_{backend}_times"].append(None)
                print(f"FAILED: {e}")

    # ─── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    if len(backends) == 2:
        cpu_times = [t for t in results["vqe_cpu_times"] if t is not None]
        gpu_times = [t for t in results["vqe_gpu_times"] if t is not None]
        if cpu_times and gpu_times:
            avg_cpu = np.mean(cpu_times)
            avg_gpu = np.mean(gpu_times)
            speedup = avg_cpu / avg_gpu
            total_cpu = sum(cpu_times)
            total_gpu = sum(gpu_times)
            print(f"  Average speedup:  {speedup:.1f}×")
            print(f"  Total CPU time:   {total_cpu:.1f}s")
            print(f"  Total GPU time:   {total_gpu:.1f}s")
            print(f"  Time saved:       {total_cpu - total_gpu:.1f}s")

    # ─── Save results ────────────────────────────────────────────────────
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    label = f"n2_{args.basis}_cas{norb}"
    if args.taper:
        label += "_tapered"
    out_path = out_dir / f"{label}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")
    print("=" * 78)


# ═══════════════════════════════════════════════════════════════════════════════
# Presets
# ═══════════════════════════════════════════════════════════════════════════════

PRESETS = {
    "test": {
        # Quick verification (~5 min on CPU)
        "cas": 6, "basis": "sto-3g", "ansatz": "upccd",
        "optimizer": "adam", "learning_rate": 0.01, "maxiter": 200,
        "npoints": 3, "r_min": 0.9, "r_max": 1.5,
    },
    "prod": {
        # Publication hero run (~2–4 hrs on GPU, very slow on CPU)
        "cas": 10, "basis": "cc-pvdz", "ansatz": "uccsd",
        "optimizer": "adam", "learning_rate": 0.005, "maxiter": 1000,
        "npoints": 15, "r_min": 0.8, "r_max": 3.0,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="N₂ Dissociation PES — VQE benchmark on Qoro",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  --test   Quick verification: CAS(6,6)/STO-3G, UpCCD, 3 pts, 200 iters
  --prod   Hero run:           CAS(10,10)/cc-pVDZ, UCCSD, 15 pts, 1000 iters

Individual flags override preset values, e.g.:
  python run_pes_scan.py --prod --gpu --maxiter 500
""",
    )

    # Backend
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--gpu", action="store_true", help="Run VQE on GPU only")
    mode.add_argument("--cpu", action="store_true", help="Run VQE on CPU only")
    mode.add_argument("--both", action="store_true", help="Run on both and compare")

    # Preset
    preset = parser.add_mutually_exclusive_group()
    preset.add_argument("--test", action="store_const", dest="preset", const="test",
                        help="Quick verification preset (CAS(6,6), 3 pts, ~5 min)")
    preset.add_argument("--prod", action="store_const", dest="preset", const="prod",
                        help="Publication hero run (CAS(10,10), 15 pts, ~2-4 hrs GPU)")

    # Individual overrides (defaults are None so we can detect user overrides)
    parser.add_argument("--cas", type=int, default=None, choices=[6, 8, 10])
    parser.add_argument("--basis", type=str, default=None)
    parser.add_argument("--taper", action="store_true")
    parser.add_argument("--ansatz", type=str, default=None,
                        choices=["upccd", "uccsd", "hardware_efficient"])
    parser.add_argument("--optimizer", type=str, default=None,
                        choices=["adam", "COBYLA", "L-BFGS-B"])
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--npoints", type=int, default=None)
    parser.add_argument("--r-min", type=float, default=None)
    parser.add_argument("--r-max", type=float, default=None)

    args = parser.parse_args()

    # Apply preset defaults, then user overrides
    base = PRESETS.get(args.preset or "test")  # default to test if no preset
    for key, val in base.items():
        if getattr(args, key) is None:
            setattr(args, key, val)

    run_pes_scan(args)


if __name__ == "__main__":
    main()
