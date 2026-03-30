#!/usr/bin/env python3
# Copyright 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Butyronitrile C≡N Dissociation — VQE on Maestro GPU
=====================================================

Computes the potential energy surface for the dissociation of the nitrile
(C≡N) group in butyronitrile (C₃H₇CN). This system is relevant for
lithium-ion battery electrolytes and solar cell technologies.

Two stages are provided: a baseline active space commonly used in the
literature, and an extended active space that pushes beyond typical
demonstrations — showing Maestro GPU's ability to handle larger problems.

Two stages
----------

  Stage 1 — Match:    CAS(8,8)  / STO-3G   → 16 qubits  (same as QRunch)
  Stage 2 — Outdo:    CAS(14,14)/ 6-31G*   → 28 qubits  (beyond QRunch)

The molecule is butyronitrile (CH₃CH₂CH₂C≡N), 12 atoms. We stretch the
C≡N bond from its equilibrium distance (~1.16 Å) through complete
dissociation, keeping all other atoms frozen.

Usage
-----
  # Stage 1 only — baseline setup (16 qubits, CPU ok)
  python run_cn_dissociation.py --stage 1 --cpu

  # Stage 2 — larger active space (28 qubits, GPU recommended)
  python run_cn_dissociation.py --stage 2 --gpu

  # Both stages, GPU vs CPU comparison
  python run_cn_dissociation.py --stage both --both

  # Quick test (3 geometry points)
  python run_cn_dissociation.py --stage 1 --cpu --quick

Results are saved to results/ as JSON.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from pyscf import gto, scf, mcscf

from qoro_pyscf import QoroSolver


# ═══════════════════════════════════════════════════════════════════════════════
# Butyronitrile geometry
# ═══════════════════════════════════════════════════════════════════════════════

# Equilibrium geometry for butyronitrile (CH₃CH₂CH₂C≡N).
# Source: optimised geometry.
# Only the N atom position changes during dissociation.
EQUILIBRIUM_ATOMS = [
    ("C",  ( 0.71951, -0.72284,  0.00000)),
    ("C",  ( 0.00000,  0.63021,  0.00000)),
    ("C",  (-1.45592,  0.49995,  0.00000)),  # C attached to N (index 2)
    ("N",  (-2.20171,  0.42056,  0.00000)),  # N being dissociated (index 3)
    ("C",  ( 2.23248, -0.56128,  0.00000)),
    ("H",  ( 0.39949, -1.29704,  0.87880)),
    ("H",  ( 0.39949, -1.29704, -0.87880)),
    ("H",  ( 0.28619,  1.21996,  0.88101)),
    ("H",  ( 0.28619,  1.21996, -0.88101)),
    ("H",  ( 2.72786, -1.53832,  0.00000)),
    ("H",  ( 2.57639, -0.01393, -0.88677)),
    ("H",  ( 2.57639, -0.01393,  0.88677)),
]

# C atom (index 2) position — the N is pulled away from this
C_ANCHOR = np.array([-1.45592, 0.49995, 0.00000])

# Direction of dissociation (unit vector from C toward N)
_n_eq = np.array([-2.20171, 0.42056, 0.00000])
CN_DIRECTION = (_n_eq - C_ANCHOR) / np.linalg.norm(_n_eq - C_ANCHOR)

# Equilibrium C≡N distance (Å)
CN_EQ_DIST = np.linalg.norm(_n_eq - C_ANCHOR)

# ── 9-frame C≡N distances for the dissociation scan ──
# Frame 0: C≡N ≈ 1.157 Å (equilibrium)
# Frame 8: C≡N ≈ 2.749 Å (dissociated)
CN_DISTANCES = [
    1.157, 1.356, 1.555, 1.754, 1.953, 2.152, 2.351, 2.550, 2.749
]


def build_butyronitrile(cn_distance: float, basis: str = "sto-3g") -> gto.Mole:
    """
    Build butyronitrile with the C≡N bond set to `cn_distance` Å.

    All atoms except N are kept at their equilibrium positions.
    The N atom is placed along the original C→N direction.
    """
    atoms = []
    for i, (elem, coords) in enumerate(EQUILIBRIUM_ATOMS):
        if i == 3:  # Nitrogen
            n_pos = C_ANCHOR + CN_DIRECTION * cn_distance
            atoms.append(f"{elem}  {n_pos[0]:.5f}  {n_pos[1]:.5f}  {n_pos[2]:.5f}")
        else:
            atoms.append(f"{elem}  {coords[0]:.5f}  {coords[1]:.5f}  {coords[2]:.5f}")

    return gto.M(
        atom="\n".join(atoms),
        basis=basis,
        symmetry=False,
        verbose=0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Stage configurations
# ═══════════════════════════════════════════════════════════════════════════════

STAGES = {
    0: {
        "label": "Stage 0 — CPU Quick",
        "description": "Compact active space: CAS(4,4)/STO-3G = 8 qubits (fast CPU demo)",
        "norb": 4,
        "nelec": 4,
        "basis": "sto-3g",
        # UCCSD with COBYLA: ~2 min per geometry on CPU, good for local demos.
        # Uses a smaller active space to keep runtime practical without a GPU.
        "ansatz": "uccsd",
        "maxiter": 400,
        "optimizer": "COBYLA",
        "learning_rate": None,
        "adapt_greedy": False,
        "adapt_pool": "sd",
        "n_qubits": 8,
        "distances": CN_DISTANCES,
    },
    1: {
        "label": "Stage 1 — Baseline",
        "description": "Standard active space: CAS(8,8)/STO-3G = 16 qubits",
        "norb": 8,
        "nelec": 8,
        "basis": "sto-3g",
        # ADAPT-VQE with greedy Rotosolve: each ADAPT step adds one
        # operator and analytically optimises its parameter in just
        # 3 circuit evaluations.  Fast and accurate.
        # NOTE: gradient screening (784 operators × 2 evals) takes ~10 min/step
        # on CPU — run with --gpu for practical runtimes.
        "ansatz": "adapt",
        "maxiter": 100,
        "optimizer": "ROTOSOLVE",
        "learning_rate": None,
        "adapt_greedy": True,
        "adapt_pool": "d",
        "n_qubits": 16,
        "distances": CN_DISTANCES,
    },
    2: {
        "label": "Stage 2 — Extended",
        "description": "Larger active space + better basis: CAS(14,14)/6-31G* = 28 qubits",
        "norb": 14,
        "nelec": 14,
        "basis": "6-31g*",
        "ansatz": "uccsd",
        "maxiter": 800,
        "optimizer": "adam",
        "learning_rate": 0.005,
        "n_qubits": 28,
        "distances": CN_DISTANCES,
    },
}


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


def compute_mp2_energy(mol: gto.Mole) -> float:
    """MP2 energy — the simplest correlated method."""
    from pyscf import mp
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    mp2 = mp.MP2(mf)
    mp2.verbose = 0
    mp2.kernel()
    return mp2.e_tot


# ═══════════════════════════════════════════════════════════════════════════════
# VQE runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_vqe_point(
    mol: gto.Mole,
    norb: int,
    nelec: int,
    backend: str = "gpu",
    maxiter: int = 500,
    optimizer: str = "L-BFGS-B",
    ansatz: str = "uccsd",
    learning_rate: float | None = None,
    previous_params: np.ndarray | None = None,
    adapt_greedy: bool = False,
    adapt_pool: str = "sd",
) -> dict:
    """Run VQE at one geometry and return results dict."""
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    cas = mcscf.CASCI(mf, norb, nelec)
    cas.verbose = 0

    solver_kwargs = dict(
        ansatz=ansatz,
        optimizer=optimizer,
        maxiter=maxiter,
        backend=backend,
        verbose=True,
        adapt_greedy=adapt_greedy,
        adapt_pool=adapt_pool,
    )
    if learning_rate is not None:
        solver_kwargs["learning_rate"] = learning_rate

    solver = QoroSolver(**solver_kwargs)

    # Warm-start from previous geometry
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
# PES scan for one stage
# ═══════════════════════════════════════════════════════════════════════════════

def run_stage(stage_num: int, backends: list[str], quick: bool = False):
    """Run one stage of the butyronitrile dissociation scan."""
    cfg = STAGES[stage_num]
    distances = cfg["distances"][:3] if quick else cfg["distances"]

    norb = cfg["norb"]
    nelec = cfg["nelec"]
    basis = cfg["basis"]
    n_qubits = cfg["n_qubits"]

    print(flush=True)
    print("=" * 78, flush=True)
    print(f"  {cfg['label']}", flush=True)
    print(f"  {cfg['description']}", flush=True)
    print(f"  Ansatz: {cfg['ansatz'].upper()}  |  Optimizer: {cfg['optimizer']}", flush=True)
    print(f"  C≡N distances: {distances[0]:.3f} → {distances[-1]:.3f} Å ({len(distances)} points)", flush=True)
    print(f"  Backends: {', '.join(b.upper() for b in backends)}", flush=True)
    print("=" * 78, flush=True)

    results = {
        "metadata": {
            "stage": stage_num,
            "label": cfg["label"],
            "basis": basis,
            "norb": norb,
            "nelec": nelec,
            "n_qubits": n_qubits,
            "ansatz": cfg["ansatz"],
            "optimizer": cfg["optimizer"],
            "maxiter": cfg["maxiter"],
        },
        "cn_distances": distances,
        "hf_energies": [],
        "fci_energies": [],
        "mp2_energies": [],
    }
    for backend in backends:
        results[f"vqe_{backend}_energies"] = []
        results[f"vqe_{backend}_times"] = []

    # ─── Reference energies ──────────────────────────────────────────────
    print("\n  Computing reference energies...", flush=True)
    print(f"  {'d(C≡N)':>10s}  {'HF':>14s}  {'MP2':>14s}  {'FCI(CAS)':>14s}", flush=True)
    print("  " + "-" * 58, flush=True)
    sys.stdout.flush()

    for r in distances:
        mol = build_butyronitrile(r, basis)

        e_hf = compute_hf_energy(mol)
        results["hf_energies"].append(e_hf)

        try:
            e_mp2 = compute_mp2_energy(mol)
            results["mp2_energies"].append(e_mp2)
        except Exception:
            results["mp2_energies"].append(None)

        # FCI within CAS (only feasible for stage 1; stage 2 may be slow)
        try:
            e_fci = compute_fci_casci(mol, norb, nelec)
            results["fci_energies"].append(e_fci)
        except Exception:
            results["fci_energies"].append(None)

        mp2_str = f"{e_mp2:+14.8f}" if results["mp2_energies"][-1] is not None else "       N/A     "
        fci_str = f"{e_fci:+14.8f}" if results["fci_energies"][-1] is not None else "       N/A     "
        print(f"  {r:10.3f}  {e_hf:+14.8f}  {mp2_str}  {fci_str}", flush=True)

    # ─── VQE on each backend ─────────────────────────────────────────────
    for backend in backends:
        print(f"\n  ─── VQE on {backend.upper()} ({n_qubits} qubits) ───")
        prev_params = None

        for i, r in enumerate(distances):
            mol = build_butyronitrile(r, basis)

            print(f"    d(C≡N) = {r:.3f} Å  ... ", end="", flush=True)
            try:
                out = run_vqe_point(
                    mol, norb, nelec,
                    backend=backend,
                    maxiter=cfg["maxiter"],
                    optimizer=cfg["optimizer"],
                    ansatz=cfg["ansatz"],
                    learning_rate=cfg["learning_rate"],
                    previous_params=prev_params,
                    adapt_greedy=cfg.get("adapt_greedy", False),
                    adapt_pool=cfg.get("adapt_pool", "sd"),
                )
                results[f"vqe_{backend}_energies"].append(out["energy"])
                results[f"vqe_{backend}_times"].append(out["wall_time"])
                prev_params = np.array(out["optimal_params"]) if out["optimal_params"] else None

                # Error vs FCI if available
                if results["fci_energies"][i] is not None:
                    error_mha = abs(out["energy"] - results["fci_energies"][i]) * 1000
                    err_str = f"err = {error_mha:.2f} mHa"
                else:
                    err_str = "no FCI ref"

                status = "✓" if out["converged"] else "✗"
                print(f"E = {out['energy']:+.8f}  |  {err_str}  |  "
                      f"{out['wall_time']:.1f}s  {status}")
            except Exception as e:
                results[f"vqe_{backend}_energies"].append(None)
                results[f"vqe_{backend}_times"].append(None)
                print(f"FAILED: {e}")

    # ─── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 78)

    # Accuracy summary
    for backend in backends:
        vqe_energies = results[f"vqe_{backend}_energies"]
        fci_energies = results["fci_energies"]
        errors = []
        for ve, fe in zip(vqe_energies, fci_energies):
            if ve is not None and fe is not None:
                errors.append(abs(ve - fe) * 1000)
        if errors:
            print(f"  {backend.upper()} — Max error: {max(errors):.2f} mHa  |  "
                  f"Mean error: {np.mean(errors):.2f} mHa")
            chem_acc = "✓ YES" if max(errors) < 1.6 else "✗ NO"
            print(f"           Chemical accuracy (< 1.6 mHa): {chem_acc}")

    # Timing comparison
    if len(backends) == 2:
        cpu_times = [t for t in results["vqe_cpu_times"] if t is not None]
        gpu_times = [t for t in results["vqe_gpu_times"] if t is not None]
        if cpu_times and gpu_times:
            avg_cpu = np.mean(cpu_times)
            avg_gpu = np.mean(gpu_times)
            speedup = avg_cpu / avg_gpu
            print(f"\n  GPU speedup:    {speedup:.1f}×")
            print(f"  Total CPU time: {sum(cpu_times):.1f}s")
            print(f"  Total GPU time: {sum(gpu_times):.1f}s")

    # ─── Save results ────────────────────────────────────────────────────
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    label = f"butyronitrile_stage{stage_num}_{basis.replace('*', 'star')}_cas{norb}"
    out_path = out_dir / f"{label}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")
    print("=" * 78)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Butyronitrile C≡N Dissociation — VQE on Maestro",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  --stage 0     CPU quick:  CAS(4,4)/STO-3G = 8 qubits  (~2 min/point on CPU)
  --stage 1     Baseline:   CAS(8,8)/STO-3G = 16 qubits  (GPU recommended)
  --stage 2     Extended:   CAS(14,14)/6-31G* = 28 qubits (GPU required)
  --stage both  Run stages 1 + 2 sequentially

Examples:
  python run_cn_dissociation.py --stage 0 --cpu          # Fast CPU demo (~20 min)
  python run_cn_dissociation.py --stage 0 --cpu --quick  # 3-point test (~6 min)
  python run_cn_dissociation.py --stage 1 --gpu          # Baseline, GPU
  python run_cn_dissociation.py --stage 1 --both         # GPU vs CPU comparison
  python run_cn_dissociation.py --stage 2 --gpu          # Big run, GPU only
  python run_cn_dissociation.py --stage both --gpu        # Full demo
""",
    )

    # Backend
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--gpu", action="store_true", help="Run VQE on GPU only")
    mode.add_argument("--cpu", action="store_true", help="Run VQE on CPU only")
    mode.add_argument("--both", action="store_true", help="Run on both and compare")

    # Stage
    parser.add_argument("--stage", type=str, required=True,
                        choices=["0", "1", "2", "both"],
                        help="Which stage to run: 0 (cpu-quick), 1 (match), 2 (outdo), both")

    # Options
    parser.add_argument("--quick", action="store_true",
                        help="Only run 3 geometry points (for quick testing)")

    args = parser.parse_args()

    backends = []
    if args.both:
        backends = ["cpu", "gpu"]
    elif args.gpu:
        backends = ["gpu"]
    else:
        backends = ["cpu"]

    stages_to_run = [1, 2] if args.stage == "both" else [int(args.stage)]

    print()
    print("╔" + "═" * 76 + "╗")
    print("║  BUTYRONITRILE C≡N DISSOCIATION — VQE ON MAESTRO" + " " * 27 + "║")
    print("║  Battery electrolyte chemistry benchmark" + " " * 35 + "║")
    print("╚" + "═" * 76 + "╝")

    all_results = {}
    for stage_num in stages_to_run:
        all_results[stage_num] = run_stage(stage_num, backends, quick=args.quick)

    # Final comparison across stages
    if len(stages_to_run) == 2:
        print("\n" + "=" * 78)
        print("  CROSS-STAGE COMPARISON")
        print("=" * 78)
        for s in stages_to_run:
            cfg = STAGES[s]
            print(f"\n  {cfg['label']}")
            print(f"    {cfg['description']}")
            for backend in backends:
                times = [t for t in all_results[s].get(f"vqe_{backend}_times", []) if t is not None]
                energies = [e for e in all_results[s].get(f"vqe_{backend}_energies", []) if e is not None]
                if times:
                    print(f"    {backend.upper()}: {len(energies)} points  |  "
                          f"total {sum(times):.1f}s  |  avg {np.mean(times):.1f}s/point")

    print("\n  Done!\n")


if __name__ == "__main__":
    main()
