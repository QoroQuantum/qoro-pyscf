# Copyright 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Example 14 — QSCI (Quantum-Selected Configuration Interaction)
==============================================================

Demonstrates QSCI on the LiH dissociation curve (6 qubits).

This example shows the key advantage of QSCI over raw VQE: even when the
VQE optimizer doesn't fully converge (using a hardware-efficient ansatz
with limited iterations), QSCI recovers near-exact energies by classically
diagonalizing in the quantum-selected subspace.

Reference: Kanno et al., arXiv:2302.11320

Usage
-----
    python examples/14_qsci.py
"""

import numpy as np
from pyscf import gto, scf, mcscf, fci

from qoro_maestro_pyscf import MaestroSolver, QSCISolver


def compute_energies(bond_length: float) -> dict:
    """Compute HF, VQE, QSCI, and FCI energies for LiH at a given bond length."""
    mol = gto.M(
        atom=f"Li 0 0 0; H 0 0 {bond_length}",
        basis="sto-3g",
        verbose=0,
    )
    hf = scf.RHF(mol).run()

    # FCI reference
    cisolver = fci.FCI(hf)
    e_fci = cisolver.kernel()[0]

    # Active space: (2e, 3o) = 6 qubits
    # Freeze core Li 1s, correlate 2 valence electrons in 3 orbitals
    norb, nelec = 3, 2

    # --- VQE with hardware-efficient ansatz (intentionally imperfect) ---
    cas_vqe = mcscf.CASCI(hf, norb, nelec)
    cas_vqe.fcisolver = MaestroSolver(
        ansatz="hardware_efficient",
        ansatz_layers=2,
        maxiter=100,   # limited iterations — shows QSCI's recovery power
        verbose=False,
    )
    e_vqe = cas_vqe.kernel()[0]

    # --- QSCI with the same ansatz ---
    cas_qsci = mcscf.CASCI(hf, norb, nelec)
    inner = MaestroSolver(
        ansatz="hardware_efficient",
        ansatz_layers=2,
        maxiter=100,
        verbose=False,
    )
    cas_qsci.fcisolver = QSCISolver(
        inner_solver=inner,
        n_samples=500,
        verbose=False,
    )
    e_qsci = cas_qsci.kernel()[0]

    return {
        "r": bond_length,
        "hf": hf.e_tot,
        "vqe": e_vqe,
        "qsci": e_qsci,
        "fci": e_fci,
    }


def main():
    print("=" * 62)
    print("  LiH Dissociation Curve: VQE vs QSCI (6 qubits)")
    print("  Ansatz: hardware-efficient (2 layers, 100 iters)")
    print("  Active space: (2e, 3o) — freeze Li 1s core")
    print("=" * 62)

    # Scan bond lengths from equilibrium to stretched
    bond_lengths = [1.0, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0]

    results = []
    for r in bond_lengths:
        print(f"\n  Computing r = {r:.1f} Å ...", end=" ", flush=True)
        data = compute_energies(r)
        results.append(data)
        vqe_err = abs(data["vqe"] - data["fci"])
        qsci_err = abs(data["qsci"] - data["fci"])
        print(f"VQE err={vqe_err:.2e}  QSCI err={qsci_err:.2e}")

    # Summary table
    print("\n" + "=" * 62)
    print(f"  {'r (Å)':>6}  {'E(HF)':>12}  {'E(VQE)':>12}  "
          f"{'E(QSCI)':>12}  {'E(FCI)':>12}")
    print("-" * 62)
    for d in results:
        print(f"  {d['r']:6.1f}  {d['hf']:+12.7f}  {d['vqe']:+12.7f}  "
              f"{d['qsci']:+12.7f}  {d['fci']:+12.7f}")

    # Error comparison
    print("\n" + "=" * 62)
    print(f"  {'r (Å)':>6}  {'VQE error':>12}  {'QSCI error':>12}  "
          f"{'Improvement':>12}")
    print("-" * 62)
    for d in results:
        vqe_err = abs(d["vqe"] - d["fci"])
        qsci_err = abs(d["qsci"] - d["fci"])
        improvement = vqe_err - qsci_err
        print(f"  {d['r']:6.1f}  {vqe_err:12.2e}  {qsci_err:12.2e}  "
              f"{improvement:+12.2e}")
    print("=" * 62)


if __name__ == "__main__":
    main()
