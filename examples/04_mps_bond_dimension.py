#!/usr/bin/env python3
# This file is part of qoro-maestro-pyscf.
#
# Copyright (C) 2026 Qoro Quantum Ltd.
#
# qoro-maestro-pyscf is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# qoro-maestro-pyscf is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with qoro-maestro-pyscf. If not, see <https://www.gnu.org/licenses/>.

"""
Example 4 — Statevector vs MPS Simulation Modes
=================================================

Compares Maestro's two simulation backends — exact statevector and
MPS (Matrix Product State) — on the same VQE problem.

What is MPS simulation?
-----------------------
Statevector stores all 2^N amplitudes, limiting you to ~25-30 qubits.
MPS compresses the state into a tensor chain with bond dimension χ:

    χ = 2^(N/2)   → exact (equivalent to statevector)
    χ < 2^(N/2)   → approximate, but much cheaper for large N

For chemistry VQE circuits, which use local entangling gates (CNOT
ladders), MPS is remarkably efficient. Maestro runs MPS on the GPU,
enabling 50-100+ qubit simulations on a single card.

What this example shows
-----------------------
We run the *same* VQE on LiH using both backends and compare:

1. **Accuracy** — do they find the same energy?
2. **Chemistry** — do they both achieve chemical accuracy vs FCI?

Since LiH (4 qubits) has low entanglement, both modes should agree.
For larger systems (>20 qubits), MPS gives a practical speedup while
statevector becomes infeasible.

Usage
-----
    python 04_mps_bond_dimension.py          # CPU
    python 04_mps_bond_dimension.py --gpu    # GPU
"""

import argparse
import time

from pyscf import gto, scf, mcscf
from qoro_maestro_pyscf import MaestroSolver


def main():
    parser = argparse.ArgumentParser(
        description="Statevector vs MPS comparison"
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU backend")
    parser.add_argument("--cpu", dest="gpu", action="store_false")
    parser.set_defaults(gpu=False)
    args = parser.parse_args()

    backend = "gpu" if args.gpu else "cpu"

    print("=" * 72)
    print("  STATEVECTOR vs MPS SIMULATION MODES")
    print(f"  Backend: {backend.upper()}")
    print("=" * 72)

    # --- Molecule: LiH ---
    mol = gto.M(
        atom="Li 0 0 0; H 0 0 1.6",
        basis="sto-3g",
        verbose=0,
    )
    hf_obj = scf.RHF(mol).run()

    norb = 2
    nelec = 2
    n_qubits = 2 * norb

    print(f"\n  Molecule        : LiH (STO-3G, d=1.6 Å)")
    print(f"  Active space    : ({nelec}e, {norb}o) → {n_qubits} qubits")
    print(f"  HF energy       : {hf_obj.e_tot:+.10f} Ha")

    # --- FCI baseline ---
    cas_fci = mcscf.CASCI(hf_obj, norb, nelec)
    cas_fci.verbose = 0
    fci_e = cas_fci.kernel()[0]
    print(f"  FCI energy      : {fci_e:+.10f} Ha")

    # --- Statevector VQE ---
    print(f"\n  Running VQE with UCCSD ansatz...\n")
    print(f"  {'Mode':<20s}  {'Energy':>14s}  {'Err (mHa)':>10s}  {'Chem.Acc':>8s}  {'Time':>8s}")
    print("  " + "─" * 66)

    cas_sv = mcscf.CASCI(hf_obj, norb, nelec)
    cas_sv.fcisolver = MaestroSolver(
        ansatz="uccsd",
        backend=backend,
        maxiter=200,
        verbose=False,
    )
    t0 = time.perf_counter()
    sv_e = cas_sv.kernel()[0]
    sv_time = time.perf_counter() - t0
    sv_err = abs(sv_e - fci_e) * 1000
    sv_acc = "✓" if sv_err < 1.6 else "✗"
    print(f"  {'Statevector':<20s}  {sv_e:+14.8f}  {sv_err:10.2f}  {sv_acc:>8s}  {sv_time:7.2f}s")

    # --- MPS VQE ---
    for chi in [4, 16, 64]:
        cas_mps = mcscf.CASCI(hf_obj, norb, nelec)
        cas_mps.fcisolver = MaestroSolver(
            ansatz="uccsd",
            backend=backend,
            simulation="mps",
            mps_bond_dim=chi,
            maxiter=200,
            verbose=False,
        )
        t0 = time.perf_counter()
        mps_e = cas_mps.kernel()[0]
        mps_time = time.perf_counter() - t0
        mps_err = abs(mps_e - fci_e) * 1000
        mps_acc = "✓" if mps_err < 1.6 else "✗"
        print(f"  {'MPS (χ=' + str(chi) + ')':<20s}  {mps_e:+14.8f}  {mps_err:10.2f}  {mps_acc:>8s}  {mps_time:7.2f}s")

    print()
    print("  Key takeaways:")
    print("  • Both modes achieve the same accuracy on small systems")
    print("  • Statevector is limited to ~25 qubits (memory: 2^N amplitudes)")
    print("  • MPS scales to 50-100+ qubits with GPU acceleration")
    print("  • Use statevector for small systems, MPS for large ones")


if __name__ == "__main__":
    main()
