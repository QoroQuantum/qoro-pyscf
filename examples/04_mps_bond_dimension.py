#!/usr/bin/env python3
# Copyright 2026 Qoro Quantum Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example 4 — Statevector vs MPS: Bond Dimension Trade-off
==========================================================

Demonstrates when MPS bond dimension χ actually matters by comparing
statevector (exact) and MPS (approximate) simulations on a **12-qubit**
hydrogen chain.

Key insight
-----------
MPS is exact when χ ≥ 2^(N/2). For 12 qubits, that means χ ≥ 64.
Below that threshold, MPS is an approximation — you trade accuracy for
the ability to simulate larger systems (50-100+ qubits on a single GPU).

This example shows:
  • χ = 4   → lossy, noticeably worse than statevector
  • χ = 16  → good approximation, close to statevector
  • χ = 64  → exact, matches statevector

Molecule
--------
H₆ linear chain (STO-3G) at 1.5 Å spacing — a stretched geometry with
significant multi-reference character, making it a challenging test.

Usage
-----
    python 04_mps_bond_dimension.py          # CPU
    python 04_mps_bond_dimension.py --gpu    # GPU
"""

import argparse
import time

import numpy as np
from pyscf import gto, scf, mcscf
from qoro_pyscf import QoroSolver


def main():
    parser = argparse.ArgumentParser(
        description="Statevector vs MPS on H₆ (12 qubits)"
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU backend")
    parser.add_argument("--cpu", dest="gpu", action="store_false")
    parser.set_defaults(gpu=False)
    args = parser.parse_args()

    backend = "gpu" if args.gpu else "cpu"

    print("=" * 72)
    print("  STATEVECTOR vs MPS — BOND DIMENSION TRADE-OFF")
    print(f"  Backend: {backend.upper()}")
    print("=" * 72)

    # --- Molecule: H₆ linear chain ---
    spacing = 1.5  # Å — stretched to enhance correlation
    atoms = "; ".join(f"H 0 0 {i * spacing:.4f}" for i in range(6))
    mol = gto.M(atom=atoms, basis="sto-3g", verbose=0)
    hf_obj = scf.RHF(mol).run()

    norb = 6
    nelec = 6
    n_qubits = 2 * norb

    print(f"\n  Molecule        : H₆ chain (STO-3G, d={spacing} Å)")
    print(f"  Active space    : ({nelec}e, {norb}o) → {n_qubits} qubits")
    print(f"  Max bond dim    : 2^{n_qubits//2} = {2**(n_qubits//2)}")
    print(f"  HF energy       : {hf_obj.e_tot:+.10f} Ha")

    # --- FCI baseline ---
    cas_fci = mcscf.CASCI(hf_obj, norb, nelec)
    cas_fci.verbose = 0
    fci_e = cas_fci.kernel()[0]
    print(f"  FCI energy      : {fci_e:+.10f} Ha")

    # --- VQE runs ---
    n_layers = 3
    maxiter = 100

    print(f"\n  Ansatz: {n_layers}-layer HWE, {maxiter} COBYLA iterations")
    print(f"  (Using same random seed for fair comparison)\n")

    print(f"  {'Mode':<20s}  {'Energy':>14s}  {'Err (mHa)':>10s}  {'vs SV':>10s}  {'Time':>8s}")
    print("  " + "─" * 68)

    # --- Statevector (exact reference) ---
    np.random.seed(42)
    x0 = np.random.default_rng(42).uniform(
        -np.pi / 4, np.pi / 4, size=n_qubits * 2 * n_layers
    )

    cas_sv = mcscf.CASCI(hf_obj, norb, nelec)
    cas_sv.fcisolver = QoroSolver(
        ansatz="hardware_efficient",
        ansatz_layers=n_layers,
        backend=backend,
        maxiter=maxiter,
        initial_point=x0.copy(),
        verbose=False,
    )
    t0 = time.perf_counter()
    sv_e = cas_sv.kernel()[0]
    sv_time = time.perf_counter() - t0
    sv_err = abs(sv_e - fci_e) * 1000
    print(f"  {'Statevector':<20s}  {sv_e:+14.8f}  {sv_err:10.2f}  {'(ref)':>10s}  {sv_time:7.1f}s")

    # --- MPS at various bond dimensions ---
    for chi in [4, 8, 16, 32, 64]:
        cas_mps = mcscf.CASCI(hf_obj, norb, nelec)
        cas_mps.fcisolver = QoroSolver(
            ansatz="hardware_efficient",
            ansatz_layers=n_layers,
            backend=backend,
            simulation="mps",
            mps_bond_dim=chi,
            maxiter=maxiter,
            initial_point=x0.copy(),
            verbose=False,
        )
        t0 = time.perf_counter()
        mps_e = cas_mps.kernel()[0]
        mps_time = time.perf_counter() - t0
        mps_err = abs(mps_e - fci_e) * 1000
        vs_sv = abs(mps_e - sv_e) * 1000  # difference from statevector
        print(f"  {'MPS (χ=' + str(chi) + ')':<20s}  {mps_e:+14.8f}  {mps_err:10.2f}  {vs_sv:10.2f}  {mps_time:7.1f}s")

    print()
    print("  Key takeaways:")
    print(f"  • 12 qubits → max bond dimension is {2**(n_qubits//2)}")
    print("  • χ < max gives approximate results (vs SV column)")
    print("  • χ = max reproduces statevector exactly")
    print("  • For 50+ qubits, statevector is infeasible — MPS is the only option")
    print("  • Qoro runs MPS on GPU for maximum throughput")


if __name__ == "__main__":
    main()
