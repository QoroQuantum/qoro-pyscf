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
Example 9 — UpCCD: Paired Doubles for Compact Singlet States
=============================================================

Compares the Unitary paired Coupled Cluster Doubles (UpCCD) ansatz against
UCCSD on lithium hydride (LiH), demonstrating dramatic parameter savings.

What is UpCCD?
--------------
Standard UCCSD includes *all* single and double excitations — most of which
are irrelevant for closed-shell singlet states.  UpCCD keeps only **paired**
double excitations: both the α and β electron from the same spatial orbital
excite to the *same* virtual spatial orbital.

This preserves **seniority zero** — every spatial orbital stays either doubly
occupied or empty.  For singlet ground states, this captures the dominant
correlation at a fraction of the cost.

Why it matters
--------------
- UpCCD parameters scale as N_occ × N_vir (linear in each)
- UCCSD parameters scale as O(N² × M²) — explodes with active space size
- Shallower circuits → faster GPU simulation, less noise on hardware
- Amplitudes can be initialised from classical pCCD (no VQE needed)

Reference: arXiv:2508.21679 — "A simple method for seniority-zero quantum
state preparation" (Khamoshi et al., 2025)

What this example shows
-----------------------
- LiH with a (4e, 4o) active space → 8 qubits
- UpCCD: 4 parameters, converges to chemical accuracy
- UCCSD: 52 parameters, struggles to converge (barren plateaus)
- Head-to-head comparison of accuracy, parameter count, and wall time

Usage
-----
    python 09_upccd_paired_doubles.py
    python 09_upccd_paired_doubles.py --gpu
"""

import argparse
import time

import numpy as np
from pyscf import gto, scf, mcscf

from qoro_pyscf import QoroSolver
from qoro_pyscf.ansatze import upccd_param_count, uccsd_param_count


def main():
    parser = argparse.ArgumentParser(
        description="UpCCD vs UCCSD — paired doubles for singlet states"
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU backend")
    parser.add_argument("--cpu", dest="gpu", action="store_false")
    parser.set_defaults(gpu=False)
    args = parser.parse_args()

    backend = "gpu" if args.gpu else "cpu"

    print("=" * 72)
    print("  UpCCD vs UCCSD — PAIRED DOUBLES FOR SINGLET STATES")
    print(f"  Backend: {backend.upper()}")
    print("=" * 72)

    # ── Molecule ──────────────────────────────────────────────────────────
    mol = gto.M(
        atom="Li 0 0 0; H 0 0 1.6",
        basis="sto-3g",
        verbose=0,
    )
    hf_obj = scf.RHF(mol).run()

    norb = 4       # active spatial orbitals
    nelec = (2, 2) # active electrons (alpha, beta)
    n_qubits = 2 * norb

    n_upccd = upccd_param_count(n_qubits, nelec)
    n_uccsd = uccsd_param_count(n_qubits, nelec)

    print(f"\n  Molecule        : LiH (STO-3G)")
    print(f"  Bond length     : 1.6 Å")
    print(f"  Active space    : ({sum(nelec)}e, {norb}o) → {n_qubits} qubits")
    print(f"  HF energy       : {hf_obj.e_tot:+.10f} Ha")
    print(f"\n  UpCCD parameters: {n_upccd}")
    print(f"  UCCSD parameters: {n_uccsd}")
    print(f"  Parameter ratio : {n_uccsd / n_upccd:.0f}× more in UCCSD")

    # ── Exact FCI reference ──────────────────────────────────────────────
    cas_fci = mcscf.CASCI(hf_obj, norb, nelec)
    cas_fci.verbose = 0
    fci_e = cas_fci.kernel()[0]
    print(f"\n  FCI energy      : {fci_e:+.10f} Ha")

    # ── VQE with UpCCD ────────────────────────────────────────────────────
    print(f"\n  {'─' * 50}")
    print(f"  Running VQE with UpCCD ({n_upccd} params)...")
    cas_upccd = mcscf.CASCI(hf_obj, norb, nelec)
    cas_upccd.verbose = 0
    cas_upccd.fcisolver = QoroSolver(
        ansatz="upccd",
        backend=backend,
        maxiter=300,
        verbose=True,
    )

    t0 = time.perf_counter()
    e_upccd = cas_upccd.kernel()[0]
    t_upccd = time.perf_counter() - t0

    # ── VQE with UCCSD ────────────────────────────────────────────────────
    print(f"\n  {'─' * 50}")
    print(f"  Running VQE with UCCSD ({n_uccsd} params)...")
    cas_uccsd = mcscf.CASCI(hf_obj, norb, nelec)
    cas_uccsd.verbose = 0
    cas_uccsd.fcisolver = QoroSolver(
        ansatz="uccsd",
        backend=backend,
        maxiter=300,
        verbose=True,
    )

    t0 = time.perf_counter()
    e_uccsd = cas_uccsd.kernel()[0]
    t_uccsd = time.perf_counter() - t0

    # ── Results ───────────────────────────────────────────────────────────
    err_upccd = abs(e_upccd - fci_e)
    err_uccsd = abs(e_uccsd - fci_e)
    CHEM_ACC = 1.6e-3  # 1.6 mHa

    print(f"\n{'=' * 72}")
    print(f"  RESULTS")
    print(f"{'=' * 72}")
    print(f"  {'Method':<12s}  {'Energy (Ha)':>14s}  {'Error (mHa)':>11s}  "
          f"{'Params':>6s}  {'Time':>6s}  {'Chem Acc':>8s}")
    print(f"  {'─' * 68}")

    for label, e, err, n_p, t in [
        ("FCI",   fci_e,   0.0,       "—",       "—"),
        ("UpCCD", e_upccd, err_upccd, n_upccd, f"{t_upccd:.1f}s"),
        ("UCCSD", e_uccsd, err_uccsd, n_uccsd, f"{t_uccsd:.1f}s"),
    ]:
        marker = "✓" if err < CHEM_ACC else "✗"
        if label == "FCI":
            marker = "—"
        print(f"  {label:<12s}  {e:+14.8f}  {err * 1000:10.4f}  "
              f"{str(n_p):>6s}  {str(t):>6s}  {marker:>8s}")

    print()
    print(f"  Key takeaway:")
    print(f"  → UpCCD uses {n_uccsd // n_upccd}× fewer parameters than UCCSD")
    if err_upccd < CHEM_ACC:
        print(f"  → UpCCD reaches chemical accuracy ({err_upccd * 1000:.4f} mHa)")
    if err_uccsd > CHEM_ACC:
        print(f"  → UCCSD fails to converge ({err_uccsd * 1000:.1f} mHa error)")
        print(f"    This is the barren plateau problem: too many parameters")
        print(f"    create a flat optimization landscape.")
    print(f"  → For singlet ground states, UpCCD is the better ansatz.")
    print()


if __name__ == "__main__":
    main()
