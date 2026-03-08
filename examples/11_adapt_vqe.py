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
Example 11 — ADAPT-VQE: Adaptive Circuit Growing
==================================================

Demonstrates ADAPT-VQE on LiH, comparing it against fixed UCCSD and UpCCD
ansatze.  ADAPT-VQE grows the circuit one operator at a time, selecting the
operator with the largest energy gradient at each step.

Why ADAPT-VQE?
--------------
Fixed ansatze like UCCSD include *every* excitation operator, even those
that contribute nothing.  This wastes parameters and circuit depth, leading
to barren plateaus in optimisation.

ADAPT-VQE solves this by only including operators that actually lower the
energy.  The result is a compact, molecule-specific circuit that's typically
much shorter than UCCSD but just as accurate.

Reference: Grimsley et al., Nature Communications 10, 3007 (2019)

What this example shows
-----------------------
- LiH with CAS(4,4) = 8 qubits
- ADAPT-VQE automatically selects ~6 operators from a pool of 52
- Comparison with UCCSD (52 params) and UpCCD (4 params)
- All three measured against exact FCI

Usage
-----
    python 11_adapt_vqe.py
    python 11_adapt_vqe.py --gpu
"""

import argparse
import time

import numpy as np
from pyscf import gto, scf, mcscf

from qoro_maestro_pyscf import MaestroSolver
from qoro_maestro_pyscf.ansatze import upccd_param_count, uccsd_param_count


def main():
    parser = argparse.ArgumentParser(description="ADAPT-VQE on LiH")
    parser.add_argument("--gpu", action="store_true", help="Use GPU backend")
    parser.add_argument("--cpu", dest="gpu", action="store_false")
    parser.set_defaults(gpu=False)
    args = parser.parse_args()

    backend = "gpu" if args.gpu else "cpu"

    print("=" * 72)
    print("  ADAPT-VQE — ADAPTIVE CIRCUIT GROWING")
    print(f"  Backend: {backend.upper()}")
    print("=" * 72)

    # ── Molecule ──────────────────────────────────────────────────────────
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    hf_obj = scf.RHF(mol).run()

    norb = 4
    nelec = (2, 2)
    n_qubits = 2 * norb

    print(f"\n  Molecule        : LiH (STO-3G)")
    print(f"  Active space    : ({sum(nelec)}e, {norb}o) → {n_qubits} qubits")
    print(f"  UCCSD params    : {uccsd_param_count(n_qubits, nelec)}")
    print(f"  UpCCD params    : {upccd_param_count(n_qubits, nelec)}")
    print(f"  ADAPT pool size : 52 (singles + doubles)")

    # ── FCI reference ─────────────────────────────────────────────────────
    cas_fci = mcscf.CASCI(hf_obj, norb, nelec)
    cas_fci.verbose = 0
    fci_e = cas_fci.kernel()[0]
    print(f"\n  FCI energy      : {fci_e:+.10f} Ha")

    # ── ADAPT-VQE ─────────────────────────────────────────────────────────
    print(f"\n  {'─' * 50}")
    print(f"  Running ADAPT-VQE...")
    cas_adapt = mcscf.CASCI(hf_obj, norb, nelec)
    cas_adapt.verbose = 0
    cas_adapt.fcisolver = MaestroSolver(
        ansatz="adapt",
        backend=backend,
        adapt_threshold=1e-3,
        adapt_max_ops=30,
        verbose=True,
    )

    t0 = time.perf_counter()
    e_adapt = cas_adapt.kernel()[0]
    t_adapt = time.perf_counter() - t0
    n_adapt_ops = len(cas_adapt.fcisolver.optimal_params)

    # ── UpCCD ─────────────────────────────────────────────────────────────
    print(f"\n  {'─' * 50}")
    n_upccd = upccd_param_count(n_qubits, nelec)
    print(f"  Running UpCCD ({n_upccd} params)...")
    cas_upccd = mcscf.CASCI(hf_obj, norb, nelec)
    cas_upccd.verbose = 0
    cas_upccd.fcisolver = MaestroSolver(
        ansatz="upccd", backend=backend, verbose=False,
    )

    t0 = time.perf_counter()
    e_upccd = cas_upccd.kernel()[0]
    t_upccd = time.perf_counter() - t0

    # ── Results ───────────────────────────────────────────────────────────
    err_adapt = abs(e_adapt - fci_e)
    err_upccd = abs(e_upccd - fci_e)
    CHEM_ACC = 1.6e-3

    print(f"\n{'=' * 72}")
    print(f"  RESULTS")
    print(f"{'=' * 72}")
    print(f"  {'Method':<12s}  {'Energy (Ha)':>14s}  {'Error (mHa)':>11s}  "
          f"{'Params':>6s}  {'Time':>6s}  {'Chem Acc':>8s}")
    print(f"  {'─' * 68}")

    for label, e, err, n_p, t in [
        ("FCI",      fci_e,   0.0,       "—",        "—"),
        ("ADAPT-VQE", e_adapt, err_adapt, n_adapt_ops, f"{t_adapt:.1f}s"),
        ("UpCCD",    e_upccd, err_upccd, n_upccd,    f"{t_upccd:.1f}s"),
    ]:
        marker = "✓" if err < CHEM_ACC else "✗"
        if label == "FCI":
            marker = "—"
        print(f"  {label:<12s}  {e:+14.8f}  {err * 1000:10.4f}  "
              f"{str(n_p):>6s}  {str(t):>6s}  {marker:>8s}")

    print()
    print(f"  Key takeaway:")
    print(f"  → ADAPT-VQE used {n_adapt_ops} operators (from a pool of 52)")
    print(f"  → It automatically found which excitations matter for LiH")
    if err_adapt < CHEM_ACC:
        print(f"  → Reached chemical accuracy ({err_adapt * 1000:.4f} mHa)")
    print()


if __name__ == "__main__":
    main()
