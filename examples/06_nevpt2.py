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
Example 6 — NEVPT2 Perturbation Theory
=======================================

Adds second-order perturbation theory (NEVPT2) on top of a CASSCF
calculation. This recovers dynamic correlation energy that the small
active space misses.

Why NEVPT2?
-----------
CASSCF captures *static* correlation (near-degenerate orbitals, bond
breaking) but misses *dynamic* correlation (short-range electron-electron
cusp). NEVPT2 (N-Electron Valence Perturbation Theory, 2nd order) adds
the dynamic part perturbatively — it's cheap and size-consistent.

The beauty is that NEVPT2 only needs the RDMs from the CASSCF. Since our
QoroSolver already computes these, NEVPT2 comes for free — PySCF does
all the work.

What this example shows
-----------------------
- CASSCF + NEVPT2 energy for H₂O
- Each layer adds accuracy: HF → CASSCF → CASSCF+NEVPT2 → FCI
- NEVPT2 correction is purely classical; only CASSCF uses the GPU

This workflow is publication-quality quantum chemistry.

Usage
-----
    python 07_nevpt2.py
    python 07_nevpt2.py --gpu
"""

import argparse
import time

from pyscf import gto, scf, mcscf, mrpt
from qoro_pyscf import QoroSolver


def main():
    parser = argparse.ArgumentParser(description="CASSCF + NEVPT2")
    parser.add_argument("--gpu", action="store_true", help="Use GPU backend")
    parser.add_argument("--cpu", dest="gpu", action="store_false")
    parser.set_defaults(gpu=False)
    args = parser.parse_args()

    backend = "gpu" if args.gpu else "cpu"

    print("=" * 72)
    print("  NEVPT2 PERTURBATION THEORY — H₂O")
    print(f"  Backend: {backend.upper()}")
    print("=" * 72)

    # --- Molecule ---
    mol = gto.M(
        atom="O 0 0 0; H 0.757 0.587 0; H -0.757 0.587 0",
        basis="sto-3g",
        verbose=0,
    )
    hf_obj = scf.RHF(mol).run()

    norb = 4
    nelec = 4

    print(f"\n  Molecule        : H₂O (STO-3G)")
    print(f"  Active space    : ({nelec}e, {norb}o) → {2 * norb} qubits")
    print(f"  HF energy       : {hf_obj.e_tot:+.10f} Ha")

    # --- Classical FCI reference ---
    cas_fci = mcscf.CASCI(hf_obj, norb, nelec)
    cas_fci.verbose = 0
    fci_e = cas_fci.kernel()[0]

    # --- Step 1: CASSCF with QoroSolver ---
    print(f"\n  Step 1: CASSCF on Qoro ({backend.upper()})...")
    cas = mcscf.CASSCF(hf_obj, norb, nelec)
    cas.fcisolver = QoroSolver(
        ansatz="hardware_efficient",
        ansatz_layers=2,
        backend=backend,
        maxiter=200,
        verbose=False,
    )
    cas.max_cycle_macro = 10
    cas.verbose = 0

    t0 = time.perf_counter()
    casscf_e = cas.kernel()[0]
    casscf_time = time.perf_counter() - t0
    print(f"    CASSCF energy : {casscf_e:+.10f} Ha  ({casscf_time:.1f}s)")

    # --- Step 2: NEVPT2 on top of CASSCF ---
    print(f"  Step 2: NEVPT2 correction (classical)...")
    t0 = time.perf_counter()
    nevpt2_corr = mrpt.NEVPT(cas).kernel()
    nevpt2_time = time.perf_counter() - t0
    nevpt2_e = casscf_e + nevpt2_corr
    print(f"    NEVPT2 corr.  : {nevpt2_corr:+.10f} Ha  ({nevpt2_time:.1f}s)")

    # --- Summary ---
    print(f"\n  {'─' * 50}")
    print(f"  Method            Energy (Ha)        Error to FCI")
    print(f"  {'─' * 50}")
    for label, energy in [
        ("HF", hf_obj.e_tot),
        ("CASSCF (Qoro)", casscf_e),
        ("CASSCF + NEVPT2", nevpt2_e),
        ("FCI (exact)", fci_e),
    ]:
        err = abs(energy - fci_e) * 1000
        mark = "  ◀ GPU" if "Qoro" in label else ""
        print(f"  {label:<20s} {energy:+.10f}  {err:8.2f} mHa{mark}")

    print(f"\n  Each layer adds accuracy:")
    print(f"    HF      → misses all correlation")
    print(f"    CASSCF  → captures static correlation (GPU)")
    print(f"    NEVPT2  → adds dynamic correlation (classical)")


if __name__ == "__main__":
    main()
