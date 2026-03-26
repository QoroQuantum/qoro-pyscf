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
Example 5 — CASSCF Orbital Optimisation
========================================

Runs CASSCF (Complete Active Space Self-Consistent Field) where PySCF
optimises the molecular orbitals while QoroSolver handles the CI problem
on the Qoro simulator.

Why CASSCF matters
------------------
CASCI uses fixed HF orbitals, which may not be optimal for the correlated
calculation. CASSCF iteratively rotates the orbitals to minimise the total
energy — this is the standard method in production quantum chemistry.

The key requirement is that the solver must provide accurate RDMs (reduced
density matrices), because PySCF uses them to compute the orbital gradient.
Our QoroSolver reconstructs RDMs from the optimised VQE circuit via
Qoro's estimate() function.

CASSCF is what turns a VQE demo into a real chemistry tool.

What this example shows
-----------------------
- CASCI (fixed orbitals) vs CASSCF (optimised orbitals) on LiH
- CASSCF always gives lower or equal energy
- The orbital optimisation loop uses our make_rdm1/make_rdm12 methods

Usage
-----
    python 06_casscf_orbital_opt.py
    python 06_casscf_orbital_opt.py --gpu
"""

import argparse
import time

from pyscf import gto, scf, mcscf
from qoro_pyscf import QoroSolver


def main():
    parser = argparse.ArgumentParser(description="CASSCF orbital optimisation")
    parser.add_argument("--gpu", action="store_true", help="Use GPU backend")
    parser.add_argument("--cpu", dest="gpu", action="store_false")
    parser.set_defaults(gpu=False)
    args = parser.parse_args()

    backend = "gpu" if args.gpu else "cpu"

    print("=" * 72)
    print("  CASSCF ORBITAL OPTIMISATION — LiH")
    print(f"  Backend: {backend.upper()}")
    print("=" * 72)

    mol = gto.M(
        atom="Li 0 0 0; H 0 0 1.6",
        basis="sto-3g",
        verbose=0,
    )
    hf_obj = scf.RHF(mol).run()

    norb = 3   # larger active space for CASSCF
    nelec = 2

    print(f"\n  Molecule        : LiH (STO-3G)")
    print(f"  Active space    : ({nelec}e, {norb}o) → {2 * norb} qubits")
    print(f"  HF energy       : {hf_obj.e_tot:+.10f} Ha")

    # --- FCI baseline ---
    cas_fci = mcscf.CASCI(hf_obj, norb, nelec)
    cas_fci.verbose = 0
    fci_e = cas_fci.kernel()[0]
    print(f"  FCI/CASCI energy: {fci_e:+.10f} Ha")

    # --- CASCI with QoroSolver ---
    print(f"\n  [CASCI] Fixed HF orbitals + VQE...")
    cas_ci = mcscf.CASCI(hf_obj, norb, nelec)
    cas_ci.fcisolver = QoroSolver(
        ansatz="hardware_efficient",
        ansatz_layers=2,
        backend=backend,
        maxiter=200,
        verbose=False,
    )
    t0 = time.perf_counter()
    casci_e = cas_ci.kernel()[0]
    casci_time = time.perf_counter() - t0
    print(f"    Energy  : {casci_e:+.10f} Ha")
    print(f"    Time    : {casci_time:.2f} s")

    # --- CASSCF with QoroSolver ---
    print(f"\n  [CASSCF] Optimised orbitals + VQE...")
    cas_scf = mcscf.CASSCF(hf_obj, norb, nelec)
    cas_scf.fcisolver = QoroSolver(
        ansatz="hardware_efficient",
        ansatz_layers=2,
        backend=backend,
        maxiter=200,
        verbose=False,
    )
    cas_scf.max_cycle_macro = 10  # limit CASSCF macro-iterations
    cas_scf.verbose = 0

    t0 = time.perf_counter()
    casscf_e = cas_scf.kernel()[0]
    casscf_time = time.perf_counter() - t0
    print(f"    Energy  : {casscf_e:+.10f} Ha")
    print(f"    Time    : {casscf_time:.2f} s")

    # --- Summary ---
    improvement = (casci_e - casscf_e) * 1000  # mHa
    print(f"\n  {'─' * 50}")
    print(f"  CASCI energy    : {casci_e:+.10f} Ha")
    print(f"  CASSCF energy   : {casscf_e:+.10f} Ha")
    print(f"  Improvement     : {improvement:.2f} mHa")
    print(f"\n  CASSCF lowers the energy by optimising the orbitals")
    print(f"  around the VQE active space.")


if __name__ == "__main__":
    main()
