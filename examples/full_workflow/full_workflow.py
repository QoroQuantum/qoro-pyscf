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
Example 9 — Full Workflow: BeH₂ Insertion Reaction
=====================================================

A complete computational chemistry workflow using qoro-pyscf:
molecule → HF → CASCI + VQE (GPU) → NEVPT2 → properties.

This models the BeH₂ molecule, which has interesting multi-reference
character due to the near-degeneracy of the Be 2s and 2p orbitals.

Why this example?
-----------------
This is what a real computational chemistry study looks like:

1. Define the molecule and basis set
2. Run Hartree-Fock to get starting orbitals
3. CASCI with QoroSolver to capture static correlation
4. NEVPT2 to add dynamic correlation
5. Extract properties (dipole, natural occupations)

It demonstrates that qoro-pyscf isn't just a VQE toy — it's a
full quantum embedding tool that fits into standard chemistry pipelines.

Usage
-----
    python 10_full_workflow.py
    python 10_full_workflow.py --gpu
"""

import argparse
import time

import numpy as np
from pyscf import gto, scf, mcscf, mrpt

from qoro_pyscf import QoroSolver
from qoro_pyscf.properties import (
    compute_dipole_moment,
    compute_natural_orbitals,
)


def main():
    parser = argparse.ArgumentParser(description="Full workflow: BeH₂")
    parser.add_argument("--gpu", action="store_true", help="Use GPU backend")
    parser.add_argument("--cpu", dest="gpu", action="store_false")
    parser.set_defaults(gpu=False)
    args = parser.parse_args()

    backend = "gpu" if args.gpu else "cpu"

    print("=" * 72)
    print("  FULL WORKFLOW — BeH₂")
    print(f"  Backend: {backend.upper()}")
    print("=" * 72)

    t_total = time.perf_counter()

    # ─────────────────────────────────────────────────────────────────────
    # Step 1: Molecule Definition
    # ─────────────────────────────────────────────────────────────────────
    print("\n  ┌─ Step 1: Molecule")
    mol = gto.M(
        atom="""
            Be  0.000  0.000  0.000
            H   0.000  0.000  1.334
            H   0.000  0.000 -1.334
        """,
        basis="sto-3g",
        symmetry=False,
        verbose=0,
    )
    print(f"  │  BeH₂ (STO-3G), linear geometry")
    print(f"  │  Atoms: {mol.natm}, AOs: {mol.nao_nr()}")

    # ─────────────────────────────────────────────────────────────────────
    # Step 2: Hartree-Fock
    # ─────────────────────────────────────────────────────────────────────
    print(f"  │")
    print(f"  ├─ Step 2: Hartree-Fock")
    hf_obj = scf.RHF(mol).run()
    print(f"  │  E(HF) = {hf_obj.e_tot:+.10f} Ha")

    # ─────────────────────────────────────────────────────────────────────
    # Step 3: CASCI with VQE on Qoro
    # ─────────────────────────────────────────────────────────────────────
    norb = 3   # Be 2s, 2pz + H 1s orbitals
    nelec = 2

    print(f"  │")
    print(f"  ├─ Step 3: CASCI + VQE on Qoro ({backend.upper()})")
    print(f"  │  Active space: ({nelec}e, {norb}o) → {2*norb} qubits")

    cas = mcscf.CASCI(hf_obj, norb, nelec)
    cas.fcisolver = QoroSolver(
        ansatz="uccsd",
        backend=backend,
        maxiter=300,
        verbose=False,
    )

    t0 = time.perf_counter()
    casci_e = cas.kernel()[0]
    casci_time = time.perf_counter() - t0
    print(f"  │  E(CASCI+VQE) = {casci_e:+.10f} Ha  ({casci_time:.1f}s)")

    # ─────────────────────────────────────────────────────────────────────
    # Step 4: NEVPT2 Perturbation Theory
    # ─────────────────────────────────────────────────────────────────────
    print(f"  │")
    print(f"  ├─ Step 4: NEVPT2 (classical)")
    t0 = time.perf_counter()
    # NEVPT2 requires a 3-RDM from the CI vector. VQE solvers produce
    # parameterised circuits, not FCI-format CI vectors. When the full
    # NEVPT2 is available (e.g., with a classical CASSCF), it runs;
    # otherwise we skip it and note the limitation.
    try:
        nevpt2_corr = mrpt.NEVPT(cas).kernel()
        nevpt2_e = casci_e + nevpt2_corr
        nevpt2_time = time.perf_counter() - t0
        print(f"  │  ΔE(NEVPT2) = {nevpt2_corr:+.10f} Ha")
        print(f"  │  E(CASCI+NEVPT2) = {nevpt2_e:+.10f} Ha  ({nevpt2_time:.1f}s)")
    except (AssertionError, RuntimeError, TypeError):
        nevpt2_e = None
        print(f"  │  ⚠ NEVPT2 requires a 3-RDM from the CI vector.")
        print(f"  │  VQE solvers produce circuits, not CI vectors.")
        print(f"  │  Skipping — use a classical FCI/DMRG solver for NEVPT2.")

    # ─────────────────────────────────────────────────────────────────────
    # Step 5: Properties
    # ─────────────────────────────────────────────────────────────────────
    print(f"  │")
    print(f"  ├─ Step 5: Properties")

    # RDMs from the VQE circuit
    ci = cas.ci
    rdm1 = cas.fcisolver.make_rdm1(ci, norb, (1, 1))

    # Dipole moment
    dipole, mag = compute_dipole_moment(mol, cas.mo_coeff, rdm1)
    print(f"  │  Dipole: [{dipole[0]:+.4f}, {dipole[1]:+.4f}, {dipole[2]:+.4f}] D")
    print(f"  │  |μ| = {mag:.4f} D  (should be ~0 by symmetry)")

    # Natural orbital occupations
    occ, _ = compute_natural_orbitals(rdm1)
    print(f"  │  Natural occupations: [{', '.join(f'{n:.4f}' for n in occ)}]")

    # ─────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────
    total_time = time.perf_counter() - t_total

    # FCI reference
    cas_fci = mcscf.CASCI(hf_obj, norb, nelec)
    cas_fci.verbose = 0
    fci_e = cas_fci.kernel()[0]

    print(f"  │")
    print(f"  └─ Summary")
    print(f"     {'─' * 52}")
    print(f"     {'Method':<22s}  {'Energy (Ha)':>14s}  {'Δ FCI (mHa)':>12s}")
    print(f"     {'─' * 52}")
    entries = [
        ("HF", hf_obj.e_tot),
        ("CASCI+VQE (Qoro)", casci_e),
    ]
    if nevpt2_e is not None:
        entries.append(("CASCI + NEVPT2", nevpt2_e))
    entries.append(("FCI (exact)", fci_e))
    for label, e in entries:
        err = abs(e - fci_e) * 1000
        print(f"     {label:<22s}  {e:+14.8f}  {err:12.2f}")
    print(f"     {'─' * 52}")
    print(f"     Total time: {total_time:.1f}s")
    print()


if __name__ == "__main__":
    main()
