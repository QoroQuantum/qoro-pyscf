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
Example 7 — Dipole Moment from VQE
====================================

Computes the molecular dipole moment of LiH from the VQE wavefunction.

Why dipole moments?
-------------------
A bare energy number is rarely enough for a chemist. Molecular properties
like dipole moments, polarisabilities, and spectroscopic constants are what
actually connect quantum simulations to experiments.

The dipole moment is a one-electron property — it only needs the 1-RDM:

    μ = Σ_pq D_pq ⟨p|r|q⟩ + nuclear contribution

Since our QoroSolver already computes the 1-RDM from the VQE circuit,
we can compute any one-electron property. PySCF provides the integrals
⟨p|r|q⟩, and we provide D_pq.

What this example shows
-----------------------
- VQE on LiH → optimised circuit → 1-RDM → dipole moment
- Comparison with classical CASCI dipole
- The VQE wavefunction gives physically meaningful observables

Usage
-----
    python 08_dipole_moment.py
    python 08_dipole_moment.py --gpu
"""

import argparse

import numpy as np
from pyscf import gto, scf, mcscf

from qoro_pyscf import QoroSolver
from qoro_pyscf.properties import compute_dipole_moment, compute_natural_orbitals


def main():
    parser = argparse.ArgumentParser(description="Dipole moment from VQE")
    parser.add_argument("--gpu", action="store_true", help="Use GPU backend")
    parser.add_argument("--cpu", dest="gpu", action="store_false")
    parser.set_defaults(gpu=False)
    args = parser.parse_args()

    backend = "gpu" if args.gpu else "cpu"

    print("=" * 72)
    print("  DIPOLE MOMENT FROM VQE — LiH")
    print(f"  Backend: {backend.upper()}")
    print("=" * 72)

    mol = gto.M(
        atom="Li 0 0 0; H 0 0 1.6",
        basis="sto-3g",
        verbose=0,
    )
    hf_obj = scf.RHF(mol).run()

    norb = 2
    nelec = 2

    # --- Classical CASCI reference ---
    cas_fci = mcscf.CASCI(hf_obj, norb, nelec)
    cas_fci.verbose = 0
    fci_e, fcivec = cas_fci.kernel()[:2]
    fci_rdm1 = cas_fci.fcisolver.make_rdm1(fcivec, norb, nelec)
    fci_dipole, fci_mag = compute_dipole_moment(mol, cas_fci.mo_coeff, fci_rdm1)

    # --- VQE with QoroSolver ---
    print(f"\n  Running VQE...")
    cas_vqe = mcscf.CASCI(hf_obj, norb, nelec)
    cas_vqe.fcisolver = QoroSolver(
        ansatz="uccsd",
        backend=backend,
        maxiter=200,
        verbose=False,
    )
    vqe_e, vqe_ci = cas_vqe.kernel()[:2]

    # Get 1-RDM from VQE
    vqe_rdm1 = cas_vqe.fcisolver.make_rdm1(vqe_ci, norb, nelec)
    vqe_dipole, vqe_mag = compute_dipole_moment(mol, cas_vqe.mo_coeff, vqe_rdm1)

    # Natural orbital occupations
    occ, _ = compute_natural_orbitals(vqe_rdm1)

    # --- HF dipole for comparison ---
    hf_dm = hf_obj.make_rdm1()
    AU_TO_DEBYE = 2.541746473
    charges = mol.atom_charges()
    coords = mol.atom_coords()
    nuc_dip = np.einsum("i,ix->x", charges, coords)
    with mol.with_common_orig((0, 0, 0)):
        dip_ints = mol.intor_symmetric("int1e_r", comp=3)
    elec_dip = -np.einsum("xij,ji->x", dip_ints, hf_dm)
    hf_dipole = (nuc_dip + elec_dip) * AU_TO_DEBYE
    hf_mag = np.linalg.norm(hf_dipole)

    # --- Results ---
    print(f"\n  {'─' * 50}")
    print(f"  Energies:")
    print(f"    HF    : {hf_obj.e_tot:+.10f} Ha")
    print(f"    VQE   : {vqe_e:+.10f} Ha")
    print(f"    FCI   : {fci_e:+.10f} Ha")

    print(f"\n  Dipole moments (z-component, Debye):")
    print(f"    HF    : {hf_dipole[2]:+.6f} D  (|μ| = {hf_mag:.6f} D)")
    print(f"    VQE   : {vqe_dipole[2]:+.6f} D  (|μ| = {vqe_mag:.6f} D)")
    print(f"    FCI   : {fci_dipole[2]:+.6f} D  (|μ| = {fci_mag:.6f} D)")

    print(f"\n  Natural orbital occupations (VQE):")
    for i, n in enumerate(occ):
        bar = "█" * int(n * 20)
        print(f"    NO {i}: {n:.4f}  {bar}")
    print(f"\n  Occupations near 1.0 indicate strong correlation.")


if __name__ == "__main__":
    main()
