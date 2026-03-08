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
# along with qoro-maestro-pyscf. If not, see <https://www.gnu.org/licenses/\>.

"""
Example 1 — H₂ Bond Dissociation Curve
=======================================

Scans the potential energy surface (PES) of molecular hydrogen by stretching
the H–H bond from 0.4 Å to 2.5 Å.

Why this matters
----------------
The H₂ dissociation curve is *the* classic benchmark for quantum chemistry
methods. At equilibrium (~0.74 Å), single-reference Hartree-Fock (HF) is
reasonable. But as the bond stretches, the electronic wavefunction becomes
strongly correlated — two electrons sharing two orbitals with near-degenerate
energies. HF fails badly here, predicting an energy far too high.

This is exactly the regime where quantum algorithms like VQE shine: even a
small quantum circuit can capture the multi-reference character that breaks
classical mean-field methods.

What this example shows
-----------------------
- PySCF computes molecular integrals at each geometry
- MaestroSolver runs a VQE for each bond length on the GPU
- We compare HF, VQE (Maestro), and exact FCI energies
- At large bond lengths, VQE tracks FCI while HF diverges

Usage
-----
    python 01_h2_dissociation.py          # CPU (default)
    python 01_h2_dissociation.py --gpu    # GPU accelerated
"""

import argparse
import numpy as np

from pyscf import gto, scf, mcscf
from qoro_maestro_pyscf import MaestroSolver


def main():
    parser = argparse.ArgumentParser(description="H₂ bond dissociation curve")
    parser.add_argument("--gpu", action="store_true", help="Use GPU backend")
    parser.add_argument("--cpu", dest="gpu", action="store_false")
    parser.set_defaults(gpu=False)
    args = parser.parse_args()

    backend = "gpu" if args.gpu else "cpu"

    print("=" * 72)
    print("  H₂ BOND DISSOCIATION CURVE")
    print(f"  Backend: {backend.upper()}")
    print("=" * 72)
    print()
    print(f"  {'d (Å)':>8s}  {'HF':>12s}  {'VQE':>12s}  {'FCI':>12s}  {'err (mHa)':>10s}")
    print("  " + "-" * 60)

    bond_lengths = np.arange(0.4, 2.6, 0.2)
    max_error = 0.0
    prev_params = None

    for d in bond_lengths:
        mol = gto.M(atom=f"H 0 0 0; H 0 0 {d:.4f}", basis="sto-3g", verbose=0)
        hf_obj = scf.RHF(mol).run()

        # Exact FCI
        cas_fci = mcscf.CASCI(hf_obj, 2, 2)
        cas_fci.verbose = 0
        fci_e = cas_fci.kernel()[0]

        # VQE on Maestro
        cas_vqe = mcscf.CASCI(hf_obj, 2, 2)
        cas_vqe.fcisolver = MaestroSolver(
            ansatz="hardware_efficient",
            ansatz_layers=3,
            backend=backend,
            maxiter=300,
            initial_point=prev_params,
            verbose=False,
        )
        vqe_e = cas_vqe.kernel()[0]
        prev_params = cas_vqe.fcisolver.optimal_params

        error = abs(vqe_e - fci_e) * 1000  # mHa
        max_error = max(max_error, error)

        print(f"  {d:8.2f}  {hf_obj.e_tot:+12.6f}  {vqe_e:+12.6f}  "
              f"{fci_e:+12.6f}  {error:10.2f}")

    print()
    print(f"  Max error:  {max_error:.2f} mHa")
    chem_acc = "✓ YES" if max_error < 1.6 else "✗ NO"
    print(f"  Chemical accuracy (< 1.6 mHa): {chem_acc}")
    print()
    print("  Note: At large bond lengths, HF diverges from FCI.")
    print("  VQE captures the multi-reference correlation energy.")


if __name__ == "__main__":
    main()
