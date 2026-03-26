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
- QoroSolver runs a VQE for each bond length
- We compare HF, VQE (Qoro), and exact FCI energies
- At large bond lengths, VQE tracks FCI while HF diverges

Usage
-----
    python 01_h2_dissociation.py          # CPU (default)
    python 01_h2_dissociation.py --gpu    # GPU (faster, requires license)
"""

import argparse
import numpy as np

from pyscf import gto, scf, mcscf
from qoro_pyscf import QoroSolver


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

    for d in bond_lengths:
        mol = gto.M(atom=f"H 0 0 0; H 0 0 {d:.4f}", basis="sto-3g", verbose=0)
        hf_obj = scf.RHF(mol).run()

        # Exact FCI
        cas_fci = mcscf.CASCI(hf_obj, 2, 2)
        cas_fci.verbose = 0
        fci_e = cas_fci.kernel()[0]

        # VQE on Qoro
        cas_vqe = mcscf.CASCI(hf_obj, 2, 2)
        cas_vqe.fcisolver = QoroSolver(
            ansatz="uccsd",
            backend=backend,
            maxiter=300,
            verbose=False,
        )
        vqe_e = cas_vqe.kernel()[0]

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
