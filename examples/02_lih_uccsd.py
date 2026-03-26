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
Example 2 — LiH with UCCSD Ansatz
==================================

Computes the ground-state energy of lithium hydride (LiH) using the Unitary
Coupled Cluster Singles and Doubles (UCCSD) ansatz on Qoro.

Why UCCSD?
----------
Hardware-efficient ansatze (random Ry/Rz + CNOTs) are generic — they work for
any problem but aren't physically motivated. UCCSD is different: it's derived
from coupled-cluster theory, which is the gold standard of classical quantum
chemistry.

UCCSD encodes specific *excitation operators* — single excitations (one electron
jumps from an occupied to a virtual orbital) and double excitations (two
electrons jump simultaneously). In the Jordan-Wigner mapping, each excitation
becomes a specific sequence of Pauli rotations and CNOT gates.

The advantage: UCCSD circuits are typically shorter and converge faster than
generic ansatze for chemistry problems, because the circuit structure matches
the physics.

What this example shows
-----------------------
- LiH molecule with STO-3G basis (10 spin-orbitals, but we use a (2e, 2o)
  active space → 4 qubits)
- UCCSD ansatz: chemistry-motivated excitation operators
- Comparison with exact FCI in the same active space

Usage
-----
    python 02_lih_uccsd.py
    python 02_lih_uccsd.py --gpu
"""

import argparse
import time

import numpy as np
from pyscf import gto, scf, mcscf

from qoro_pyscf import QoroSolver


def main():
    parser = argparse.ArgumentParser(description="LiH with UCCSD ansatz")
    parser.add_argument("--gpu", action="store_true", help="Use GPU backend")
    parser.add_argument("--cpu", dest="gpu", action="store_false")
    parser.set_defaults(gpu=False)
    args = parser.parse_args()

    backend = "gpu" if args.gpu else "cpu"

    print("=" * 72)
    print("  LiH — UCCSD ANSATZ")
    print(f"  Backend: {backend.upper()}")
    print("=" * 72)

    # --- Molecule ---
    mol = gto.M(
        atom="Li 0 0 0; H 0 0 1.6",
        basis="sto-3g",
        verbose=0,
    )
    hf_obj = scf.RHF(mol).run()

    norb = 2   # active orbitals
    nelec = 2  # active electrons
    n_qubits = 2 * norb

    print(f"\n  Molecule        : LiH (STO-3G)")
    print(f"  Bond length     : 1.6 Å")
    print(f"  Active space    : ({nelec}e, {norb}o) → {n_qubits} qubits")
    print(f"  HF energy       : {hf_obj.e_tot:+.10f} Ha")

    # --- Exact FCI ---
    cas_fci = mcscf.CASCI(hf_obj, norb, nelec)
    cas_fci.verbose = 0
    fci_e = cas_fci.kernel()[0]
    print(f"  FCI energy      : {fci_e:+.10f} Ha")

    # --- VQE with UCCSD ---
    print(f"\n  Running VQE with UCCSD ansatz...")
    cas_vqe = mcscf.CASCI(hf_obj, norb, nelec)
    cas_vqe.fcisolver = QoroSolver(
        ansatz="uccsd",
        backend=backend,
        maxiter=200,
        verbose=True,
    )

    t0 = time.perf_counter()
    vqe_e = cas_vqe.kernel()[0]
    elapsed = time.perf_counter() - t0

    error = abs(vqe_e - fci_e)

    print(f"\n  {'─' * 50}")
    print(f"  VQE energy (UCCSD) : {vqe_e:+.10f} Ha")
    print(f"  FCI energy (exact) : {fci_e:+.10f} Ha")
    print(f"  Error              : {error:.2e} Ha ({error * 1000:.2f} mHa)")
    print(f"  Chemical accuracy  : {'✓' if error < 1.6e-3 else '✗'}")
    print(f"  Time               : {elapsed:.2f} s")


if __name__ == "__main__":
    main()
