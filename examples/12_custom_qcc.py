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
Example 12 — Custom Ansatz: Qubit Coupled Cluster (QCC)
========================================================

Demonstrates the ``ansatz="custom"`` feature with a Qubit Coupled Cluster
(QCC) circuit injected into QoroSolver.

What is QCC?
------------
Qubit Coupled Cluster (QCC) works directly in the qubit space rather than
mapping from fermionic operators.  It builds entanglers from **multi-qubit
Pauli rotations** of the form:

    exp(-i θ/2  P),   P ∈ {I, X, Y, Z}^⊗n

These "Pauli word" generators are chosen to maximally lower the energy,
analogous to ADAPT-VQE but operating in the qubit picture.  QCC circuits
are naturally compact and GPU-friendly.

Reference: Ryabinkin, Yen, Genin & Izmaylov,
J. Chem. Theory Comput. 14, 6317 (2018).

How this example works
-----------------------
1. We define a ``qcc_ansatz_generator`` callable that builds a Qoro
   QuantumCircuit from a set of Pauli-word entanglers and variational
   parameters.
2. The callable is injected via ``custom_ansatz=qcc_ansatz_generator``.
3. QoroSolver calls it every VQE iteration with updated parameters,
   enabling the full iterative QCC workflow.
4. We compare QCC against UCCSD and FCI on H₂.

Usage
-----
    python 12_custom_qcc.py
    python 12_custom_qcc.py --gpu
"""

import argparse
import time
from typing import TYPE_CHECKING

import numpy as np
from pyscf import gto, scf, mcscf

from qoro_pyscf import QoroSolver
from qoro_pyscf.ansatze import (
    _QC,
    _apply_hf_gates,
    uccsd_param_count,
)

if TYPE_CHECKING:
    from maestro.circuits import QuantumCircuit


# ──────────────────────────────────────────────────────────────────────────────
# QCC Ansatz Builder
# ──────────────────────────────────────────────────────────────────────────────

def _apply_pauli_rotation(
    qc: "QuantumCircuit",
    pauli_word: str,
    theta: float,
) -> None:
    """
    Apply exp(-i θ/2 P) for a multi-qubit Pauli word P.

    Uses the standard basis-change + CNOT-cascade + Rz decomposition:
      1. Rotate each qubit into the eigenbasis of its Pauli operator
         (H for X, Rx(π/2) for Y, nothing for Z, skip I).
      2. CNOT cascade to compute the parity onto a single target qubit.
      3. Rz(θ) on the target qubit.
      4. Undo steps 2 and 1.
    """
    n = len(pauli_word)

    # Identify active qubits (non-identity)
    active = [(q, p) for q, p in enumerate(pauli_word) if p != "I"]
    if not active:
        return  # all-identity → global phase, skip

    # Step 1: Basis change
    for q, p in active:
        if p == "X":
            qc.h(q)
        elif p == "Y":
            qc.rx(q, np.pi / 2)
        # Z: no rotation needed

    # Step 2: CNOT cascade (compute parity onto last active qubit)
    active_qubits = [q for q, _ in active]
    for i in range(len(active_qubits) - 1):
        qc.cx(active_qubits[i], active_qubits[i + 1])

    # Step 3: Rz rotation on the target qubit
    target = active_qubits[-1]
    qc.rz(target, theta)

    # Step 4: Undo CNOT cascade
    for i in range(len(active_qubits) - 2, -1, -1):
        qc.cx(active_qubits[i], active_qubits[i + 1])

    # Step 5: Undo basis change
    for q, p in active:
        if p == "X":
            qc.h(q)
        elif p == "Y":
            qc.rx(q, -np.pi / 2)


def make_qcc_ansatz_generator(
    entanglers: list[str],
    nelec: tuple[int, int] | int,
):
    """
    Create a QCC ansatz generator callable for QoroSolver.

    Parameters
    ----------
    entanglers : list of str
        Pauli words defining the QCC entanglers, e.g. ["XYXY", "YXYX"].
        Each entangler becomes a parameterised Pauli rotation.
    nelec : int or (int, int)
        Electron count for the Hartree-Fock reference state.

    Returns
    -------
    generator : callable
        Signature: ``(params, n_qubits, nelec) → QuantumCircuit``.
    n_params : int
        Number of variational parameters (= len(entanglers)).
    """

    def qcc_ansatz(
        params: np.ndarray,
        n_qubits: int,
        nelec_: tuple[int, int],
    ) -> "QuantumCircuit":
        """Build QCC circuit: |HF⟩ + Π exp(-i θ_k/2 P_k)."""
        qc = _QC()

        # 1. Hartree-Fock reference state
        _apply_hf_gates(qc, n_qubits, nelec_)

        # 2. Apply each QCC entangler as a parameterised Pauli rotation
        for k, word in enumerate(entanglers):
            _apply_pauli_rotation(qc, word, float(params[k]))

        return qc

    return qcc_ansatz, len(entanglers)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QCC custom ansatz on H₂")
    parser.add_argument("--gpu", action="store_true", help="Use GPU backend")
    parser.add_argument("--cpu", dest="gpu", action="store_false")
    parser.set_defaults(gpu=False)
    args = parser.parse_args()

    backend = "gpu" if args.gpu else "cpu"

    print("=" * 72)
    print("  CUSTOM ANSATZ — QUBIT COUPLED CLUSTER (QCC)")
    print(f"  Backend: {backend.upper()}")
    print("=" * 72)

    # ── Molecule ──────────────────────────────────────────────────────────
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    hf_obj = scf.RHF(mol).run()

    norb = 2
    nelec = (1, 1)
    n_qubits = 2 * norb

    print(f"\n  Molecule        : H₂ (STO-3G, r=0.74 Å)")
    print(f"  Active space    : ({sum(nelec)}e, {norb}o) → {n_qubits} qubits")
    print(f"  UCCSD params    : {uccsd_param_count(n_qubits, nelec)}")

    # ── Define QCC entanglers ─────────────────────────────────────────────
    # For H₂ in the STO-3G basis, the dominant correlation comes from the
    # double excitation |01⟩ ↔ |10⟩.  In the qubit picture, this maps to
    # Pauli words that generate rotations in the relevant subspace.
    #
    # These entanglers are chosen following the QCC prescription:
    # screen Pauli words by their energy gradient magnitude and keep the
    # largest contributors.
    qcc_entanglers = [
        "XYXY",  # dominant double excitation analogue
        "YXYX",  # conjugate term
        "XXYY",  # additional correlation
    ]

    qcc_generator, n_qcc_params = make_qcc_ansatz_generator(
        qcc_entanglers, nelec
    )

    print(f"  QCC entanglers  : {len(qcc_entanglers)}")
    for ent in qcc_entanglers:
        print(f"    → {ent}")
    print(f"  QCC params      : {n_qcc_params}")

    # ── FCI reference ─────────────────────────────────────────────────────
    cas_fci = mcscf.CASCI(hf_obj, norb, nelec)
    cas_fci.verbose = 0
    fci_e = cas_fci.kernel()[0]
    print(f"\n  FCI energy      : {fci_e:+.10f} Ha")

    # ── VQE with QCC (custom ansatz) ──────────────────────────────────────
    print(f"\n  {'─' * 50}")
    print(f"  Running VQE with QCC ({n_qcc_params} params)...")
    cas_qcc = mcscf.CASCI(hf_obj, norb, nelec)
    cas_qcc.verbose = 0
    cas_qcc.fcisolver = QoroSolver(
        ansatz="custom",
        custom_ansatz=qcc_generator,
        custom_ansatz_n_params=n_qcc_params,
        backend=backend,
        maxiter=200,
        verbose=True,
    )

    t0 = time.perf_counter()
    e_qcc = cas_qcc.kernel()[0]
    t_qcc = time.perf_counter() - t0

    # ── VQE with UCCSD ────────────────────────────────────────────────────
    n_uccsd = uccsd_param_count(n_qubits, nelec)
    print(f"\n  {'─' * 50}")
    print(f"  Running VQE with UCCSD ({n_uccsd} params)...")
    cas_uccsd = mcscf.CASCI(hf_obj, norb, nelec)
    cas_uccsd.verbose = 0
    cas_uccsd.fcisolver = QoroSolver(
        ansatz="uccsd",
        backend=backend,
        maxiter=200,
        verbose=True,
    )

    t0 = time.perf_counter()
    e_uccsd = cas_uccsd.kernel()[0]
    t_uccsd = time.perf_counter() - t0

    # ── Results ───────────────────────────────────────────────────────────
    err_qcc = abs(e_qcc - fci_e)
    err_uccsd = abs(e_uccsd - fci_e)
    CHEM_ACC = 1.6e-3

    print(f"\n{'=' * 72}")
    print(f"  RESULTS")
    print(f"{'=' * 72}")
    print(f"  {'Method':<12s}  {'Energy (Ha)':>14s}  {'Error (mHa)':>11s}  "
          f"{'Params':>6s}  {'Time':>6s}  {'Chem Acc':>8s}")
    print(f"  {'─' * 68}")

    for label, e, err, n_p, t in [
        ("FCI",   fci_e,   0.0,      "—",          "—"),
        ("QCC",   e_qcc,   err_qcc,  n_qcc_params, f"{t_qcc:.1f}s"),
        ("UCCSD", e_uccsd, err_uccsd, n_uccsd,     f"{t_uccsd:.1f}s"),
    ]:
        marker = "✓" if err < CHEM_ACC else "✗"
        if label == "FCI":
            marker = "—"
        print(f"  {label:<12s}  {e:+14.8f}  {err * 1000:10.4f}  "
              f"{str(n_p):>6s}  {str(t):>6s}  {marker:>8s}")

    print()
    print(f"  Key takeaway:")
    print(f"  → QCC uses only {n_qcc_params} Pauli-word entanglers (vs {n_uccsd} UCCSD params)")
    print(f"  → Custom ansatz callable was called every VQE iteration")
    if err_qcc < CHEM_ACC:
        print(f"  → QCC reached chemical accuracy ({err_qcc * 1000:.4f} mHa)")
    print(f"  → This enables iterative QCC workflows where entanglers are")
    print(f"    updated between macro-iterations.")
    print()


if __name__ == "__main__":
    main()
