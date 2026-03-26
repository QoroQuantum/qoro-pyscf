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
Reduced Density Matrix (RDM) reconstruction from Qoro circuits.

Reconstructs 1-RDM and 2-RDM from the optimised VQE circuit by measuring
the Jordan-Wigner-mapped creation/annihilation operator pairs via Qoro's
expectation value estimation.

This is the critical module that enables CASSCF orbital optimisation — PySCF
needs proper density matrices to update the orbitals.

Background
----------
In second quantisation, the 1-RDM and 2-RDM elements are:

    γ_{pq}     = ⟨ψ| a†_p a_q |ψ⟩
    Γ_{pqrs}   = ⟨ψ| a†_p a†_r a_s a_q |ψ⟩

Under the Jordan-Wigner transformation, each ``a†_p a_q`` maps to a sum of
Pauli strings (typically 2 terms for p≠q, 1 term for p=q). We evaluate these
expectation values on the Qoro circuit and reconstruct the RDM tensors.
"""

from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from maestro.circuits import QuantumCircuit

from openfermion import FermionOperator, jordan_wigner

from qoro_pyscf.backends import BackendConfig
from qoro_pyscf.expectation import evaluate_expectation


# ──────────────────────────────────────────────────────────────────────────────
# 1-RDM
# ──────────────────────────────────────────────────────────────────────────────

def compute_1rdm_spinorbital(
    circuit: QuantumCircuit,
    n_qubits: int,
    config: BackendConfig,
) -> np.ndarray:
    """
    Compute the full spin-orbital 1-RDM: γ_{pq} = ⟨a†_p a_q⟩.

    Parameters
    ----------
    circuit : QuantumCircuit
        The optimised VQE circuit.
    n_qubits : int
        Number of spin-orbitals (qubits).
    config : BackendConfig
        Qoro backend configuration.

    Returns
    -------
    rdm1 : np.ndarray, shape (n_qubits, n_qubits)
        Spin-orbital 1-RDM.
    """
    rdm1 = np.zeros((n_qubits, n_qubits), dtype=complex)

    for p in range(n_qubits):
        for q in range(n_qubits):
            # Build the fermion operator a†_p a_q
            fop = FermionOperator(f"{p}^ {q}")
            # Map to qubits via Jordan-Wigner
            qop = jordan_wigner(fop)

            # Evaluate each Pauli term
            val = _evaluate_qubit_operator(circuit, qop, n_qubits, config)
            rdm1[p, q] = val

    return rdm1


def compute_1rdm_spatial(
    circuit: QuantumCircuit,
    n_qubits: int,
    config: BackendConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute spin-resolved spatial 1-RDMs for PySCF.

    Returns
    -------
    rdm1_a : np.ndarray, shape (norb, norb)
        Alpha-spin 1-RDM.
    rdm1_b : np.ndarray, shape (norb, norb)
        Beta-spin 1-RDM.
    """
    norb = n_qubits // 2
    rdm1_so = compute_1rdm_spinorbital(circuit, n_qubits, config)

    # Extract alpha (even indices) and beta (odd indices) blocks
    rdm1_a = np.zeros((norb, norb))
    rdm1_b = np.zeros((norb, norb))

    for p in range(norb):
        for q in range(norb):
            rdm1_a[p, q] = rdm1_so[2 * p, 2 * q].real
            rdm1_b[p, q] = rdm1_so[2 * p + 1, 2 * q + 1].real

    return rdm1_a, rdm1_b


# ──────────────────────────────────────────────────────────────────────────────
# 2-RDM
# ──────────────────────────────────────────────────────────────────────────────

def compute_2rdm_spinorbital(
    circuit: QuantumCircuit,
    n_qubits: int,
    config: BackendConfig,
) -> np.ndarray:
    """
    Compute the full spin-orbital 2-RDM: Γ_{pqrs} = ⟨a†_p a†_r a_s a_q⟩.

    This is the most expensive operation — O(n⁴) Pauli measurements.
    For larger systems, consider approximate methods.

    Parameters
    ----------
    circuit : QuantumCircuit
        The optimised VQE circuit.
    n_qubits : int
        Number of spin-orbitals (qubits).
    config : BackendConfig
        Qoro backend configuration.

    Returns
    -------
    rdm2 : np.ndarray, shape (n_qubits, n_qubits, n_qubits, n_qubits)
        Spin-orbital 2-RDM.
    """
    rdm2 = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits), dtype=complex)

    for p in range(n_qubits):
        for q in range(n_qubits):
            for r in range(n_qubits):
                for s in range(n_qubits):
                    # Build a†_p a†_r a_s a_q
                    fop = FermionOperator(f"{p}^ {r}^ {s} {q}")
                    qop = jordan_wigner(fop)
                    val = _evaluate_qubit_operator(circuit, qop, n_qubits, config)
                    rdm2[p, q, r, s] = val

    return rdm2


def compute_2rdm_spatial(
    circuit: QuantumCircuit,
    n_qubits: int,
    config: BackendConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spin-resolved spatial 2-RDMs in chemist index ordering for PySCF.

    Returns
    -------
    rdm2_aa : np.ndarray, shape (norb, norb, norb, norb)
        Alpha-alpha 2-RDM.
    rdm2_ab : np.ndarray, shape (norb, norb, norb, norb)
        Alpha-beta 2-RDM.
    rdm2_bb : np.ndarray, shape (norb, norb, norb, norb)
        Beta-beta 2-RDM.
    """
    norb = n_qubits // 2
    rdm2_so = compute_2rdm_spinorbital(circuit, n_qubits, config)

    rdm2_aa = np.zeros((norb, norb, norb, norb))
    rdm2_ab = np.zeros((norb, norb, norb, norb))
    rdm2_bb = np.zeros((norb, norb, norb, norb))

    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    # αα: even indices
                    rdm2_aa[p, q, r, s] = rdm2_so[
                        2*p, 2*q, 2*r, 2*s
                    ].real
                    # ββ: odd indices
                    rdm2_bb[p, q, r, s] = rdm2_so[
                        2*p+1, 2*q+1, 2*r+1, 2*s+1
                    ].real
                    # αβ: mixed
                    rdm2_ab[p, q, r, s] = rdm2_so[
                        2*p, 2*q+1, 2*r+1, 2*s
                    ].real

    return rdm2_aa, rdm2_ab, rdm2_bb


# ──────────────────────────────────────────────────────────────────────────────
# Spin tracing
# ──────────────────────────────────────────────────────────────────────────────

def trace_spin_rdm1(
    rdm1_a: np.ndarray, rdm1_b: np.ndarray
) -> np.ndarray:
    """Compute spin-traced 1-RDM: rdm1 = rdm1_a + rdm1_b."""
    return rdm1_a + rdm1_b


def trace_spin_rdm2(
    rdm2_aa: np.ndarray, rdm2_ab: np.ndarray, rdm2_bb: np.ndarray
) -> np.ndarray:
    """
    Compute spin-traced 2-RDM in chemist ordering.

    rdm2 = rdm2_aa + rdm2_ab + rdm2_ab^T + rdm2_bb

    (where rdm2_ab^T transposes the spin indices, i.e. rdm2_ba[p,q,r,s] =
    rdm2_ab[r,s,p,q]).
    """
    rdm2_ba = rdm2_ab.transpose(2, 3, 0, 1)
    return rdm2_aa + rdm2_ab + rdm2_ba + rdm2_bb


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _evaluate_qubit_operator(
    circuit: QuantumCircuit,
    qubit_op,
    n_qubits: int,
    config: BackendConfig,
) -> complex:
    """
    Evaluate a single OpenFermion QubitOperator on the circuit.

    Decomposes into Pauli terms and evaluates each via Qoro.
    """
    result = 0.0 + 0.0j

    # Collect non-identity terms
    pauli_labels = []
    pauli_coeffs = []
    identity_coeff = 0.0 + 0.0j

    for term, coeff in qubit_op.terms.items():
        if not term:
            identity_coeff = coeff
            continue
        label = ["I"] * n_qubits
        for qi, pc in term:
            label[qi] = pc
        pauli_labels.append("".join(label))
        pauli_coeffs.append(coeff)

    # Identity contribution
    result += identity_coeff

    # Pauli contributions (batched)
    if pauli_labels:
        exp_vals = evaluate_expectation(circuit, pauli_labels, config)
        for coeff, ev in zip(pauli_coeffs, exp_vals):
            result += coeff * ev

    return result
