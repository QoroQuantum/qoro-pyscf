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
Hamiltonian construction: PySCF integrals → OpenFermion QubitOperator.

Converts one- and two-electron integrals (as delivered by PySCF's CASCI/CASSCF
``kernel`` call) into a qubit Hamiltonian via the Jordan-Wigner transformation.
The resulting Pauli terms are formatted for Qoro's ``estimate()`` API.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from pyscf import ao2mo

from openfermion import InteractionOperator, QubitOperator, jordan_wigner


# Type aliases matching PySCF's calling convention
OneBodyIntegrals = Union[np.ndarray, tuple[np.ndarray, np.ndarray]]
TwoBodyIntegrals = Union[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]


def _restore_eri(eri: np.ndarray, norb: int) -> np.ndarray:
    """
    Restore two-electron integrals to a full 4D tensor.

    PySCF's CASCI/CASSCF kernel passes ERIs in compressed formats:
      - 1D array of length norb*(norb+1)/2 * (norb*(norb+1)/2+1)/2 (8-fold)
      - 2D array of shape (norb*(norb+1)//2, norb*(norb+1)//2) (4-fold)
      - Already 4D (norb, norb, norb, norb)

    We always need the full (norb, norb, norb, norb) tensor.
    """
    if eri.ndim == 4:
        return eri
    # ao2mo.restore(1, ...) restores any compressed format to full 4D
    return ao2mo.restore(1, eri, norb)


def integrals_to_qubit_hamiltonian(
    h1: OneBodyIntegrals,
    h2: TwoBodyIntegrals,
    norb: int,
) -> tuple[QubitOperator, float]:
    """
    Convert PySCF one- and two-electron integrals to a qubit Hamiltonian.

    Supports both RHF (single arrays) and UHF (tuple of spin-resolved arrays)
    integral formats, matching the calling convention of PySCF's
    ``fcisolver.kernel``.

    Parameters
    ----------
    h1 : ndarray or (ndarray, ndarray)
        One-electron integrals. Single array for RHF; tuple (h1_a, h1_b)
        for UHF.
    h2 : ndarray or (ndarray, ndarray, ndarray)
        Two-electron integrals in chemist's notation (possibly compressed).
        Single array for RHF; tuple (h2_aa, h2_ab, h2_bb) for UHF.
    norb : int
        Number of spatial orbitals.

    Returns
    -------
    qubit_op : QubitOperator
        Qubit Hamiltonian (Jordan-Wigner mapped).
    identity_offset : float
        Coefficient of the identity term (energy offset).
    """
    n_qubits = 2 * norb

    # --- Unpack integrals ---
    if isinstance(h1, tuple):
        h1_a, h1_b = h1
    else:
        h1_a = h1
        h1_b = h1  # RHF: alpha = beta

    if isinstance(h2, tuple):
        h2_aa = _restore_eri(h2[0], norb)
        h2_ab = _restore_eri(h2[1], norb)
        h2_bb = _restore_eri(h2[2], norb)
    else:
        h2_aa = _restore_eri(h2, norb)
        h2_ab = h2_aa  # RHF: all spin blocks identical
        h2_bb = h2_aa

    # --- Build spin-orbital one-body tensor ---
    one_body = np.zeros((n_qubits, n_qubits))
    for p in range(norb):
        for q in range(norb):
            one_body[2 * p, 2 * q] = h1_a[p, q]          # α-α
            one_body[2 * p + 1, 2 * q + 1] = h1_b[p, q]  # β-β

    # --- Build spin-orbital two-body tensor ---
    # PySCF delivers chemist notation: h2[p,q,r,s] = (pq|rs).
    # OpenFermion's InteractionOperator uses physicist convention:
    #   H = Σ_{pq} h1[p,q] a†_p a_q
    #     + Σ_{pqrs} two_body[p,q,r,s] a†_p a†_q a_s a_r
    #
    # The correct mapping from chemist to InteractionOperator is:
    #   two_body[p,q,r,s] = 0.5 * h2_chem[p,s,r,q]
    #
    # Verified against openfermionpyscf's MolecularData output.
    two_body = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))

    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    # αα
                    two_body[2*p, 2*q, 2*r, 2*s] = 0.5 * h2_aa[p, s, r, q]
                    # ββ
                    two_body[2*p+1, 2*q+1, 2*r+1, 2*s+1] = 0.5 * h2_bb[p, s, r, q]
                    # αβ
                    two_body[2*p, 2*q+1, 2*r+1, 2*s] = 0.5 * h2_ab[p, s, r, q]
                    # βα
                    two_body[2*p+1, 2*q, 2*r, 2*s+1] = 0.5 * h2_ab[r, q, p, s]

    # --- Build InteractionOperator and apply Jordan-Wigner ---
    iop = InteractionOperator(
        constant=0.0,
        one_body_tensor=one_body,
        two_body_tensor=two_body,
    )
    qubit_op = jordan_wigner(iop)

    # --- Extract identity offset ---
    identity_offset = 0.0
    if () in qubit_op.terms:
        identity_offset = qubit_op.terms[()].real

    return qubit_op, identity_offset


def qubit_op_to_pauli_list(
    qubit_op: QubitOperator,
    n_qubits: int,
) -> tuple[float, list[str], np.ndarray]:
    """
    Convert an OpenFermion QubitOperator into Qoro-compatible observables.

    Parameters
    ----------
    qubit_op : QubitOperator
        The qubit Hamiltonian.
    n_qubits : int
        Number of qubits (spin-orbitals).

    Returns
    -------
    identity_coeff : float
        Coefficient of the all-identity term (classical energy offset).
    pauli_labels : list[str]
        Pauli strings for Qoro, e.g. ``["XZIY", "ZZII"]``.
    pauli_coeffs : np.ndarray
        Complex coefficients for each Pauli term.
    """
    identity_coeff = 0.0
    pauli_labels = []
    pauli_coeffs = []

    for term, coeff in qubit_op.terms.items():
        if not term:
            identity_coeff = coeff.real
            continue

        label = ["I"] * n_qubits
        for qubit_idx, pauli_char in term:
            label[qubit_idx] = pauli_char

        pauli_labels.append("".join(label))
        pauli_coeffs.append(coeff)

    return identity_coeff, pauli_labels, np.array(pauli_coeffs, dtype=complex)
