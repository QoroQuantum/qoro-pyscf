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
Z₂ symmetry-based qubit tapering.

Reduces the number of qubits by exploiting particle-number and spin-parity
conservation symmetries.  For a typical molecular Hamiltonian in the
Jordan-Wigner encoding, this saves 2–3 qubits (one for each independent
Z₂ symmetry: total N parity, S_z parity, and sometimes spatial symmetry).

Uses OpenFermion's ``taper_off_qubits`` under the hood.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from openfermion import QubitOperator, taper_off_qubits

logger = logging.getLogger(__name__)


@dataclass
class TaperingResult:
    """Result of qubit tapering.

    Attributes
    ----------
    tapered_op : QubitOperator
        The Hamiltonian with reduced qubit count.
    original_n_qubits : int
        Number of qubits before tapering.
    tapered_n_qubits : int
        Number of qubits after tapering.
    removed_positions : list[int]
        Indices of qubits that were removed.
    stabilizers : list[QubitOperator]
        The Z₂ stabilizer generators used.
    """
    tapered_op: QubitOperator
    original_n_qubits: int
    tapered_n_qubits: int
    removed_positions: list[int]
    stabilizers: list[QubitOperator]


def find_z2_symmetries(
    n_qubits: int,
    nelec: tuple[int, int],
) -> list[QubitOperator]:
    """
    Build Z₂ stabilizer generators from particle-number and spin symmetries.

    For a Jordan-Wigner encoded molecular Hamiltonian, the following
    symmetries are always present:

    1. **Total electron parity**: Z on all qubits (eigenvalue ±1 based
       on even/odd N).
    2. **Alpha-spin parity**: Z on all even-indexed qubits.
    3. **Beta-spin parity**: Z on all odd-indexed qubits.

    Parameters
    ----------
    n_qubits : int
        Total number of qubits (spin-orbitals).
    nelec : (int, int)
        Number of (alpha, beta) electrons.

    Returns
    -------
    stabilizers : list of QubitOperator
        Independent Z₂ stabilizers. Typically 2 generators for molecular
        systems (alpha parity and beta parity; total parity is their product).
    """
    n_alpha, n_beta = nelec

    # Alpha-spin parity: Z on all even qubits
    alpha_parity = QubitOperator(())  # start with identity
    for q in range(0, n_qubits, 2):
        alpha_parity *= QubitOperator(f"Z{q}")

    # Determine eigenvalue: (-1)^{n_alpha}
    alpha_sign = (-1) ** n_alpha
    if alpha_sign < 0:
        alpha_parity = -alpha_parity

    # Beta-spin parity: Z on all odd qubits
    beta_parity = QubitOperator(())
    for q in range(1, n_qubits, 2):
        beta_parity *= QubitOperator(f"Z{q}")

    beta_sign = (-1) ** n_beta
    if beta_sign < 0:
        beta_parity = -beta_parity

    return [alpha_parity, beta_parity]


def taper_hamiltonian(
    qubit_op: QubitOperator,
    n_qubits: int,
    nelec: tuple[int, int],
    stabilizers: Optional[list[QubitOperator]] = None,
) -> TaperingResult:
    """
    Taper a qubit Hamiltonian using Z₂ symmetries.

    Reduces the qubit count by eliminating qubits whose state is
    fully determined by symmetry constraints.

    Parameters
    ----------
    qubit_op : QubitOperator
        The full (un-tapered) qubit Hamiltonian.
    n_qubits : int
        Number of qubits in the original Hamiltonian.
    nelec : (int, int)
        Number of (alpha, beta) electrons.
    stabilizers : list of QubitOperator or None
        Custom stabilizer generators. If None, uses the default
        particle-number parity stabilizers from :func:`find_z2_symmetries`.

    Returns
    -------
    TaperingResult
        Contains the tapered Hamiltonian, qubit counts, removed
        positions, and stabilizers used.

    Examples
    --------
    >>> from openfermion import QubitOperator
    >>> from qoro_pyscf.tapering import taper_hamiltonian
    >>> H = QubitOperator("Z0 Z1", 0.5) + QubitOperator("X0 X1", 0.25)
    >>> result = taper_hamiltonian(H, n_qubits=4, nelec=(1, 1))
    >>> result.tapered_n_qubits < 4
    True
    """
    if stabilizers is None:
        stabilizers = find_z2_symmetries(n_qubits, nelec)

    # OpenFermion's taper_off_qubits handles the heavy lifting
    tapered_op, removed = taper_off_qubits(
        qubit_op,
        stabilizers,
        output_tapered_positions=True,
    )

    tapered_n = n_qubits - len(removed)

    logger.info(
        "Tapering: %d → %d qubits (removed %s)",
        n_qubits, tapered_n, removed,
    )

    return TaperingResult(
        tapered_op=tapered_op,
        original_n_qubits=n_qubits,
        tapered_n_qubits=tapered_n,
        removed_positions=list(removed),
        stabilizers=stabilizers,
    )
