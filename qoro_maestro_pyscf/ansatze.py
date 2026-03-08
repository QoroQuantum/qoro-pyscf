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
Ansatz builders for VQE using native Maestro QuantumCircuit objects.

All builders return ``maestro.circuits.QuantumCircuit`` instances — no QASM
string intermediary.
"""

from __future__ import annotations

import itertools

import numpy as np

from maestro.circuits import QuantumCircuit


# ──────────────────────────────────────────────────────────────────────────────
# Hartree-Fock initial state
# ──────────────────────────────────────────────────────────────────────────────

def hartree_fock_circuit(n_qubits: int, nelec: int | tuple[int, int]) -> QuantumCircuit:
    """
    Build the Hartree-Fock initial state as a Maestro QuantumCircuit.

    In the Jordan-Wigner mapping, the HF state corresponds to X gates on the
    spin-orbitals that are occupied:
      - Spin-orbitals 0, 2, 4, ... are α (even indices)
      - Spin-orbitals 1, 3, 5, ... are β (odd indices)

    Parameters
    ----------
    n_qubits : int
        Total number of qubits (spin-orbitals).
    nelec : int or (int, int)
        Number of electrons. If tuple, (n_alpha, n_beta).

    Returns
    -------
    QuantumCircuit
        Circuit that prepares |HF⟩ from |0...0⟩.
    """
    if isinstance(nelec, (tuple, list)):
        n_alpha, n_beta = nelec
    else:
        n_beta = nelec // 2
        n_alpha = nelec - n_beta

    qc = QuantumCircuit()

    # Occupy alpha spin-orbitals: qubits 0, 2, 4, ...
    for i in range(n_alpha):
        qc.x(2 * i)

    # Occupy beta spin-orbitals: qubits 1, 3, 5, ...
    for i in range(n_beta):
        qc.x(2 * i + 1)

    return qc


# ──────────────────────────────────────────────────────────────────────────────
# Hardware-Efficient Ansatz
# ──────────────────────────────────────────────────────────────────────────────

def hardware_efficient_ansatz(
    params: np.ndarray,
    n_qubits: int,
    n_layers: int,
    include_hf: bool = False,
    nelec: int | tuple[int, int] | None = None,
) -> QuantumCircuit:
    """
    Build a hardware-efficient ansatz as a native Maestro QuantumCircuit.

    Structure per layer:
        Ry(θ) + Rz(θ) on every qubit, followed by a linear CNOT ladder.

    Parameters
    ----------
    params : array-like, shape (n_qubits * 2 * n_layers,)
        Variational parameters for Ry and Rz rotations.
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of entangling layers.
    include_hf : bool
        If True, prepend the Hartree-Fock initial state.
    nelec : int or (int, int) or None
        Number of electrons. Required if ``include_hf`` is True.

    Returns
    -------
    QuantumCircuit
        The parameterised circuit.
    """
    qc = QuantumCircuit()

    # Optional HF initial state
    if include_hf:
        if nelec is None:
            raise ValueError("nelec is required when include_hf=True")
        _apply_hf_gates(qc, n_qubits, nelec)

    idx = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(q, float(params[idx]))
            qc.rz(q, float(params[idx + 1]))
            idx += 2
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)

    return qc


def hardware_efficient_param_count(n_qubits: int, n_layers: int) -> int:
    """Return the number of parameters for the hardware-efficient ansatz."""
    return n_qubits * 2 * n_layers


# ──────────────────────────────────────────────────────────────────────────────
# UCCSD Ansatz
# ──────────────────────────────────────────────────────────────────────────────

def _get_uccsd_excitations(
    n_qubits: int, nelec: tuple[int, int]
) -> tuple[list[tuple[int, int]], list[tuple[int, int, int, int]]]:
    """
    Enumerate single and double excitation operator indices (spin-orbital basis).

    Returns
    -------
    singles : list of (i, a) tuples
        Each (i, a) represents the excitation a†_a a_i − h.c.
    doubles : list of (i, j, a, b) tuples
        Each (i, j, a, b) represents a†_a a†_b a_j a_i − h.c.
    """
    n_alpha, n_beta = nelec
    n_spatial = n_qubits // 2

    # Occupied and virtual spin-orbital indices
    occ_alpha = [2 * i for i in range(n_alpha)]
    occ_beta = [2 * i + 1 for i in range(n_beta)]
    vir_alpha = [2 * i for i in range(n_alpha, n_spatial)]
    vir_beta = [2 * i + 1 for i in range(n_beta, n_spatial)]

    occupied = sorted(occ_alpha + occ_beta)
    virtual = sorted(vir_alpha + vir_beta)

    # Singles: i → a
    singles = [(i, a) for i in occupied for a in virtual]

    # Doubles: (i, j) → (a, b)
    doubles = [
        (i, j, a, b)
        for i, j in itertools.combinations(occupied, 2)
        for a, b in itertools.combinations(virtual, 2)
    ]

    return singles, doubles


def uccsd_ansatz(
    params: np.ndarray,
    n_qubits: int,
    nelec: int | tuple[int, int],
) -> QuantumCircuit:
    """
    Build a first-order Trotterised UCCSD ansatz as a Maestro QuantumCircuit.

    Prepends the Hartree-Fock state, then applies parameterised single and
    double excitation circuits mapped through Jordan-Wigner.

    The excitation operator ``θ(a†_a a_i − a†_i a_a)`` is mapped to a sequence
    of CNOT + Ry gates using the standard JW decomposition.

    Parameters
    ----------
    params : array-like
        Variational amplitudes. Length = n_singles + n_doubles.
    n_qubits : int
        Number of qubits (spin-orbitals).
    nelec : int or (int, int)
        Number of electrons.

    Returns
    -------
    QuantumCircuit
        The UCCSD circuit.
    """
    if isinstance(nelec, int):
        n_beta = nelec // 2
        n_alpha = nelec - n_beta
        nelec = (n_alpha, n_beta)

    qc = QuantumCircuit()

    # 1. Hartree-Fock initial state
    _apply_hf_gates(qc, n_qubits, nelec)

    # 2. Enumerate excitations
    singles, doubles = _get_uccsd_excitations(n_qubits, nelec)

    idx = 0

    # 3. Apply single excitation circuits
    for (i, a) in singles:
        theta = float(params[idx])
        _apply_single_excitation(qc, i, a, theta)
        idx += 1

    # 4. Apply double excitation circuits
    for (i, j, a, b) in doubles:
        theta = float(params[idx])
        _apply_double_excitation(qc, i, j, a, b, theta)
        idx += 1

    return qc


def uccsd_param_count(n_qubits: int, nelec: int | tuple[int, int]) -> int:
    """Return the number of UCCSD parameters."""
    if isinstance(nelec, int):
        n_beta = nelec // 2
        n_alpha = nelec - n_beta
        nelec = (n_alpha, n_beta)
    singles, doubles = _get_uccsd_excitations(n_qubits, nelec)
    return len(singles) + len(doubles)


# ──────────────────────────────────────────────────────────────────────────────
# Circuit primitives for excitation operators (JW-mapped)
# ──────────────────────────────────────────────────────────────────────────────

def _apply_single_excitation(
    qc: QuantumCircuit, i: int, a: int, theta: float
) -> None:
    """
    Apply a single excitation gate exp(θ(a†_a a_i − h.c.)) via JW mapping.

    Uses the Givens rotation decomposition:
      exp(-iθ/2 (X_i Y_a − Y_i X_a) ∏_{k=i+1}^{a-1} Z_k)

    Decomposed into CNOT ladder + Ry rotations.
    """
    lo, hi = min(i, a), max(i, a)

    # CNOT staircase to propagate JW Z-string
    for q in range(lo, hi):
        qc.cx(q, q + 1)

    # Parameterised rotation
    qc.ry(hi, theta)

    # Reverse CNOT staircase
    for q in range(hi - 1, lo - 1, -1):
        qc.cx(q, q + 1)


def _apply_double_excitation(
    qc: QuantumCircuit, i: int, j: int, a: int, b: int, theta: float
) -> None:
    """
    Apply a double excitation gate exp(θ(a†_a a†_b a_j a_i − h.c.)).

    Uses the exact decomposition into 8 Pauli evolution terms from the
    JW mapping of the double excitation operator. Each term is a product
    of Paulis on the 4 active qubits (with a Z-string on intermediate
    qubits for JW), evolved via CNOT + Rz circuits.

    Reference: Yordanov et al., Phys. Rev. A 102, 062612 (2020)
    """
    qubits = sorted([i, j, a, b])
    p, q, r, s = qubits

    # The 8 Pauli terms of the double excitation (JW-mapped):
    #   θ/8 * (XXXY - XXYX + XYXX - YXXX - XYYY + YXYY - YYXY + YYYX)
    # where X,Y act on qubits p,q,r,s respectively, with Z-strings between.
    #
    # Each term exp(-i θ/8 P) is implemented via:
    #   1. Basis rotation (H for X, Sdg+H for Y) on each qubit
    #   2. CNOT ladder to compute parity
    #   3. Rz(±θ/4) on the last qubit
    #   4. Reverse CNOT ladder
    #   5. Undo basis rotation

    pauli_terms = [
        ("XXXY", +1),
        ("XXYY", -1),  # -XXYX mapped to sorted order
        ("XYXY", +1),  # XYXX mapped to sorted order
        ("YXXY", -1),  # -YXXX mapped to sorted order
        ("XYYY", -1),
        ("YXYY", +1),
        ("YYXY", -1),
        ("YYYY", +1),  # YYYX mapped to sorted order
    ]

    # We need to handle the actual Pauli assignments on [p,q,r,s]
    # For the double excitation (i,j) -> (a,b), the 8 terms are:
    pauli_terms_actual = [
        ([("X", p), ("X", q), ("X", r), ("Y", s)], +1),
        ([("X", p), ("X", q), ("Y", r), ("X", s)], -1),
        ([("X", p), ("Y", q), ("X", r), ("X", s)], +1),
        ([("Y", p), ("X", q), ("X", r), ("X", s)], -1),
        ([("X", p), ("Y", q), ("Y", r), ("Y", s)], -1),
        ([("Y", p), ("X", q), ("Y", r), ("Y", s)], +1),
        ([("Y", p), ("Y", q), ("X", r), ("Y", s)], -1),
        ([("Y", p), ("Y", q), ("Y", r), ("X", s)], +1),
    ]

    # Include Z-string on intermediate qubits for JW
    intermediate = list(range(p + 1, s))
    intermediate = [k for k in intermediate if k not in qubits]

    for paulis, sign in pauli_terms_actual:
        angle = sign * theta / 8.0
        if abs(angle) < 1e-15:
            continue

        active_qubits = []

        # Basis rotation into Z basis
        for pauli, qubit in paulis:
            if pauli == "X":
                qc.h(qubit)
                active_qubits.append(qubit)
            elif pauli == "Y":
                # Rx(π/2) = S†·H maps Y→Z
                qc.rx(qubit, np.pi / 2)
                active_qubits.append(qubit)

        # Z-string qubits are already in Z basis
        for zq in intermediate:
            active_qubits.append(zq)

        active_qubits.sort()

        # CNOT cascade to compute parity
        for idx in range(len(active_qubits) - 1):
            qc.cx(active_qubits[idx], active_qubits[idx + 1])

        # Rz rotation on the last qubit
        qc.rz(active_qubits[-1], 2 * angle)

        # Reverse CNOT cascade
        for idx in range(len(active_qubits) - 2, -1, -1):
            qc.cx(active_qubits[idx], active_qubits[idx + 1])

        # Undo basis rotation
        for pauli, qubit in paulis:
            if pauli == "X":
                qc.h(qubit)
            elif pauli == "Y":
                qc.rx(qubit, -np.pi / 2)


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def _apply_hf_gates(
    qc: QuantumCircuit,
    n_qubits: int,
    nelec: int | tuple[int, int],
) -> None:
    """
    Apply Hartree-Fock X gates directly into an existing circuit.

    Applies X gates on the occupied spin-orbitals (JW convention):
      - α spin-orbitals: even-indexed qubits 0, 2, 4, ...
      - β spin-orbitals: odd-indexed qubits 1, 3, 5, ...
    """
    if isinstance(nelec, (tuple, list)):
        n_alpha, n_beta = nelec
    else:
        n_beta = nelec // 2
        n_alpha = nelec - n_beta

    for i in range(n_alpha):
        qc.x(2 * i)
    for i in range(n_beta):
        qc.x(2 * i + 1)
