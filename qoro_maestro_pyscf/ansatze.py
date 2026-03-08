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
Ansatz builders for VQE using native Maestro QuantumCircuit objects.

All builders return ``maestro.circuits.QuantumCircuit`` instances — no QASM
string intermediary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import itertools

import numpy as np

if TYPE_CHECKING:
    from maestro.circuits import QuantumCircuit


def _QC() -> QuantumCircuit:
    """Lazy-import QuantumCircuit to avoid top-level libmaestro.so dependency."""
    from maestro.circuits import QuantumCircuit as _QC_cls
    return _QC_cls()


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

    qc = _QC()

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
    qc = _QC()

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

    qc = _QC()

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
# UpCCD Ansatz (Unitary paired Coupled Cluster Doubles)
# ──────────────────────────────────────────────────────────────────────────────

def _get_upccd_excitations(
    n_qubits: int, nelec: tuple[int, int]
) -> list[tuple[int, int]]:
    """
    Enumerate paired double excitation indices (spatial orbital basis).

    Paired doubles excite both α and β electrons from the same occupied
    spatial orbital *i* to the same virtual spatial orbital *a*:

        a†_{aα} a†_{aβ} a_{iβ} a_{iα} − h.c.

    This preserves seniority zero (all spatial orbitals are either doubly
    occupied or empty), making UpCCD an excellent ansatz for singlet states.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (spin-orbitals).
    nelec : (int, int)
        Number of (alpha, beta) electrons.

    Returns
    -------
    pairs : list of (i, a)
        Spatial orbital indices for each paired excitation.
        The corresponding spin-orbital qubits are (2i, 2i+1) → (2a, 2a+1).

    References
    ----------
    - Nam et al., arXiv:2309.12396
    - Khamoshi, arXiv:2508.21679 (oo-pCCD-UpCCD)
    """
    n_alpha, n_beta = nelec
    n_spatial = n_qubits // 2

    # Paired excitations: spatial orbital i (doubly occupied) → a (empty)
    # Only pairs where both α and β are occupied/virtual
    n_occ = min(n_alpha, n_beta)  # number of doubly-occupied spatial orbitals
    occ_spatial = list(range(n_occ))
    vir_spatial = list(range(max(n_alpha, n_beta), n_spatial))

    pairs = [(i, a) for i in occ_spatial for a in vir_spatial]
    return pairs


def upccd_ansatz(
    params: np.ndarray,
    n_qubits: int,
    nelec: int | tuple[int, int],
) -> QuantumCircuit:
    """
    Build a UpCCD (Unitary paired Coupled Cluster Doubles) ansatz.

    Uses only **paired** double excitations — both electrons from the same
    spatial orbital excite to the same virtual spatial orbital.  This yields
    an extremely compact circuit suitable for seniority-zero singlet states.

    Compared to UCCSD:
    - Far fewer parameters: N_occ × N_vir (vs. O(N² × M²) for full doubles)
    - Shallower circuits with fewer CNOT gates
    - Exact for seniority-zero states when combined with orbital optimisation

    Amplitudes can be initialised from a classical pCCD calculation (no VQE
    optimisation needed), following arXiv:2508.21679.

    Parameters
    ----------
    params : array-like
        Variational amplitudes. Length = number of paired excitations.
        Can be from VQE optimisation or from classical pCCD amplitudes.
    n_qubits : int
        Number of qubits (spin-orbitals).
    nelec : int or (int, int)
        Number of electrons.

    Returns
    -------
    QuantumCircuit
        The UpCCD circuit.
    """
    if isinstance(nelec, int):
        n_beta = nelec // 2
        n_alpha = nelec - n_beta
        nelec = (n_alpha, n_beta)

    qc = _QC()

    # 1. Hartree-Fock initial state
    _apply_hf_gates(qc, n_qubits, nelec)

    # 2. Enumerate paired excitations
    pairs = _get_upccd_excitations(n_qubits, nelec)

    # 3. Apply paired double excitation circuits
    for idx, (i_spat, a_spat) in enumerate(pairs):
        theta = float(params[idx])
        # Map spatial → spin-orbital indices (interleaved JW ordering)
        i_alpha, i_beta = 2 * i_spat, 2 * i_spat + 1
        a_alpha, a_beta = 2 * a_spat, 2 * a_spat + 1
        _apply_double_excitation(qc, i_alpha, i_beta, a_alpha, a_beta, theta)

    return qc


def upccd_param_count(n_qubits: int, nelec: int | tuple[int, int]) -> int:
    """Return the number of UpCCD parameters (= N_occ × N_vir)."""
    if isinstance(nelec, int):
        n_beta = nelec // 2
        n_alpha = nelec - n_beta
        nelec = (n_alpha, n_beta)
    pairs = _get_upccd_excitations(n_qubits, nelec)
    return len(pairs)


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

    Uses a compact parity-mapping decomposition that reduces the 4-qubit
    operation to a single Ry rotation:

    1. CNOT cascade maps the two target states (occupied and excited)
       to differ only on a single qubit.
    2. Ry rotation on that qubit implements the excitation.
    3. Reverse CNOT cascade restores the parity encoding.

    This is exact (not Trotterised) and uses only 6 CNOTs + 1 Ry.
    """
    qubits = sorted([i, j, a, b])
    p, q, r, s = qubits

    # Parity mapping: reduce 4-qubit difference to 1-qubit
    # After these CNOTs, the occupied state |..pq..| and excited state
    # |..rs..| differ only on qubit p.
    qc.cx(p, q)
    qc.cx(r, s)
    qc.cx(p, r)

    # The actual excitation rotation
    qc.ry(p, 2 * theta)

    # Undo parity mapping
    qc.cx(p, r)
    qc.cx(r, s)
    qc.cx(p, q)


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
