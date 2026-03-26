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
Expectation value engine using Qoro's native QuantumCircuit.estimate().

Wraps circuit evaluation so that the rest of the library can call a single
function without worrying about backend configuration details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from maestro.circuits import QuantumCircuit

from qoro_pyscf.backends import BackendConfig


def evaluate_expectation(
    circuit: QuantumCircuit,
    pauli_labels: list[str],
    config: BackendConfig,
) -> np.ndarray:
    """
    Evaluate expectation values of Pauli observables on a Qoro circuit.

    All observables are batched into a single ``qc.estimate()`` call, so
    Qoro evaluates them in one statevector (or MPS) pass on the GPU.

    Parameters
    ----------
    circuit : QuantumCircuit
        The prepared (parameterised) circuit.
    pauli_labels : list[str]
        Pauli observable strings, e.g. ``["ZZII", "IXYZ"]``.
    config : BackendConfig
        Qoro backend configuration.

    Returns
    -------
    expectation_values : np.ndarray, shape (len(pauli_labels),)
        Real-valued expectation values ⟨ψ|Pᵢ|ψ⟩.
    """
    if not pauli_labels:
        return np.array([], dtype=float)

    estimate_kwargs = {
        "observables": pauli_labels,
        "simulator_type": config.simulator_type,
        "simulation_type": config.simulation_type,
    }
    if config.mps_bond_dim is not None:
        estimate_kwargs["max_bond_dimension"] = config.mps_bond_dim

    result = circuit.estimate(**estimate_kwargs)

    return np.array(result["expectation_values"], dtype=float)


def compute_energy(
    circuit: QuantumCircuit,
    identity_offset: float,
    pauli_labels: list[str],
    pauli_coeffs: np.ndarray,
    config: BackendConfig,
) -> float:
    """
    Compute the total energy ⟨H⟩ = c₀ + Σᵢ Re(cᵢ)·⟨Pᵢ⟩ for a given circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        The prepared circuit.
    identity_offset : float
        Coefficient of the identity term.
    pauli_labels : list[str]
        Non-identity Pauli terms.
    pauli_coeffs : np.ndarray
        Complex coefficients for each Pauli term.
    config : BackendConfig
        Qoro backend configuration.

    Returns
    -------
    energy : float
        The expectation value of the Hamiltonian.
    """
    exp_vals = evaluate_expectation(circuit, pauli_labels, config)
    # Coefficients are real for a Hermitian Hamiltonian (physical observable).
    # We use .real to drop any floating-point imaginary noise from OpenFermion.
    return identity_offset + float(np.dot(pauli_coeffs.real, exp_vals))


def get_state_probabilities(
    circuit: QuantumCircuit,
    config: BackendConfig,
) -> np.ndarray:
    """
    Get the full probability distribution |⟨k|ψ⟩|² for each computational basis state.

    Wraps Qoro's native ``get_probabilities()`` with the configured backend.

    Parameters
    ----------
    circuit : QuantumCircuit
        The prepared circuit.
    config : BackendConfig
        Qoro backend configuration.

    Returns
    -------
    probabilities : np.ndarray, shape (2**n_qubits,)
        Probability of each computational basis state.
    """
    import maestro

    kwargs = {
        "simulator_type": config.simulator_type,
        "simulation_type": config.simulation_type,
    }
    if config.mps_bond_dim is not None:
        kwargs["max_bond_dimension"] = config.mps_bond_dim

    probs = maestro.get_probabilities(circuit, **kwargs)
    return np.array(probs, dtype=float)


def compute_state_fidelity(
    circuit_a: QuantumCircuit,
    circuit_b: QuantumCircuit,
    config: BackendConfig,
) -> float:
    """
    Compute the classical fidelity between two circuit states.

    Uses the Bhattacharyya coefficient: F = (Σ √(pᵢ·qᵢ))², which equals
    the true quantum fidelity |⟨ψ_a|ψ_b⟩|² when both states are pure and
    have non-negative real amplitudes (common for VQE ground states).

    For general states with complex phases, this is a lower bound on the
    true fidelity.

    Parameters
    ----------
    circuit_a, circuit_b : QuantumCircuit
        The two circuits to compare.
    config : BackendConfig
        Qoro backend configuration.

    Returns
    -------
    fidelity : float
        Classical fidelity in [0, 1].  1.0 = identical probability distributions.
    """
    p = get_state_probabilities(circuit_a, config)
    q = get_state_probabilities(circuit_b, config)
    min_len = min(len(p), len(q))
    bhatt = float(np.sum(np.sqrt(p[:min_len] * q[:min_len])))
    return bhatt ** 2


def compute_overlap(
    circuit_a: QuantumCircuit,
    circuit_b: QuantumCircuit,
    config: BackendConfig,
) -> float:
    """
    Compute the quantum state overlap |⟨ψ_a|ψ_b⟩|² between two circuits.

    This is the **canonical** function for VQD overlap penalties.  It
    automatically uses the best available Qoro API:

    1. **``maestro.inner_product(c1, c2, ...)``** — exact, native, and
       scales to large qubit counts (e.g. MPS inner product).  Used when
       available.
    2. **Bhattacharyya fallback** — extracts probability distributions via
       ``maestro.get_probabilities`` and computes F = (Σ √(pᵢqᵢ))².
       This is a lower bound on the true fidelity but works with the
       current installed wheel.

    Parameters
    ----------
    circuit_a, circuit_b : QuantumCircuit
        The two ansatz circuits whose states will be compared.
    config : BackendConfig
        Qoro backend configuration (simulator type, simulation mode,
        MPS bond dimension, etc.).

    Returns
    -------
    overlap : float
        |⟨ψ_a|ψ_b⟩|² in [0, 1].

    Notes
    -----
    Once ``maestro.inner_product`` is available, this function will use
    it automatically with no changes required at call sites.
    """
    import maestro

    if hasattr(maestro, "inner_product"):
        # --- Exact path: native Qoro inner product ---
        kwargs: dict = {
            "simulator_type": config.simulator_type,
            "simulation_type": config.simulation_type,
        }
        if config.mps_bond_dim is not None:
            kwargs["max_bond_dimension"] = config.mps_bond_dim
        result = maestro.inner_product(circuit_a, circuit_b, **kwargs)
        # inner_product returns |⟨ψ_a|ψ_b⟩|² directly (real, non-negative)
        return float(abs(result))

    # --- Fallback: Bhattacharyya probability-based fidelity ---
    # TODO: remove fallback once maestro.inner_product is available in the wheel
    return compute_state_fidelity(circuit_a, circuit_b, config)


def compute_statevector_fidelity(
    circuit_a: QuantumCircuit,
    circuit_b: QuantumCircuit,
    config: BackendConfig,
) -> float:
    """
    Compute the quantum fidelity between two pure circuit states.

    Delegates to :func:`compute_overlap`, which automatically uses
    ``maestro.inner_product`` when available.

    Parameters
    ----------
    circuit_a, circuit_b : QuantumCircuit
        The two circuits to compare.
    config : BackendConfig
        Qoro backend configuration.

    Returns
    -------
    fidelity : float
        |⟨ψ_a|ψ_b⟩|² in [0, 1].
    """
    return compute_overlap(circuit_a, circuit_b, config)

