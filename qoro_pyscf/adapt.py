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
ADAPT-VQE: Adaptive Derivative-Assembled Pseudo-Trotter ansatz.

Grows the circuit one operator at a time by selecting the operator from a
pool whose energy gradient is largest.  Produces the most compact circuit
for a given accuracy target.

Reference: Grimsley et al., Nature Communications 10, 3007 (2019)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

if TYPE_CHECKING:
    from maestro.circuits import QuantumCircuit

from qoro_pyscf.ansatze import (
    _apply_double_excitation,
    _apply_hf_gates,
    _apply_single_excitation,
    _get_uccsd_excitations,
    _QC,
)
from qoro_pyscf.backends import BackendConfig
from qoro_pyscf.expectation import compute_energy

logger = logging.getLogger(__name__)

# Step size for finite-difference gradient estimation
_EPSILON = 0.01


@dataclass
class Operator:
    """A fermionic excitation operator in the pool."""

    kind: str  # "single" or "double"
    indices: tuple  # (i, a) for singles, (i, j, a, b) for doubles

    def apply(self, qc: QuantumCircuit, theta: float) -> None:
        """Append this operator to a circuit with angle theta."""
        if self.kind == "single":
            _apply_single_excitation(qc, *self.indices, theta)
        else:
            _apply_double_excitation(qc, *self.indices, theta)


def build_operator_pool(
    n_qubits: int,
    nelec: tuple[int, int],
    pool: str = "sd",
) -> list[Operator]:
    """
    Build the operator pool for ADAPT-VQE.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    nelec : (int, int)
        Number of (alpha, beta) electrons.
    pool : str
        Pool type: "sd" (singles + doubles) or "d" (doubles only).

    Returns
    -------
    operators : list[Operator]
        The operator pool.
    """
    singles, doubles = _get_uccsd_excitations(n_qubits, nelec)

    ops = []
    if pool != "d":
        for idx in singles:
            ops.append(Operator(kind="single", indices=idx))
    for idx in doubles:
        ops.append(Operator(kind="double", indices=idx))

    return ops


def _build_adapt_circuit(
    n_qubits: int,
    nelec: tuple[int, int],
    selected_ops: list[Operator],
    params: np.ndarray,
) -> QuantumCircuit:
    """Build the current ADAPT circuit: HF + selected operators."""
    qc = _QC()
    _apply_hf_gates(qc, n_qubits, nelec)
    for op, theta in zip(selected_ops, params):
        op.apply(qc, float(theta))
    return qc


def run_adapt_vqe(
    n_qubits: int,
    nelec: tuple[int, int],
    identity_offset: float,
    pauli_labels: list[str],
    pauli_coeffs: np.ndarray,
    config: BackendConfig,
    pool: str = "sd",
    gradient_threshold: float = 1e-3,
    max_operators: int = 50,
    optimizer: str = "COBYLA",
    maxiter_per_step: int = 100,
    verbose: bool = False,
) -> dict:
    """
    Run ADAPT-VQE: adaptively grow the circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    nelec : (int, int)
        Number of (alpha, beta) electrons.
    identity_offset : float
        Identity coefficient of the Hamiltonian.
    pauli_labels : list[str]
        Non-identity Pauli strings.
    pauli_coeffs : np.ndarray
        Coefficients for each Pauli string.
    config : BackendConfig
        Qoro backend configuration.
    pool : str
        Operator pool: "sd" or "d".
    gradient_threshold : float
        Stop when max |gradient| falls below this.
    max_operators : int
        Maximum number of operators to add.
    optimizer : str
        SciPy optimizer for re-optimization.
    maxiter_per_step : int
        Max VQE iterations per ADAPT step.
    verbose : bool
        Print progress.

    Returns
    -------
    result : dict
        Keys: energy, params, selected_ops, converged, n_operators,
        energy_history, circuit.
    """
    op_pool = build_operator_pool(n_qubits, nelec, pool)

    if verbose:
        print(f"  [ADAPT-VQE] Operator pool size: {len(op_pool)}")
        print(f"  [ADAPT-VQE] Gradient threshold: {gradient_threshold}")

    selected_ops: list[Operator] = []
    params = np.array([], dtype=float)
    energy_history: list[float] = []

    # Initial HF energy
    hf_circuit = _build_adapt_circuit(n_qubits, nelec, [], np.array([]))
    current_energy = compute_energy(
        hf_circuit, identity_offset, pauli_labels, pauli_coeffs, config
    )
    energy_history.append(current_energy)

    if verbose:
        print(f"  [ADAPT-VQE] HF energy: {current_energy:+.10f}")

    for step in range(max_operators):
        # --- Screen all operators: compute gradients via finite differences ---
        gradients = np.zeros(len(op_pool))

        for idx, op in enumerate(op_pool):
            trial_ops = selected_ops + [op]

            # Forward: current circuit + op(+ε)
            params_fwd = np.append(params, +_EPSILON)
            qc_fwd = _build_adapt_circuit(n_qubits, nelec, trial_ops, params_fwd)
            e_fwd = compute_energy(
                qc_fwd, identity_offset, pauli_labels, pauli_coeffs, config
            )

            # Backward: current circuit + op(-ε)
            params_bwd = np.append(params, -_EPSILON)
            qc_bwd = _build_adapt_circuit(n_qubits, nelec, trial_ops, params_bwd)
            e_bwd = compute_energy(
                qc_bwd, identity_offset, pauli_labels, pauli_coeffs, config
            )

            gradients[idx] = (e_fwd - e_bwd) / (2.0 * _EPSILON)

        # --- Select operator with largest gradient ---
        max_idx = np.argmax(np.abs(gradients))
        max_grad = np.abs(gradients[max_idx])

        if verbose:
            print(
                f"  [ADAPT-VQE] Step {step + 1}: "
                f"max |grad| = {max_grad:.6f}  "
                f"(op {max_idx}: {op_pool[max_idx].kind} {op_pool[max_idx].indices})"
            )

        # --- Check convergence ---
        if max_grad < gradient_threshold:
            if verbose:
                print(f"  [ADAPT-VQE] Converged! max |grad| < {gradient_threshold}")
            break

        # --- Add the selected operator ---
        selected_ops.append(op_pool[max_idx])
        params = np.append(params, 0.0)  # initial angle for new operator

        # --- Re-optimise all parameters ---
        def cost(p):
            qc = _build_adapt_circuit(n_qubits, nelec, selected_ops, p)
            return compute_energy(
                qc, identity_offset, pauli_labels, pauli_coeffs, config
            )

        opts = {"maxiter": maxiter_per_step}
        if optimizer.upper() == "COBYLA":
            opts["rhobeg"] = 0.3

        opt = minimize(cost, params, method=optimizer, options=opts)
        params = opt.x
        current_energy = opt.fun
        energy_history.append(current_energy)

        if verbose:
            print(
                f"  [ADAPT-VQE]   → E = {current_energy:+.10f}  "
                f"({len(selected_ops)} ops, {len(params)} params)"
            )

    # --- Build final circuit ---
    final_circuit = _build_adapt_circuit(n_qubits, nelec, selected_ops, params)

    converged = (
        step < max_operators - 1  # didn't hit the cap
        if max_operators > 0
        else True
    )

    return {
        "energy": current_energy,
        "params": params,
        "selected_ops": selected_ops,
        "converged": converged,
        "n_operators": len(selected_ops),
        "energy_history": energy_history,
        "circuit": final_circuit,
    }
