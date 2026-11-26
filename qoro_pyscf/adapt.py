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

import itertools
import logging
import time
from dataclasses import dataclass
from functools import cached_property

import maestro
import numpy as np
from maestro.circuits import QuantumCircuit
from openfermion import FermionOperator, hermitian_conjugated, jordan_wigner
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

from divi.qprog import ParameterSpace
from divi.qprog.optimizers import RotosolveOptimizer
from qoro_pyscf.backends import BackendConfig

logger = logging.getLogger(__name__)

# Step size for finite-difference gradient estimation
_EPSILON = 0.01


def _get_uccsd_excitations(
    n_qubits: int, nelec: tuple[int, int]
) -> tuple[list[tuple[int, int]], list[tuple[int, int, int, int]]]:
    """Spin-conserving single ``(i, a)`` and double ``(i, j, a, b)`` spin-orbital
    excitations, alpha on even qubits and beta on odd ones."""
    n_alpha, n_beta = nelec
    n_spatial = n_qubits // 2
    occupied = sorted(
        [2 * i for i in range(n_alpha)] + [2 * i + 1 for i in range(n_beta)]
    )
    virtual = sorted(
        [2 * i for i in range(n_alpha, n_spatial)]
        + [2 * i + 1 for i in range(n_beta, n_spatial)]
    )
    singles = [(i, a) for i in occupied for a in virtual if i % 2 == a % 2]
    doubles = [
        (i, j, a, b)
        for i, j in itertools.combinations(occupied, 2)
        for a, b in itertools.combinations(virtual, 2)
        if sorted([i % 2, j % 2]) == sorted([a % 2, b % 2])
    ]
    return singles, doubles


def _apply_pauli_rotation(
    qc: QuantumCircuit, word: dict[int, str], theta: float
) -> None:
    """``exp(-i theta/2 P)`` for the Pauli word ``P`` given as ``{qubit: letter}``."""
    qubits = sorted(word)
    for qubit in qubits:
        if word[qubit] == "X":
            qc.h(qubit)
        elif word[qubit] == "Y":
            qc.rx(qubit, np.pi / 2)
    for control, target in itertools.pairwise(qubits):
        qc.cx(control, target)
    qc.rz(qubits[-1], theta)
    for control, target in reversed(list(itertools.pairwise(qubits))):
        qc.cx(control, target)
    for qubit in qubits:
        if word[qubit] == "X":
            qc.h(qubit)
        elif word[qubit] == "Y":
            qc.rx(qubit, -np.pi / 2)


def _apply_hf_gates(qc: QuantumCircuit, nelec: tuple[int, int]) -> None:
    """X on the occupied spin-orbitals: alpha on even qubits, beta on odd."""
    n_alpha, n_beta = nelec
    for i in range(n_alpha):
        qc.x(2 * i)
    for i in range(n_beta):
        qc.x(2 * i + 1)


def _compute_energy(
    circuit: QuantumCircuit,
    identity_offset: float,
    pauli_labels: list[str],
    pauli_coeffs: np.ndarray,
    config: BackendConfig,
) -> float:
    """``c0 + sum_i Re(c_i) <P_i>`` on Maestro; label character ``k`` is qubit ``k``."""
    if not pauli_labels:
        return identity_offset
    simulator_config = maestro.SimulatorConfig(
        simulator_type=config.simulator_type,
        simulation_type=config.simulation_type,
        max_bond_dimension=config.mps_bond_dim,
    )
    result = circuit.estimate(pauli_labels, simulator_config)
    values = np.asarray(result["expectation_values"], dtype=float)
    return identity_offset + float(np.dot(np.real(pauli_coeffs), values))


def build_pauli_terms(
    hamiltonian: SparsePauliOp,
) -> tuple[float, list[str], np.ndarray]:
    """Split a Qiskit Hamiltonian into :func:`run_adapt_vqe`'s inputs.

    Returns:
        The identity coefficient, the other Pauli labels with character ``k``
        on qubit ``k`` (Qiskit's labels reversed), and their coefficients.
    """
    hamiltonian = hamiltonian.simplify(atol=0)
    offset, labels, coeffs = 0.0, [], []
    for label, coeff in zip(hamiltonian.paulis.to_labels(), hamiltonian.coeffs):
        if set(label) == {"I"}:
            offset += coeff.real
        else:
            labels.append(label[::-1])
            coeffs.append(coeff)
    return offset, labels, np.array(coeffs)


@dataclass
class Operator:
    """A fermionic excitation operator in the pool.

    ``apply`` adds ``exp(theta (T - T†))`` under Jordan-Wigner, with
    ``T = a†_a a_i`` for a single ``(i, a)`` and ``a†_a a†_b a_j a_i`` for a
    double ``(i, j, a, b)``. Both are Givens rotations, so the energy along
    ``theta`` holds its first two harmonics.
    """

    kind: str  # "single" or "double"
    indices: tuple  # (i, a) for singles, (i, j, a, b) for doubles

    @cached_property
    def _rotations(self) -> list[tuple[dict[int, str], float]]:
        """Commuting Pauli words of ``T - T†`` and their rotation angle per unit
        ``theta``: each term ``i r P`` contributes ``exp(-i (-2 r theta)/2 P)``."""
        occupied = self.indices[: len(self.indices) // 2]
        virtual = self.indices[len(self.indices) // 2 :]
        term = " ".join(
            [f"{q}^" for q in virtual] + [f"{q}" for q in reversed(occupied)]
        )
        excitation = FermionOperator(term)
        generator = jordan_wigner(excitation - hermitian_conjugated(excitation))
        return [
            (dict(word), -2.0 * coeff.imag)
            for word, coeff in generator.terms.items()
            if abs(coeff) > 1e-12
        ]

    def apply(self, qc: QuantumCircuit, theta: float) -> None:
        """Append this operator to a circuit with angle theta."""
        for word, angle in self._rotations:
            _apply_pauli_rotation(qc, word, angle * theta)


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
    nelec: tuple[int, int],
    selected_ops: list[Operator],
    params: np.ndarray,
) -> QuantumCircuit:
    """Build the current ADAPT circuit: HF + selected operators."""
    qc = QuantumCircuit()
    _apply_hf_gates(qc, nelec)
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
    greedy: bool = False,
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
        Non-identity Pauli strings, character ``k`` acting on qubit ``k``.
    pauli_coeffs : np.ndarray
        Coefficients for each Pauli string.
    config : BackendConfig
        Maestro backend configuration.
    pool : str
        Operator pool: "sd" or "d".
    gradient_threshold : float
        Stop when max |gradient| falls below this.
    max_operators : int
        Maximum number of operators to add.
    optimizer : str
        Optimizer for re-optimization: ``"ROTOSOLVE"`` for divi's
        :class:`~divi.qprog.optimizers.RotosolveOptimizer`, swept until the
        energy changes by less than 1e-8, or any SciPy method.
    maxiter_per_step : int
        Max iterations, or Rotosolve sweeps, per ADAPT step (ignored when
        ``greedy``).
    verbose : bool
        Print progress.
    greedy : bool
        If True, only optimize the last-added parameter at each step, with one
        Rotosolve sweep over it. Much faster but less accurate than
        re-optimizing all parameters.

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
        mode_str = "greedy (last-param only)" if greedy else "full re-optimization"
        print(f"  [ADAPT-VQE] Optimization mode: {mode_str}")

    selected_ops: list[Operator] = []
    params = np.array([], dtype=float)
    energy_history: list[float] = []

    def cost(p):
        qc = _build_adapt_circuit(nelec, selected_ops, p)
        return _compute_energy(qc, identity_offset, pauli_labels, pauli_coeffs, config)

    def cost_rows(rows):
        return np.array([cost(row) for row in np.atleast_2d(rows)])

    def cost_newest(rows):
        newest = np.atleast_2d(rows)[:, 0]
        return cost_rows(np.column_stack([np.tile(params[:-1], (len(newest), 1)), newest]))

    current_energy = cost(params)
    energy_history.append(current_energy)

    if verbose:
        print(f"  [ADAPT-VQE] HF energy: {current_energy:+.10f}")

    t_start = time.perf_counter()

    selected_indices: set[int] = set()  # track pool indices already used
    converged = False

    for step in range(max_operators):
        # --- Screen all operators: compute gradients via finite differences ---
        gradients = np.zeros(len(op_pool))
        n_pool = len(op_pool)

        if verbose:
            print(
                f"  [ADAPT-VQE] Screening {n_pool} operators...",
                end="", flush=True,
            )

        for idx, op in enumerate(op_pool):
            # In greedy mode, skip operators already in the circuit —
            # re-adding the same operator with only last-param optimisation
            # creates redundancy.
            if greedy and idx in selected_indices:
                continue

            trial_ops = selected_ops + [op]
            e_fwd, e_bwd = (
                _compute_energy(
                    _build_adapt_circuit(nelec, trial_ops, np.append(params, shift)),
                    identity_offset,
                    pauli_labels,
                    pauli_coeffs,
                    config,
                )
                for shift in (_EPSILON, -_EPSILON)
            )
            gradients[idx] = (e_fwd - e_bwd) / (2.0 * _EPSILON)

            if verbose and (idx + 1) % 100 == 0:
                print(f" {idx + 1}/{n_pool}", end="", flush=True)

        if verbose:
            print(" done", flush=True)

        # --- Select operator with largest gradient ---
        max_idx = int(np.argmax(np.abs(gradients)))
        max_grad = np.abs(gradients[max_idx])

        if verbose:
            elapsed = time.perf_counter() - t_start
            print(
                f"  [ADAPT-VQE] Step {step + 1}: "
                f"max |grad| = {max_grad:.6f}  "
                f"(op {max_idx}: {op_pool[max_idx].kind} "
                f"{op_pool[max_idx].indices})  [{elapsed:.1f}s]"
            )

        # --- Check convergence ---
        if max_grad < gradient_threshold:
            converged = True
            if verbose:
                print(f"  [ADAPT-VQE] Converged! max |grad| < {gradient_threshold}")
            break

        # --- Add the selected operator ---
        selected_ops.append(op_pool[max_idx])
        selected_indices.add(max_idx)
        params = np.append(params, 0.0)  # initial angle for new operator

        # --- Optimise parameters ---
        if greedy:
            result = RotosolveOptimizer().optimize(
                cost_newest,
                params[-1:],
                parameter_space=lambda: ParameterSpace(((1.0, 2),)),
                max_iterations=1,
            )
            params[-1] = result.x[0]
            current_energy = float(result.fun[0])
        elif optimizer.upper() == "ROTOSOLVE":
            rotosolve = RotosolveOptimizer()
            space = ParameterSpace(((1.0, 2),) * len(params))
            for _ in range(maxiter_per_step):
                previous_energy = current_energy
                result = rotosolve.optimize(
                    cost_rows,
                    params,
                    parameter_space=lambda: space,
                    max_iterations=1,
                )
                current_energy = float(result.fun[0])
                if abs(previous_energy - current_energy) < 1e-8:
                    break
            params = result.x
        else:
            opts = {"maxiter": maxiter_per_step}
            if optimizer.upper() == "COBYLA":
                opts["rhobeg"] = 0.3

            opt = minimize(cost, params, method=optimizer, options=opts)
            params = opt.x
            current_energy = float(opt.fun)

        energy_history.append(current_energy)

        if verbose:
            elapsed = time.perf_counter() - t_start
            print(
                f"  [ADAPT-VQE]   → E = {current_energy:+.10f}  "
                f"({len(selected_ops)} ops, {len(params)} params)"
                f"  [{elapsed:.1f}s]"
            )

    return {
        "energy": current_energy,
        "params": params,
        "selected_ops": selected_ops,
        "converged": converged,
        "n_operators": len(selected_ops),
        "energy_history": energy_history,
        "circuit": _build_adapt_circuit(nelec, selected_ops, params),
    }
