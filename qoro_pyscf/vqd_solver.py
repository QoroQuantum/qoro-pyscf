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

"""Variational Quantum Deflation as a PySCF active-space solver, on divi's VQE."""

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
import numpy.typing as npt
from pyscf import ao2mo
from qiskit import QuantumCircuit
from scipy.optimize import OptimizeResult

from divi.backends import MaestroSimulator
from divi.qprog.algorithms import (
    VQE,
    Ansatz,
    ElectronCounts,
    ReferenceStateAnsatz,
    VQEFCISolver,
)
from divi.qprog.optimizers import (
    Optimizer,
    ScipyMethod,
    ScipyOptimizer,
    argmin_finite,
)
from divi.qprog.problems import MolecularProblem


@dataclass
class _OptimizedState:
    """One optimised VQD root."""

    root: int
    energy: float
    params: np.ndarray
    statevector: np.ndarray
    vqe: VQE
    success: bool
    rdms: tuple | None = None


class VQDSolver(VQEFCISolver):
    """Ground and excited states by Variational Quantum Deflation (VQD).

    Root ``k`` minimises ``<H>(theta) + sum_{i<k} beta_i |<psi_i|psi(theta)>|^2``,
    the overlaps taken exactly from the backend's state vectors. No gradient of
    the penalty is supplied: gradient-based SciPy methods fall back to finite
    differences. RDM methods act on the active root (the last one unless
    :meth:`set_active_root` picks another).

    Args:
        backend: Backend the VQEs run on; it also supplies the state vectors.
        optimizer: Optimizer configuration; each root trains with its own copy.
        ansatz: Ansatz of every root, or None for :class:`VQE`'s default.
        max_iterations: Optimisation iterations per root.
        num_states: Roots to compute, the ground state included.
        penalty_weights: Overlap penalty ``beta``, one for all previous roots or
            one per root (padded with the last, truncated when too long).
        callback: Called per cost evaluation as
            ``(root, evaluation, cost, params)``.
        initial_point: Ground-state starting parameters; random when None.
        **vqe_kwargs: Further :class:`VQE` arguments, e.g. ``n_layers``.
    """

    def __init__(
        self,
        *,
        backend: MaestroSimulator,
        optimizer: Optimizer,
        ansatz: Ansatz | None = None,
        max_iterations: int = 200,
        num_states: int = 2,
        penalty_weights: float | Sequence[float] = 5.0,
        callback: Callable[[int, int, float, np.ndarray], None] | None = None,
        initial_point: npt.ArrayLike | None = None,
        **vqe_kwargs: Any,
    ) -> None:
        super().__init__(
            backend=backend,
            optimizer=optimizer,
            ansatz=ansatz,
            max_iterations=max_iterations,
            **vqe_kwargs,
        )
        self.num_states = num_states
        self.penalty_weights = penalty_weights
        self.callback = callback
        self.initial_point = initial_point
        self.energies: np.ndarray | None = None
        self.optimized_states: list[_OptimizedState] = []
        self._active_root = 0

    @property
    def n_roots(self) -> int:
        """Roots :meth:`kernel` computes: ``num_states`` or PySCF's ``nroots``."""
        return max(self.num_states, self.nroots)

    def _resolve_penalty_weights(self) -> list[float]:
        """``penalty_weights`` as a list of length ``n_roots - 1``."""
        n = max(self.n_roots - 1, 0)
        if isinstance(self.penalty_weights, Real):
            return [float(self.penalty_weights)] * n
        weights = [float(w) for w in self.penalty_weights]
        if len(weights) < n:
            weights.extend([weights[-1]] * (n - len(weights)))
        return weights[:n]

    def _build_initial_point(self, vqe: VQE, root: int) -> np.ndarray:
        """Ground: ``initial_point`` or small random angles; excited: the previous
        root's optimum shifted by ``+-pi/2`` per parameter."""
        rng = np.random.default_rng(42 + root)
        if root > 0:
            previous = self.optimized_states[-1].params
            return previous + rng.choice([-np.pi / 2, np.pi / 2], size=vqe.n_params)
        if self.initial_point is not None:
            return np.asarray(self.initial_point, dtype=float)
        starts_at_reference = (
            isinstance(vqe.ansatz, ReferenceStateAnsatz)
            and vqe.ansatz.starts_at_reference_state
        )
        scale = 0.05 if starts_at_reference else np.pi / 4
        return rng.uniform(-scale, scale, size=vqe.n_params)

    def _optimize_root(
        self, problem: MolecularProblem, root: int, weights: list[float]
    ) -> _OptimizedState:
        """Minimise the energy plus the overlap penalties against earlier roots."""
        vqe = VQE(
            problem,
            ansatz=self.ansatz,
            optimizer=self.optimizer.copy(),
            max_iterations=self.max_iterations,
            backend=self.backend,
            **self.vqe_kwargs,
        )
        reference = ElectronCounts(problem.n_electrons, problem.n_alpha, problem.n_beta)
        previous = np.array([state.statevector for state in self.optimized_states])
        root_weights = np.array(weights[: len(previous)])
        evaluations = 0

        def compute_energies(param_sets: np.ndarray) -> np.ndarray:
            values = vqe.evaluate(param_sets, vqe.cost_preprocessor())
            return np.array([values[i][0] for i in range(len(param_sets))])

        def cost_fn(params: np.ndarray) -> float | np.ndarray:
            nonlocal evaluations
            param_sets = np.atleast_2d(params)
            costs = compute_energies(param_sets)
            if len(previous):
                states = np.array(
                    [self._build_statevector(vqe, reference, row) for row in param_sets]
                )
                costs = costs + np.abs(previous.conj() @ states.T).T ** 2 @ root_weights
            for row, value in zip(param_sets, costs):
                evaluations += 1
                if self.callback is not None:
                    self.callback(root, evaluations, float(value), row)
            return costs if np.ndim(params) > 1 else float(costs[0])

        best_loss, best_params, iterations = np.inf, None, 0
        optimizer = vqe.optimizer
        stops_by_callback = (
            isinstance(optimizer, ScipyOptimizer)
            and optimizer.method == ScipyMethod.COBYLA
        )

        def track_best(result: OptimizeResult) -> None:
            nonlocal best_loss, best_params, iterations
            index = argmin_finite(result.fun)
            if index is not None and np.ravel(result.fun)[index] < best_loss:
                best_loss = float(np.ravel(result.fun)[index])
                best_params = np.atleast_2d(result.x)[index].copy()
            iterations += 1
            if stops_by_callback and iterations >= self.max_iterations:
                raise StopIteration

        center = self._build_initial_point(vqe, root)
        evaluators = {
            key: evaluator
            for key, evaluator in optimizer.build_evaluators(vqe).items()
            if key != "jac"
        }
        result = optimizer.optimize(
            cost_fn,
            initial_params=vqe.parameter_space.build_population(
                center, optimizer.n_param_sets, np.random.default_rng(42 + root)
            ),
            callback_fn=track_best,
            max_iterations=self.max_iterations,
            rng=np.random.default_rng(42 + root),
            **evaluators,
        )
        index = argmin_finite(result.fun)
        if index is not None and np.ravel(result.fun)[index] < best_loss:
            best_params = np.atleast_2d(result.x)[index].copy()
        success = bool(result.get("success", True))
        params = center if best_params is None else best_params
        energy = float(compute_energies(params[np.newaxis])[0])
        statevector = self._build_statevector(vqe, reference, params)
        return _OptimizedState(root, energy, params, statevector, vqe, success)

    def _build_statevector(
        self, vqe: VQE, reference: ElectronCounts, params: np.ndarray
    ) -> np.ndarray:
        """Amplitudes of ``vqe``'s initial state and ansatz at ``params``."""
        circuit = QuantumCircuit(vqe.n_qubits)
        circuit.compose(
            vqe.initial_state.build(list(range(vqe.n_qubits))), inplace=True
        )
        circuit.compose(
            vqe.ansatz.build(
                params, n_qubits=vqe.n_qubits, n_layers=vqe.n_layers, reference=reference
            ),
            inplace=True,
        )
        return self.backend.statevector(circuit)

    def kernel(
        self,
        h1e: npt.ArrayLike,
        eri: npt.ArrayLike,
        norb: int,
        nelec: int | tuple[int, int],
        ci0: Any = None,
        ecore: float = 0.0,
        **kwargs: Any,
    ) -> tuple[np.ndarray, list["VQDSolver"]]:
        """Optimise each root in turn.

        Returns:
            The energy of every root, and ``[self] * n_roots`` as PySCF's CI
            vectors.
        """
        if np.ndim(h1e) != 2:
            raise NotImplementedError(
                "VQDSolver needs restricted integrals; got separate alpha and beta ones."
            )
        if isinstance(nelec, (int, np.integer)):
            nelec = (nelec - nelec // 2, nelec // 2)
        problem = MolecularProblem(
            np.asarray(h1e),
            ao2mo.restore(1, np.asarray(eri), norb),
            n_alpha=int(nelec[0]),
            n_beta=int(nelec[1]),
            constant=ecore,
        )
        weights = self._resolve_penalty_weights()
        self.optimized_states = []
        for root in range(self.n_roots):
            self.optimized_states.append(self._optimize_root(problem, root, weights))

        self.energies = np.array([state.energy for state in self.optimized_states])
        self.converged = all(state.success for state in self.optimized_states)
        self._active_root = len(self.optimized_states) - 1
        return self.energies, [self] * len(self.optimized_states)

    approx_kernel = kernel

    def _require_root(self, root: int) -> _OptimizedState:
        if not 0 <= root < len(self.optimized_states):
            raise IndexError(
                f"Root {root} not available. "
                f"Only {len(self.optimized_states)} states computed."
            )
        return self.optimized_states[root]

    def get_state(self, root: int) -> dict:
        """``root``, ``energy`` and ``params`` of one optimised root."""
        state = self._require_root(root)
        return {"root": state.root, "energy": state.energy, "params": state.params.copy()}

    def get_statevector(self, root: int) -> np.ndarray:
        """Amplitudes of one optimised root, qubit 0 the least-significant bit."""
        return self._require_root(root).statevector.copy()

    def set_active_root(self, root: int) -> None:
        """Pick the root the RDM methods and ``spin_square`` act on."""
        self._require_root(root)
        self._active_root = root

    def make_rdm12s(self, ci: Any, norb: int, nelec: int | tuple[int, int]) -> tuple:
        """Spin RDMs of the active root, in :meth:`VQE.measure_rdms`'s layout."""
        state = self._require_root(self._active_root)
        if state.rdms is None:
            state.rdms = state.vqe.measure_rdms(state.params)
        return copy.deepcopy(state.rdms)
