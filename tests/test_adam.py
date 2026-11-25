# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from pyscf import mcscf

from divi.qprog.algorithms import VQE, UCCSDAnsatz
from divi.qprog.problems import MolecularProblem
from qoro_pyscf import AdamOptimizer
from tests._helpers import H2, compute_exact_roots, run_mean_field


def quadratic(params: np.ndarray) -> float | np.ndarray:
    values = np.sum((np.atleast_2d(params) - 1.0) ** 2, axis=1)
    return values if np.ndim(params) > 1 else float(values[0])


def test_minimises_a_quadratic():
    result = AdamOptimizer(learning_rate=0.05).optimize(
        quadratic, initial_params=np.zeros(4), max_iterations=200
    )
    assert result.fun < 0.01
    np.testing.assert_allclose(result.x, np.ones(4), atol=0.1)


def test_first_step_moves_each_parameter_by_the_learning_rate():
    iterates = []
    AdamOptimizer(learning_rate=0.1).optimize(
        quadratic,
        initial_params=np.zeros(3),
        callback_fn=lambda r: iterates.append(r.x[0]),
        max_iterations=1,
    )
    np.testing.assert_allclose(iterates[0], np.full(3, 0.1), rtol=1e-6)


def test_returns_the_best_iterate_not_the_last():
    losses, iterates = [], []

    def record(result):
        losses.append(float(result.fun[0]))
        iterates.append(result.x[0])

    result = AdamOptimizer(learning_rate=0.5).optimize(
        quadratic, initial_params=np.zeros(2), callback_fn=record, max_iterations=30
    )
    best = int(np.argmin(losses))
    assert best < len(losses) - 1
    assert result.fun == pytest.approx(losses[best])
    np.testing.assert_array_equal(result.x, iterates[best])


def test_requires_initial_params():
    with pytest.raises(ValueError, match="initial_params"):
        AdamOptimizer().optimize(quadratic)


def test_rejects_checkpointing():
    optimizer = AdamOptimizer()
    assert not optimizer.supports_checkpointing
    with pytest.raises(NotImplementedError):
        optimizer.save_state("unused")
    with pytest.raises(NotImplementedError):
        AdamOptimizer.load_state("unused")


def test_trains_a_divi_vqe(backend):
    mean_field = run_mean_field(H2)
    vqe = VQE(
        MolecularProblem.from_casci(mean_field, 2, 2),
        ansatz=UCCSDAnsatz(),
        optimizer=AdamOptimizer(learning_rate=0.05),
        max_iterations=100,
        backend=backend,
    )
    vqe.run(perform_final_computation=False)
    exact = compute_exact_roots(mcscf.CASCI(mean_field, 2, 2), 1)[0]
    assert vqe.best_loss == pytest.approx(exact, abs=1e-4)
