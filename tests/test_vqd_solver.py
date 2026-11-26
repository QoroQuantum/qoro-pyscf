# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from pyscf import ao2mo, mcscf

from divi.qprog.algorithms import VQE, UCCSDAnsatz
from divi.qprog.optimizers import RotosolveOptimizer, ScipyMethod, ScipyOptimizer
from qoro_pyscf import VQDSolver
from tests._helpers import H2, compute_exact_roots, run_mean_field


def make_vqd(backend, **kwargs) -> VQDSolver:
    options = {
        "optimizer": ScipyOptimizer(ScipyMethod.COBYLA),
        "ansatz": UCCSDAnsatz(),
        "max_iterations": 100,
    }
    return VQDSolver(backend=backend, **options | kwargs)


@pytest.mark.parametrize(
    "weights, num_states, nroots, expected",
    [
        (5.0, 3, 1, [5.0, 5.0]),
        (np.int64(5), 2, 1, [5.0]),
        ([1.0, 2.0], 3, 1, [1.0, 2.0]),
        ([1.0], 4, 1, [1.0, 1.0, 1.0]),
        ([1.0, 2.0, 3.0], 2, 1, [1.0]),
        (5.0, 1, 1, []),
        (5.0, 1, 3, [5.0, 5.0]),
    ],
)
def test_penalty_weights_resolve_to_one_per_earlier_root(
    backend, weights, num_states, nroots, expected
):
    solver = make_vqd(backend, penalty_weights=weights, num_states=num_states)
    solver.nroots = nroots
    assert solver._resolve_penalty_weights() == expected


@pytest.mark.parametrize("method", ["get_state", "get_statevector", "set_active_root"])
def test_state_access_before_kernel_raises(backend, method):
    with pytest.raises(IndexError, match="Root 0 not available"):
        getattr(make_vqd(backend), method)(0)


def test_rejects_unrestricted_integrals(backend):
    with pytest.raises(NotImplementedError, match="restricted"):
        make_vqd(backend).kernel(np.zeros((2, 2, 2)), np.zeros((2, 2, 2, 2)), 2, 2)


@pytest.fixture(scope="module")
def h2_active_space():
    """H2 CAS(2, 2): one-body integrals, core energy, full ERIs and exact roots."""
    casci = mcscf.CASCI(run_mean_field(H2), 2, 2)
    h1, ecore = casci.get_h1eff()
    eri = ao2mo.restore(1, casci.get_h2eff(), 2)
    return h1, ecore, eri, compute_exact_roots(casci, 4)


def compute_rdm_energy(solver: VQDSolver, h1, ecore, eri) -> float:
    dm1, dm2 = solver.make_rdm12(solver, 2, (1, 1))
    return ecore + np.einsum("pq,pq", h1, dm1) + 0.5 * np.einsum("pqrs,pqrs", eri, dm2)


@pytest.mark.parametrize(
    "optimizer",
    [ScipyOptimizer(ScipyMethod.COBYLA), RotosolveOptimizer()],
    ids=["cobyla", "rotosolve"],
)
def test_ground_root_is_exact_and_excited_root_deflated(
    backend, h2_active_space, optimizer
):
    h1, ecore, eri, exact = h2_active_space
    calls = []
    solver = make_vqd(
        backend,
        optimizer=optimizer,
        callback=lambda root, evaluation, cost, params: calls.append(root),
    )
    energies, ci = solver.kernel(h1, eri, 2, (1, 1), ecore=ecore)

    assert ci == [solver, solver]
    assert energies[0] == pytest.approx(exact[0], abs=1e-4)
    assert exact[0] + 0.1 < energies[1] <= exact[-1]
    assert set(calls) == {0, 1}
    ground, excited = solver.get_statevector(0), solver.get_statevector(1)
    assert np.linalg.norm(ground) == pytest.approx(1.0)
    assert abs(np.vdot(ground, excited)) ** 2 < 1e-2


def test_rdms_follow_the_active_root(backend, h2_active_space, mocker):
    h1, ecore, eri, exact = h2_active_space
    solver = make_vqd(backend)
    energies, _ = solver.kernel(h1, eri, 2, 2, ecore=ecore)
    spy = mocker.spy(VQE, "measure_rdms")

    assert compute_rdm_energy(solver, h1, ecore, eri) == pytest.approx(
        energies[1], abs=1e-4
    )
    solver.set_active_root(0)
    assert compute_rdm_energy(solver, h1, ecore, eri) == pytest.approx(
        exact[0], abs=1e-4
    )
    assert solver.spin_square(solver, 2, (1, 1))[0] == pytest.approx(0.0, abs=1e-4)
    first = solver.make_rdm12s(solver, 2, (1, 1))
    first[0][0][:] = 0.0
    assert np.trace(solver.make_rdm1(solver, 2, (1, 1))) == pytest.approx(2.0)
    assert spy.call_count == 2
