# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from pyscf import mcscf

from divi.qprog.algorithms import UCCSDAnsatz, VQEFCISolver
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from qoro_pyscf import QSCISolver
from qoro_pyscf.qsci_solver import _select_determinants
from tests._helpers import LIH, run_mean_field

PROBABILITIES = {"1100": 0.5, "0011": 0.3, "1000": 0.15, "1001": 0.05, "0110": 1e-9}


@pytest.mark.parametrize(
    "bits, nelec, alpha, beta",
    [
        ("1100", (1, 1), 1, 1),
        ("0011", (1, 1), 2, 2),
        ("101010", (3, 0), 7, 0),
        ("011000", (1, 1), 2, 1),
    ],
)
def test_bitstring_splits_into_alpha_and_beta_strings(bits, nelec, alpha, beta):
    alphas, betas, _ = _select_determinants({bits: 1.0}, nelec, 10, 0.0)
    assert (list(alphas), list(betas)) == ([alpha], [beta])


def test_selection_drops_wrong_electron_counts_and_low_probabilities():
    alphas, betas, kept = _select_determinants(PROBABILITIES, (1, 1), 10, 1e-8)
    np.testing.assert_allclose(kept, [0.5, 0.3, 0.05])
    assert (list(alphas), list(betas)) == ([1, 2], [1, 2])


def test_selection_keeps_the_most_probable_bitstrings():
    alphas, betas, kept = _select_determinants(PROBABILITIES, (1, 1), 1, 1e-8)
    np.testing.assert_allclose(kept, [0.5])
    assert (list(alphas), list(betas)) == ([1], [1])


@pytest.mark.parametrize(
    "probabilities, match",
    [({"1000": 1.0}, "n_alpha"), ({"1100": 1e-12}, "above probability")],
)
def test_selection_without_valid_bitstrings_raises(probabilities, match):
    with pytest.raises(ValueError, match=match):
        _select_determinants(probabilities, (1, 1), 10, 1e-8)


@pytest.fixture(scope="module")
def lih_mean_field():
    return run_mean_field(LIH)


def run_qsci(backend, mean_field, n_samples: int) -> mcscf.casci.CASCI:
    """CAS(2, 4) LiH with a QSCI solver selecting ``n_samples`` bitstrings."""
    casci = mcscf.CASCI(mean_field, 4, 2)
    casci.fcisolver = QSCISolver(
        VQEFCISolver(
            backend=backend,
            optimizer=ScipyOptimizer(ScipyMethod.L_BFGS_B),
            ansatz=UCCSDAnsatz(),
            max_iterations=50,
        ),
        n_samples=n_samples,
    )
    casci.kernel()
    return casci


def test_full_selection_reproduces_casci(backend, lih_mean_field):
    exact = mcscf.CASCI(lih_mean_field, 4, 2).kernel()[0]
    casci = run_qsci(backend, lih_mean_field, n_samples=50)
    solver = casci.fcisolver
    assert casci.e_tot == pytest.approx(exact, abs=1e-8)
    assert solver.vqe_energy >= exact - 1e-8
    assert solver.n_determinants == 16
    assert solver.converged
    dm1a, dm1b = solver.make_rdm1s(casci.ci, 4, 2)
    np.testing.assert_allclose(solver.make_rdm1(casci.ci, 4, 2), dm1a + dm1b)
    assert np.trace(dm1a) == pytest.approx(1.0)
    assert solver.spin_square(casci.ci, 4, 2)[0] == pytest.approx(0.0, abs=1e-8)
    assert solver.approx_kernel(None, None, 4, 2) == (solver.qsci_energy, solver)


def test_energy_is_variational_in_the_selected_subspace(backend, lih_mean_field):
    casci = run_qsci(backend, lih_mean_field, n_samples=1)
    assert casci.fcisolver.n_determinants == 1
    assert casci.e_tot == pytest.approx(lih_mean_field.e_tot, abs=1e-8)
