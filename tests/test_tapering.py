# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from openfermion import QubitOperator, get_sparse_operator
from pyscf import mcscf

from divi.qprog.problems import MolecularProblem
from qoro_pyscf import taper_hamiltonian
from qoro_pyscf.tapering import find_z2_symmetries
from tests._helpers import H2, compute_exact_roots, run_mean_field


@pytest.mark.parametrize(
    "nelec, alpha_sign, beta_sign", [((1, 1), -1, -1), ((2, 1), 1, -1)]
)
def test_find_z2_symmetries_returns_spin_parities(nelec, alpha_sign, beta_sign):
    stabilizers = find_z2_symmetries(n_qubits=4, nelec=nelec)
    assert stabilizers == [
        alpha_sign * QubitOperator("Z0 Z2"),
        beta_sign * QubitOperator("Z1 Z3"),
    ]


def test_tapering_keeps_the_h2_ground_state():
    mean_field = run_mean_field(H2)
    problem = MolecularProblem.from_casci(mean_field, 2, 2)
    hamiltonian = sum(
        (
            QubitOperator(
                " ".join(f"{p}{q}" for q, p in enumerate(label[::-1]) if p != "I"),
                coeff.real,
            )
            for label, coeff in zip(
                problem.hamiltonian.paulis.to_labels(), problem.hamiltonian.coeffs
            )
        ),
        QubitOperator(),
    )
    result = taper_hamiltonian(hamiltonian, n_qubits=4, nelec=(1, 1))
    assert result.tapered_n_qubits == 4 - len(result.removed_positions) == 2
    lowest = np.linalg.eigvalsh(get_sparse_operator(result.tapered_op, 2).toarray())[0]
    exact = compute_exact_roots(mcscf.CASCI(mean_field, 2, 2), 1)[0]
    assert lowest == pytest.approx(exact, abs=1e-10)


def test_custom_stabilizers_replace_the_defaults():
    hamiltonian = QubitOperator("Z0", 1.0) + QubitOperator("", -0.5)
    result = taper_hamiltonian(
        hamiltonian, n_qubits=2, nelec=(1, 0), stabilizers=[QubitOperator("Z0")]
    )
    assert result.stabilizers == [QubitOperator("Z0")]
    assert result.tapered_n_qubits < 2
