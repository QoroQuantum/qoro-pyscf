# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

import maestro
import numpy as np
import pytest
from maestro.circuits import QuantumCircuit
from openfermion import (
    FermionOperator,
    get_sparse_operator,
    hermitian_conjugated,
    jordan_wigner,
)
from pyscf import mcscf
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm

from divi.qprog.problems import MolecularProblem
from qoro_pyscf import build_pauli_terms, configure_backend, run_adapt_vqe
from qoro_pyscf.adapt import Operator, build_operator_pool
from tests._helpers import LIH, compute_exact_roots, run_mean_field


def test_pool_holds_only_spin_conserving_excitations():
    pool = build_operator_pool(6, (2, 1), "sd")
    assert [op.indices for op in pool if op.kind == "single"] == [
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 4),
    ]
    assert [op.indices for op in pool if op.kind == "double"] == [
        (0, 1, 3, 4),
        (0, 1, 4, 5),
        (1, 2, 3, 4),
        (1, 2, 4, 5),
    ]


@pytest.mark.parametrize(
    "kind, indices, term",
    [
        ("single", (0, 3), "3^ 0"),
        ("single", (3, 1), "1^ 3"),
        ("single", (4, 0), "0^ 4"),
        ("double", (0, 1, 2, 3), "2^ 3^ 1 0"),
        ("double", (0, 3, 1, 4), "1^ 4^ 3 0"),
        ("double", (2, 4, 0, 1), "0^ 1^ 4 2"),
    ],
)
def test_excitation_is_the_jordan_wigner_exponential(kind, indices, term):
    n_qubits, theta = 5, 0.37
    excitation = FermionOperator(term)
    generator = excitation - hermitian_conjugated(excitation)
    jordan_wigner_matrix = get_sparse_operator(jordan_wigner(generator), n_qubits)
    unitary = expm(theta * jordan_wigner_matrix.toarray())
    little_endian = [
        int(format(index, f"0{n_qubits}b")[::-1], 2) for index in range(2**n_qubits)
    ]
    phase, kick = np.exp(-0.05j * n_qubits), 0.1
    for index in range(2**n_qubits):
        circuit = QuantumCircuit()
        for qubit in range(n_qubits):
            circuit.rz(qubit, kick)
            if (index >> qubit) & 1:
                circuit.x(qubit)
        Operator(kind, indices).apply(circuit, theta)
        state = maestro.get_statevector(circuit, config=maestro.SimulatorConfig())
        expected = phase * unitary[:, little_endian[index]][little_endian]
        np.testing.assert_allclose(state, expected, atol=1e-12)


def test_doubles_pool_drops_singles():
    pool = build_operator_pool(8, (2, 2), "d")
    assert {op.kind for op in pool} == {"double"}
    assert len(pool) == 1 + 1 + 4 * 4


def test_pauli_terms_split_identity_and_reverse_labels():
    hamiltonian = SparsePauliOp(["II", "ZI", "XY"], [0.5, 1.0, 2.0])
    offset, labels, coeffs = build_pauli_terms(hamiltonian)
    assert offset == 0.5
    assert labels == ["IZ", "YX"]
    np.testing.assert_allclose(coeffs, [1.0, 2.0])


@pytest.fixture(scope="module")
def lih() -> tuple[float, float, tuple]:
    """CAS(2, 4) LiH: Hartree-Fock and exact energies, and ``run_adapt_vqe``'s
    Hamiltonian arguments."""
    mean_field = run_mean_field(LIH)
    exact = compute_exact_roots(mcscf.CASCI(mean_field, 4, 2), 1)[0]
    problem = MolecularProblem.from_casci(mean_field, 4, 2)
    return mean_field.e_tot, exact, build_pauli_terms(problem.hamiltonian)


def run_lih_adapt(lih: tuple, **options) -> dict:
    _, _, terms = lih
    return run_adapt_vqe(8, (1, 1), *terms, configure_backend(), **options)


@pytest.mark.parametrize(
    "options",
    [{}, {"optimizer": "ROTOSOLVE"}, {"greedy": True}],
    ids=["cobyla", "rotosolve", "greedy"],
)
def test_adapt_reaches_the_exact_lih_energy(lih, options):
    hartree_fock, exact, _ = lih
    result = run_lih_adapt(lih, **options)
    assert result["converged"]
    assert result["energy"] == pytest.approx(exact, abs=1e-4)
    assert result["energy_history"][0] == pytest.approx(hartree_fock, abs=1e-8)
    assert result["n_operators"] == len(result["params"]) == len(result["selected_ops"])


def test_operator_cap_leaves_the_run_unconverged(lih):
    hartree_fock, _, _ = lih
    result = run_lih_adapt(lih, max_operators=1)
    assert not result["converged"]
    assert result["n_operators"] == 1
    assert result["energy"] < hartree_fock
