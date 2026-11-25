# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from pyscf import gto, mcscf, scf

from qoro_pyscf import (
    compute_dipole_moment,
    compute_mulliken_spin_population,
    compute_natural_orbitals,
)


def test_natural_orbitals_sorted_descending_and_trace_preserving():
    rdm1 = np.array([[0.1, 0.05, 0.0], [0.05, 1.9, 0.0], [0.0, 0.0, 0.5]])
    occupations, orbitals = compute_natural_orbitals(rdm1)
    assert list(occupations) == sorted(occupations, reverse=True)
    assert occupations.sum() == pytest.approx(np.trace(rdm1))
    np.testing.assert_allclose(orbitals @ np.diag(occupations) @ orbitals.T, rdm1)


def test_dipole_matches_pyscf():
    mol = gto.M(
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587", basis="sto-3g", verbose=0
    )
    casci = mcscf.CASCI(scf.RHF(mol).run(), 4, 4)
    casci.kernel()
    rdm1 = casci.fcisolver.make_rdm1(casci.ci, 4, 4)
    dipole, magnitude = compute_dipole_moment(mol, casci.mo_coeff, rdm1, casci.ncore)
    reference = scf.hf.dip_moment(mol, casci.make_rdm1(), verbose=0)
    np.testing.assert_allclose(dipole, reference, atol=1e-6)
    assert magnitude == pytest.approx(np.linalg.norm(reference), abs=1e-6)


def test_open_shell_spin_population_matches_pyscf():
    mol = gto.M(atom="N 0 0 0; H 0 0 1.03; H 0.99 0 -0.29", basis="sto-3g", spin=1)
    mol.verbose = 0
    casci = mcscf.CASCI(scf.ROHF(mol).run(), 3, (2, 1))
    casci.kernel()
    rdm1a, rdm1b = casci.fcisolver.make_rdm1s(casci.ci, 3, (2, 1))
    population = compute_mulliken_spin_population(
        mol, casci.mo_coeff, rdm1a, rdm1b, casci.ncore
    )
    _, reference = scf.uhf.mulliken_spin_pop(mol, casci.make_rdm1s(), verbose=0)
    np.testing.assert_allclose(population, reference, atol=1e-8)
    assert population.sum() == pytest.approx(1.0)
