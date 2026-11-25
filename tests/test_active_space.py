# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pyscf import gto, scf

from qoro_pyscf import suggest_active_space, suggest_active_space_from_mp2


@pytest.fixture(scope="module")
def water() -> scf.hf.RHF:
    mol = gto.M(
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587", basis="sto-3g", verbose=0
    )
    return scf.RHF(mol).run()


def test_mp2_active_space_keeps_every_orbital(water):
    norb, nelec, mo = suggest_active_space_from_mp2(water)
    assert mo.shape == water.mo_coeff.shape
    assert nelec[0] == nelec[1]
    assert (water.mol.nelectron - sum(nelec)) // 2 + norb <= mo.shape[1]


@pytest.mark.parametrize("max_orbitals", [3, 2])
def test_mp2_orbital_cap_binds_and_keeps_every_orbital(water, max_orbitals):
    uncapped, _, _ = suggest_active_space_from_mp2(water)
    norb, nelec, mo = suggest_active_space_from_mp2(water, max_orbitals=max_orbitals)
    assert uncapped > max_orbitals
    assert norb == max_orbitals
    assert mo.shape == water.mo_coeff.shape
    assert nelec[0] == nelec[1]


def test_avas_on_h2():
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    norb, nelec, mo = suggest_active_space(mf, "H 1s", verbose=0)
    assert (norb, nelec) == (2, (1, 1))
    assert mo.shape[0] == mol.nao


def test_avas_splits_open_shell_electrons_by_spin():
    mol = gto.M(atom="O 0 0 0; O 0 0 1.2", basis="sto-3g", spin=2, verbose=0)
    mf = scf.ROHF(mol).run()
    _, (n_alpha, n_beta), _ = suggest_active_space(mf, "O 2p", verbose=0)
    assert n_alpha - n_beta == 2
