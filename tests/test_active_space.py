# Copyright 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Tests for active space selection helpers."""

import numpy as np
import pytest


class TestActiveSpaceUnit:
    """Tests for active_space.py."""

    def test_suggest_active_space_import(self):
        from qoro_pyscf import suggest_active_space
        assert callable(suggest_active_space)

    def test_suggest_active_space_from_mp2_import(self):
        from qoro_pyscf import suggest_active_space_from_mp2
        assert callable(suggest_active_space_from_mp2)

    def test_mp2_active_space_on_lih(self):
        """MP2 natural orbital selection gives a reasonable active space for LiH."""
        from pyscf import gto, scf
        from qoro_pyscf.active_space import suggest_active_space_from_mp2

        mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run()

        norb, nelec, mo = suggest_active_space_from_mp2(mf, threshold=0.02)

        assert norb >= 1, "Should select at least 1 active orbital"
        assert sum(nelec) >= 1, "Should have at least 1 active electron"
        assert mo.shape[0] == mol.nao, "MO coeffs should have nao rows"
        assert mo.shape[1] >= norb, "MO coeffs should have at least norb columns"
        assert nelec[0] >= nelec[1], "n_alpha >= n_beta for closed-shell"

    def test_mp2_max_orbitals_cap(self):
        """max_orbitals caps the active space size."""
        from pyscf import gto, scf
        from qoro_pyscf.active_space import suggest_active_space_from_mp2

        mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run()

        norb, _, _ = suggest_active_space_from_mp2(mf, threshold=0.001, max_orbitals=2)
        assert norb <= 2

    def test_avas_on_h2(self):
        """AVAS with 1s labels gives a CAS for H₂."""
        from pyscf import gto, scf
        from qoro_pyscf.active_space import suggest_active_space

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run()

        norb, nelec, mo = suggest_active_space(mf, "H 1s", verbose=0)

        assert norb >= 1
        assert sum(nelec) >= 1
        assert mo.shape[0] == mol.nao
