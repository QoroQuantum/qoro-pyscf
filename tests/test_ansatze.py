# Copyright 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ansatz building, parameter counting, and excitation enumeration."""

import numpy as np
import pytest


def _has_maestro() -> bool:
    try:
        import maestro  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


class TestAnsatzeUnit:
    """Test ansatz parameter counting and excitation enumeration."""

    def test_hardware_efficient_param_count(self):
        from qoro_maestro_pyscf.ansatze import hardware_efficient_param_count
        assert hardware_efficient_param_count(4, 2) == 16
        assert hardware_efficient_param_count(4, 1) == 8
        assert hardware_efficient_param_count(2, 3) == 12

    def test_uccsd_param_count_tuple(self):
        from qoro_maestro_pyscf.ansatze import uccsd_param_count
        assert uccsd_param_count(4, (1, 1)) == 5

    def test_uccsd_param_count_int(self):
        from qoro_maestro_pyscf.ansatze import uccsd_param_count
        assert uccsd_param_count(4, 2) == uccsd_param_count(4, (1, 1))

    def test_excitation_enumeration(self):
        from qoro_maestro_pyscf.ansatze import _get_uccsd_excitations
        singles, doubles = _get_uccsd_excitations(4, (1, 1))
        for i, a in singles:
            assert i in [0, 1]  # occupied
            assert a in [2, 3]  # virtual

    def test_excitation_counts_larger(self):
        """6 qubits, (2, 1) electrons → more excitations."""
        from qoro_maestro_pyscf.ansatze import _get_uccsd_excitations
        singles, doubles = _get_uccsd_excitations(6, (2, 1))
        assert len(singles) > 0
        assert len(doubles) > 0
        occ = {0, 2, 1}  # α: 0,2; β: 1
        vir = {4, 3, 5}  # α: 4; β: 3,5
        for i, a in singles:
            assert i in occ
            assert a in vir

    @pytest.mark.skipif(not _has_maestro(), reason="Requires maestro native library")
    def test_hf_circuit_builds(self):
        from qoro_maestro_pyscf.ansatze import hartree_fock_circuit
        qc = hartree_fock_circuit(4, (1, 1))
        assert qc is not None

    @pytest.mark.skipif(not _has_maestro(), reason="Requires maestro native library")
    def test_hf_circuit_int_nelec(self):
        from qoro_maestro_pyscf.ansatze import hartree_fock_circuit
        qc = hartree_fock_circuit(4, 2)
        assert qc is not None

    @pytest.mark.skipif(not _has_maestro(), reason="Requires maestro native library")
    def test_hardware_efficient_builds(self):
        from qoro_maestro_pyscf.ansatze import (
            hardware_efficient_ansatz, hardware_efficient_param_count,
        )
        n_params = hardware_efficient_param_count(4, 2)
        params = np.random.rand(n_params)
        qc = hardware_efficient_ansatz(params, 4, 2)
        assert qc is not None

    @pytest.mark.skipif(not _has_maestro(), reason="Requires maestro native library")
    def test_hardware_efficient_with_hf(self):
        from qoro_maestro_pyscf.ansatze import (
            hardware_efficient_ansatz, hardware_efficient_param_count,
        )
        n_params = hardware_efficient_param_count(4, 2)
        params = np.random.rand(n_params)
        qc = hardware_efficient_ansatz(
            params, 4, 2, include_hf=True, nelec=(1, 1)
        )
        assert qc is not None

    def test_hardware_efficient_hf_requires_nelec(self):
        from qoro_maestro_pyscf.ansatze import hardware_efficient_ansatz
        with pytest.raises(ValueError, match="nelec"):
            hardware_efficient_ansatz(np.zeros(16), 4, 2, include_hf=True)

    @pytest.mark.skipif(not _has_maestro(), reason="Requires maestro native library")
    def test_uccsd_builds(self):
        from qoro_maestro_pyscf.ansatze import uccsd_ansatz, uccsd_param_count
        n_params = uccsd_param_count(4, (1, 1))
        qc = uccsd_ansatz(np.zeros(n_params), 4, (1, 1))
        assert qc is not None

    @pytest.mark.skipif(not _has_maestro(), reason="Requires maestro native library")
    def test_uccsd_with_int_nelec(self):
        from qoro_maestro_pyscf.ansatze import uccsd_ansatz, uccsd_param_count
        n_params = uccsd_param_count(4, 2)
        qc = uccsd_ansatz(np.zeros(n_params), 4, 2)
        assert qc is not None


class TestUpCCDUnit:
    """Test UpCCD ansatz parameter counting and excitation enumeration."""

    def test_upccd_param_count(self):
        from qoro_maestro_pyscf.ansatze import upccd_param_count
        assert upccd_param_count(4, (1, 1)) == 1

    def test_upccd_param_count_larger(self):
        from qoro_maestro_pyscf.ansatze import upccd_param_count
        assert upccd_param_count(6, (2, 2)) == 2

    def test_upccd_param_count_int(self):
        from qoro_maestro_pyscf.ansatze import upccd_param_count
        assert upccd_param_count(4, 2) == upccd_param_count(4, (1, 1))

    def test_upccd_fewer_than_uccsd(self):
        """UpCCD should always have fewer params than UCCSD."""
        from qoro_maestro_pyscf.ansatze import uccsd_param_count, upccd_param_count
        for n_qubits, nelec in [(4, (1, 1)), (6, (2, 2)), (8, (3, 3))]:
            assert upccd_param_count(n_qubits, nelec) < uccsd_param_count(n_qubits, nelec)

    def test_upccd_excitations_are_paired(self):
        """All excitations should be between doubly-occupied and empty spatial orbitals."""
        from qoro_maestro_pyscf.ansatze import _get_upccd_excitations
        pairs = _get_upccd_excitations(8, (2, 2))
        n_spatial = 4
        for i, a in pairs:
            assert 0 <= i < 2  # doubly-occupied spatial orbitals
            assert 2 <= a < n_spatial  # virtual spatial orbitals

    @pytest.mark.skipif(not _has_maestro(), reason="Requires maestro native library")
    def test_upccd_builds(self):
        from qoro_maestro_pyscf.ansatze import upccd_ansatz, upccd_param_count
        n_params = upccd_param_count(4, (1, 1))
        qc = upccd_ansatz(np.zeros(n_params), 4, (1, 1))
        assert qc is not None

    @pytest.mark.skipif(not _has_maestro(), reason="Requires maestro native library")
    def test_upccd_with_int_nelec(self):
        from qoro_maestro_pyscf.ansatze import upccd_ansatz, upccd_param_count
        n_params = upccd_param_count(4, 2)
        qc = upccd_ansatz(np.zeros(n_params), 4, 2)
        assert qc is not None


class TestAdaptUnit:
    """Test ADAPT-VQE operator pool and convergence logic."""

    def test_operator_pool_sd(self):
        from qoro_maestro_pyscf.adapt import build_operator_pool
        ops = build_operator_pool(4, (1, 1), pool="sd")
        singles = [o for o in ops if o.kind == "single"]
        doubles = [o for o in ops if o.kind == "double"]
        assert len(singles) == 4
        assert len(doubles) == 1
        assert len(ops) == 5

    def test_operator_pool_d_only(self):
        from qoro_maestro_pyscf.adapt import build_operator_pool
        ops = build_operator_pool(4, (1, 1), pool="d")
        assert all(o.kind == "double" for o in ops)
        assert len(ops) == 1

    def test_operator_pool_larger(self):
        from qoro_maestro_pyscf.adapt import build_operator_pool
        ops = build_operator_pool(8, (2, 2), pool="sd")
        singles = [o for o in ops if o.kind == "single"]
        doubles = [o for o in ops if o.kind == "double"]
        assert len(singles) > 0
        assert len(doubles) > 0
        assert len(ops) == len(singles) + len(doubles)

    @pytest.mark.skipif(not _has_maestro(), reason="Requires maestro native library")
    def test_adapt_builds_circuit(self):
        from qoro_maestro_pyscf.adapt import Operator, _build_adapt_circuit
        ops = [Operator(kind="double", indices=(0, 1, 2, 3))]
        qc = _build_adapt_circuit(4, (1, 1), ops, np.array([0.5]))
        assert qc is not None
