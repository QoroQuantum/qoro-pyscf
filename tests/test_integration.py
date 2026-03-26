# Copyright 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Hamiltonian construction, RDM tracing, properties, and package exports.

These are the remaining tests from the original monolith that don't fit into
the more specific test files.
"""

import numpy as np
import pytest


def _has_maestro() -> bool:
    try:
        import maestro  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Backends
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackendsUnit:
    """Test backend configuration logic."""

    def test_simulation_mode_enum(self):
        from qoro_pyscf.backends import SimulationMode
        assert hasattr(SimulationMode, "STATEVECTOR") or "statevector" in dir(SimulationMode)

    def test_simulation_mode_invalid(self):
        from qoro_pyscf.backends import configure_backend
        with pytest.raises((ValueError, KeyError)):
            configure_backend(use_gpu=False, simulation="invalid_mode")

    def test_set_license_key(self):
        import os
        from qoro_pyscf.backends import set_license_key
        set_license_key("TEST-KEY-123")
        assert os.environ.get("MAESTRO_LICENSE_KEY") == "TEST-KEY-123"

    def test_set_license_key_overwrite(self):
        import os
        from qoro_pyscf.backends import set_license_key
        set_license_key("FIRST")
        set_license_key("SECOND")
        assert os.environ.get("MAESTRO_LICENSE_KEY") == "SECOND"


# ═══════════════════════════════════════════════════════════════════════════════
# RDM tracing
# ═══════════════════════════════════════════════════════════════════════════════

class TestRdmUnit:
    """Test RDM tracing functions."""

    def test_trace_spin_rdm1(self):
        from qoro_pyscf.rdm import trace_spin_rdm1
        rdm1_a = np.eye(2) * 0.5
        rdm1_b = np.eye(2) * 0.3
        result = trace_spin_rdm1(rdm1_a, rdm1_b)
        np.testing.assert_array_almost_equal(result, np.eye(2) * 0.8)

    def test_trace_spin_rdm2(self):
        from qoro_pyscf.rdm import trace_spin_rdm2
        n = 2
        rdm2_aa = np.random.rand(n, n, n, n)
        rdm2_ab = np.random.rand(n, n, n, n)
        rdm2_bb = np.random.rand(n, n, n, n)
        result = trace_spin_rdm2(rdm2_aa, rdm2_ab, rdm2_bb)
        assert result.shape == (n, n, n, n)

    def test_trace_rdm1_shape(self):
        from qoro_pyscf.rdm import trace_spin_rdm1
        n = 3
        rdm1_a = np.random.rand(n, n)
        rdm1_b = np.random.rand(n, n)
        result = trace_spin_rdm1(rdm1_a, rdm1_b)
        assert result.shape == (n, n)

    def test_trace_rdm2_symmetry(self):
        """rdm2_ba should be the transpose of rdm2_ab."""
        from qoro_pyscf.rdm import trace_spin_rdm2
        n = 2
        rdm2_aa = np.zeros((n, n, n, n))
        rdm2_bb = np.zeros((n, n, n, n))
        rdm2_ab = np.random.rand(n, n, n, n)
        result = trace_spin_rdm2(rdm2_aa, rdm2_ab, rdm2_bb)
        rdm2_ba = rdm2_ab.transpose(2, 3, 0, 1)
        expected = rdm2_ab + rdm2_ba
        np.testing.assert_array_almost_equal(result, expected)


# ═══════════════════════════════════════════════════════════════════════════════
# Properties
# ═══════════════════════════════════════════════════════════════════════════════

class TestPropertiesUnit:
    """Test molecular property computation functions."""

    def test_natural_orbitals_identity(self):
        """Identity 1-RDM → occupations are all 1."""
        from qoro_pyscf.properties import compute_natural_orbitals
        rdm1 = np.eye(3)
        occ, _ = compute_natural_orbitals(rdm1)
        np.testing.assert_array_almost_equal(occ, [1, 1, 1])

    def test_natural_orbitals_sorting(self):
        """Occupations should be sorted descending."""
        from qoro_pyscf.properties import compute_natural_orbitals
        rdm1 = np.diag([0.1, 0.9, 0.5])
        occ, _ = compute_natural_orbitals(rdm1)
        assert list(occ) == sorted(occ, reverse=True)

    def test_natural_orbitals_trace_preserved(self):
        """Sum of occupations should equal trace of 1-RDM."""
        from qoro_pyscf.properties import compute_natural_orbitals
        rdm1 = np.array([[1.5, 0.2], [0.2, 0.5]])
        occ, _ = compute_natural_orbitals(rdm1)
        assert abs(occ.sum() - np.trace(rdm1)) < 1e-10

    def test_natural_orbitals_symmetric(self):
        """Should work with symmetric matrices."""
        from qoro_pyscf.properties import compute_natural_orbitals
        A = np.random.rand(4, 4)
        rdm1 = (A + A.T) / 2
        occ, coeffs = compute_natural_orbitals(rdm1)
        assert occ.shape == (4,)
        assert coeffs.shape == (4, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# Hamiltonian (integration — requires openfermion)
# ═══════════════════════════════════════════════════════════════════════════════

class TestHamiltonianIntegration:
    """Test Hamiltonian construction with OpenFermion."""

    def test_h2_hamiltonian_shape(self):
        from qoro_pyscf.hamiltonian import integrals_to_qubit_hamiltonian
        from openfermion import QubitOperator
        norb = 2
        h1 = np.random.rand(norb, norb)
        h1 = (h1 + h1.T) / 2
        h2 = np.random.rand(norb, norb, norb, norb)
        h2 = (h2 + h2.transpose(1, 0, 3, 2)) / 2
        qubit_op, identity_offset = integrals_to_qubit_hamiltonian(h1, h2, norb)
        assert isinstance(qubit_op, QubitOperator)
        assert isinstance(identity_offset, float)
        assert len(qubit_op.terms) > 0

    def test_pauli_list_format(self):
        from qoro_pyscf.hamiltonian import (
            integrals_to_qubit_hamiltonian, qubit_op_to_pauli_list,
        )
        norb = 2
        n_qubits = 2 * norb
        h1 = np.eye(norb) * -1.0
        h2 = np.zeros((norb, norb, norb, norb))
        qubit_op, _ = integrals_to_qubit_hamiltonian(h1, h2, norb)
        identity_coeff, labels, coeffs = qubit_op_to_pauli_list(qubit_op, n_qubits)
        for label in labels:
            assert len(label) == n_qubits
            assert all(c in "IXYZ" for c in label)
        assert len(labels) == len(coeffs)

    def test_uhf_integrals(self):
        from qoro_pyscf.hamiltonian import integrals_to_qubit_hamiltonian
        norb = 2
        h1_a = np.eye(norb) * -1.0
        h1_b = np.eye(norb) * -0.9
        h2_aa = np.zeros((norb, norb, norb, norb))
        h2_ab = np.zeros((norb, norb, norb, norb))
        h2_bb = np.zeros((norb, norb, norb, norb))
        qubit_op, _ = integrals_to_qubit_hamiltonian(
            (h1_a, h1_b), (h2_aa, h2_ab, h2_bb), norb
        )
        assert len(qubit_op.terms) > 0

    def test_zero_integrals(self):
        """Zero integrals → zero Hamiltonian."""
        from qoro_pyscf.hamiltonian import integrals_to_qubit_hamiltonian
        norb = 2
        h1 = np.zeros((norb, norb))
        h2 = np.zeros((norb, norb, norb, norb))
        qubit_op, identity_offset = integrals_to_qubit_hamiltonian(h1, h2, norb)
        for term, coeff in qubit_op.terms.items():
            assert abs(coeff) < 1e-12

    def test_hermitian_hamiltonian(self):
        """Qubit Hamiltonian should be Hermitian (real coefficients for Paulis)."""
        from qoro_pyscf.hamiltonian import (
            integrals_to_qubit_hamiltonian, qubit_op_to_pauli_list,
        )
        norb = 2
        h1 = np.array([[-1.25, 0.1], [0.1, -0.5]])
        h2 = np.zeros((norb, norb, norb, norb))
        h2[0, 0, 1, 1] = 0.6
        h2[1, 1, 0, 0] = 0.6
        qubit_op, _ = integrals_to_qubit_hamiltonian(h1, h2, norb)
        _, labels, coeffs = qubit_op_to_pauli_list(qubit_op, 4)
        for c in coeffs:
            assert abs(c.imag) < 1e-10, f"Non-real coefficient: {c}"


# ═══════════════════════════════════════════════════════════════════════════════
# Package exports
# ═══════════════════════════════════════════════════════════════════════════════

class TestPackageExports:
    """Test that all public API is exported correctly."""

    def test_version(self):
        import qoro_pyscf
        assert isinstance(qoro_pyscf.__version__, str)
        assert len(qoro_pyscf.__version__) > 0

    def test_all_exports(self):
        import qoro_pyscf
        expected = {
            "QoroSolver", "BackendConfig", "configure_backend",
            "set_license_key", "compute_dipole_moment",
            "compute_natural_orbitals", "suggest_active_space",
            "suggest_active_space_from_mp2", "taper_hamiltonian",
            "TaperingResult",
        }
        assert expected.issubset(set(qoro_pyscf.__all__))

    def test_py_typed_exists(self):
        from pathlib import Path
        import qoro_pyscf
        pkg_dir = Path(qoro_pyscf.__file__).parent
        assert (pkg_dir / "py.typed").exists()

    def test_all_importable(self):
        import qoro_pyscf
        for name in qoro_pyscf.__all__:
            assert hasattr(qoro_pyscf, name), f"{name} not importable"


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-End (require all deps)
# ═══════════════════════════════════════════════════════════════════════════════

def _can_run_e2e() -> bool:
    try:
        import pyscf, maestro, openfermion  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _can_run_e2e(), reason="Requires pyscf + maestro + openfermion")
class TestE2E:
    """Full VQE pipeline tests (skipped if deps not installed)."""
    pass
