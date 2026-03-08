# Copyright 2026 Qoro Quantum Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test suite for qoro-maestro-pyscf.

Tests are structured in three tiers:
  1. Unit tests (no Maestro/PySCF required) — test pure logic
  2. Integration tests (require openfermion + numpy) — test Hamiltonian/ansatz building
  3. End-to-end tests (require all deps) — test the full VQE pipeline

Run:
    pytest tests/ -v
    pytest tests/ -v -k "unit"           # fast, no deps
    pytest tests/ -v -k "integration"    # requires openfermion
"""

import numpy as np
import pytest


def _has_maestro() -> bool:
    """Check if the Maestro native library is available."""
    try:
        import maestro  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Unit Tests — Pure logic, no external deps beyond numpy
# ══════════════════════════════════════════════════════════════════════════════

class TestAnsatzeUnit:
    """Test ansatz parameter counting and excitation enumeration."""

    def test_hardware_efficient_param_count(self):
        from qoro_maestro_pyscf.ansatze import hardware_efficient_param_count
        assert hardware_efficient_param_count(4, 2) == 16
        assert hardware_efficient_param_count(4, 1) == 8
        assert hardware_efficient_param_count(2, 3) == 12

    def test_uccsd_param_count_tuple(self):
        from qoro_maestro_pyscf.ansatze import uccsd_param_count
        # H₂: 4 qubits, (1,1) electrons
        # occupied: [0,1], virtual: [2,3]
        # singles: 4, doubles: 1, total: 5
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
        # All singles: occupied → virtual
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
            hardware_efficient_ansatz,
            hardware_efficient_param_count,
        )
        n_qubits, n_layers = 4, 2
        n_params = hardware_efficient_param_count(n_qubits, n_layers)
        params = np.zeros(n_params)
        qc = hardware_efficient_ansatz(params, n_qubits, n_layers)
        assert qc is not None

    @pytest.mark.skipif(not _has_maestro(), reason="Requires maestro native library")
    def test_hardware_efficient_with_hf(self):
        from qoro_maestro_pyscf.ansatze import (
            hardware_efficient_ansatz,
            hardware_efficient_param_count,
        )
        n_qubits, n_layers = 4, 2
        n_params = hardware_efficient_param_count(n_qubits, n_layers)
        params = np.zeros(n_params)
        qc = hardware_efficient_ansatz(
            params, n_qubits, n_layers, include_hf=True, nelec=(1, 1)
        )
        assert qc is not None

    @pytest.mark.skipif(not _has_maestro(), reason="Requires maestro native library")
    def test_hardware_efficient_hf_requires_nelec(self):
        from qoro_maestro_pyscf.ansatze import hardware_efficient_ansatz
        with pytest.raises(ValueError, match="nelec is required"):
            hardware_efficient_ansatz(np.zeros(8), 2, 2, include_hf=True)

    @pytest.mark.skipif(not _has_maestro(), reason="Requires maestro native library")
    def test_uccsd_builds(self):
        from qoro_maestro_pyscf.ansatze import uccsd_ansatz, uccsd_param_count
        n_qubits = 4
        nelec = (1, 1)
        n_params = uccsd_param_count(n_qubits, nelec)
        params = np.zeros(n_params)
        qc = uccsd_ansatz(params, n_qubits, nelec)
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
        # H₂: 4 qubits, (1,1) → 1 occ spatial, 1 vir spatial → 1 pair
        assert upccd_param_count(4, (1, 1)) == 1

    def test_upccd_param_count_larger(self):
        from qoro_maestro_pyscf.ansatze import upccd_param_count
        # 6 qubits, (2,2) → 2 occ spatial, 1 vir spatial → 2 pairs
        assert upccd_param_count(6, (2, 2)) == 2

    def test_upccd_param_count_int(self):
        from qoro_maestro_pyscf.ansatze import upccd_param_count
        assert upccd_param_count(4, 2) == upccd_param_count(4, (1, 1))

    def test_upccd_fewer_than_uccsd(self):
        """UpCCD should always have fewer params than UCCSD."""
        from qoro_maestro_pyscf.ansatze import upccd_param_count, uccsd_param_count
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
        # H2: 4 singles + 1 double = 5
        singles = [o for o in ops if o.kind == "single"]
        doubles = [o for o in ops if o.kind == "double"]
        assert len(singles) == 4
        assert len(doubles) == 1
        assert len(ops) == 5

    def test_operator_pool_d_only(self):
        from qoro_maestro_pyscf.adapt import build_operator_pool
        ops = build_operator_pool(4, (1, 1), pool="d")
        # Doubles only
        assert all(o.kind == "double" for o in ops)
        assert len(ops) == 1

    def test_operator_pool_larger(self):
        from qoro_maestro_pyscf.adapt import build_operator_pool
        ops = build_operator_pool(8, (2, 2), pool="sd")
        # Should have both singles and doubles
        singles = [o for o in ops if o.kind == "single"]
        doubles = [o for o in ops if o.kind == "double"]
        assert len(singles) > 0
        assert len(doubles) > 0
        assert len(ops) == len(singles) + len(doubles)

    @pytest.mark.skipif(not _has_maestro(), reason="Requires maestro native library")
    def test_adapt_builds_circuit(self):
        from qoro_maestro_pyscf.adapt import Operator, _build_adapt_circuit
        import numpy as np
        ops = [Operator(kind="double", indices=(0, 1, 2, 3))]
        qc = _build_adapt_circuit(4, (1, 1), ops, np.array([0.5]))
        assert qc is not None


class TestBackendsUnit:
    """Test backend configuration logic."""

    def test_simulation_mode_enum(self):
        from qoro_maestro_pyscf.backends import SimulationMode
        assert SimulationMode("statevector") == SimulationMode.STATEVECTOR
        assert SimulationMode("mps") == SimulationMode.MPS

    def test_simulation_mode_invalid(self):
        from qoro_maestro_pyscf.backends import SimulationMode
        with pytest.raises(ValueError):
            SimulationMode("invalid_mode")

    def test_set_license_key(self):
        import os
        from qoro_maestro_pyscf.backends import set_license_key
        set_license_key("TEST-KEY-1234")
        assert os.environ["MAESTRO_LICENSE_KEY"] == "TEST-KEY-1234"
        del os.environ["MAESTRO_LICENSE_KEY"]

    def test_set_license_key_overwrite(self):
        import os
        from qoro_maestro_pyscf.backends import set_license_key
        set_license_key("KEY-1")
        set_license_key("KEY-2")
        assert os.environ["MAESTRO_LICENSE_KEY"] == "KEY-2"
        del os.environ["MAESTRO_LICENSE_KEY"]


class TestRdmUnit:
    """Test RDM tracing functions."""

    def test_trace_spin_rdm1(self):
        from qoro_maestro_pyscf.rdm import trace_spin_rdm1
        rdm1_a = np.eye(2) * 0.5
        rdm1_b = np.eye(2) * 0.3
        traced = trace_spin_rdm1(rdm1_a, rdm1_b)
        np.testing.assert_allclose(traced, np.eye(2) * 0.8)

    def test_trace_spin_rdm2(self):
        from qoro_maestro_pyscf.rdm import trace_spin_rdm2
        shape = (2, 2, 2, 2)
        rdm2_aa = np.ones(shape) * 0.1
        rdm2_ab = np.ones(shape) * 0.2
        rdm2_bb = np.ones(shape) * 0.3
        traced = trace_spin_rdm2(rdm2_aa, rdm2_ab, rdm2_bb)
        # traced = aa + ab + ba + bb = 0.1 + 0.2 + 0.2 + 0.3 = 0.8
        np.testing.assert_allclose(traced, np.ones(shape) * 0.8)

    def test_trace_rdm1_shape(self):
        from qoro_maestro_pyscf.rdm import trace_spin_rdm1
        norb = 3
        rdm1_a = np.random.rand(norb, norb)
        rdm1_b = np.random.rand(norb, norb)
        traced = trace_spin_rdm1(rdm1_a, rdm1_b)
        assert traced.shape == (norb, norb)

    def test_trace_rdm2_symmetry(self):
        """rdm2_ba should be the transpose of rdm2_ab."""
        from qoro_maestro_pyscf.rdm import trace_spin_rdm2
        norb = 2
        rdm2_ab = np.random.rand(norb, norb, norb, norb)
        rdm2_aa = np.zeros_like(rdm2_ab)
        rdm2_bb = np.zeros_like(rdm2_ab)
        traced = trace_spin_rdm2(rdm2_aa, rdm2_ab, rdm2_bb)
        # Should be rdm2_ab + rdm2_ab.transpose(2,3,0,1)
        expected = rdm2_ab + rdm2_ab.transpose(2, 3, 0, 1)
        np.testing.assert_allclose(traced, expected)


class TestPropertiesUnit:
    """Test molecular property computation functions."""

    def test_natural_orbitals_identity(self):
        """Identity 1-RDM → occupations are all 1."""
        from qoro_maestro_pyscf.properties import compute_natural_orbitals
        rdm1 = np.eye(3)
        occ, coeffs = compute_natural_orbitals(rdm1)
        np.testing.assert_allclose(occ, [1.0, 1.0, 1.0], atol=1e-10)
        assert coeffs.shape == (3, 3)

    def test_natural_orbitals_sorting(self):
        """Occupations should be sorted descending."""
        from qoro_maestro_pyscf.properties import compute_natural_orbitals
        rdm1 = np.diag([0.1, 1.9, 0.5])
        occ, _ = compute_natural_orbitals(rdm1)
        assert occ[0] >= occ[1] >= occ[2]
        np.testing.assert_allclose(sorted(occ, reverse=True), occ, atol=1e-10)

    def test_natural_orbitals_trace_preserved(self):
        """Sum of occupations should equal trace of 1-RDM."""
        from qoro_maestro_pyscf.properties import compute_natural_orbitals
        rdm1 = np.array([[1.8, 0.1], [0.1, 0.2]])
        occ, _ = compute_natural_orbitals(rdm1)
        np.testing.assert_allclose(np.sum(occ), np.trace(rdm1), atol=1e-10)

    def test_natural_orbitals_symmetric(self):
        """Should work with symmetric matrices."""
        from qoro_maestro_pyscf.properties import compute_natural_orbitals
        rdm1 = np.array([[1.5, 0.3], [0.3, 0.5]])
        occ, coeffs = compute_natural_orbitals(rdm1)
        assert len(occ) == 2
        assert coeffs.shape == (2, 2)


# ══════════════════════════════════════════════════════════════════════════════
# Integration Tests — Require openfermion + numpy
# ══════════════════════════════════════════════════════════════════════════════

class TestHamiltonianIntegration:
    """Test Hamiltonian construction with OpenFermion."""

    def test_h2_hamiltonian_shape(self):
        from qoro_maestro_pyscf.hamiltonian import (
            integrals_to_qubit_hamiltonian,
        )
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
        from qoro_maestro_pyscf.hamiltonian import (
            integrals_to_qubit_hamiltonian,
            qubit_op_to_pauli_list,
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
        from qoro_maestro_pyscf.hamiltonian import integrals_to_qubit_hamiltonian
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
        from qoro_maestro_pyscf.hamiltonian import integrals_to_qubit_hamiltonian
        norb = 2
        h1 = np.zeros((norb, norb))
        h2 = np.zeros((norb, norb, norb, norb))
        qubit_op, identity_offset = integrals_to_qubit_hamiltonian(h1, h2, norb)
        for term, coeff in qubit_op.terms.items():
            assert abs(coeff) < 1e-12

    def test_hermitian_hamiltonian(self):
        """Qubit Hamiltonian should be Hermitian (real coefficients for Paulis)."""
        from qoro_maestro_pyscf.hamiltonian import (
            integrals_to_qubit_hamiltonian,
            qubit_op_to_pauli_list,
        )
        norb = 2
        h1 = np.array([[- 1.25, 0.1], [0.1, -0.5]])
        h2 = np.zeros((norb, norb, norb, norb))
        h2[0, 0, 1, 1] = 0.6
        h2[1, 1, 0, 0] = 0.6
        qubit_op, _ = integrals_to_qubit_hamiltonian(h1, h2, norb)
        _, labels, coeffs = qubit_op_to_pauli_list(qubit_op, 4)
        # All Pauli coefficients should be real for a Hermitian Hamiltonian
        for c in coeffs:
            assert abs(c.imag) < 1e-10, f"Non-real coefficient: {c}"


@pytest.mark.skipif(not _has_maestro(), reason="Requires maestro native library")
class TestMaestroSolverIntegration:
    """Test MaestroSolver construction and validation."""

    def test_solver_construction(self):
        from qoro_maestro_pyscf import MaestroSolver
        solver = MaestroSolver(
            ansatz="hardware_efficient",
            ansatz_layers=2,
            backend="cpu",
        )
        assert solver.ansatz == "hardware_efficient"
        assert solver.ansatz_layers == 2
        assert solver.backend == "cpu"

    def test_solver_uccsd_construction(self):
        from qoro_maestro_pyscf import MaestroSolver
        solver = MaestroSolver(ansatz="uccsd", backend="gpu")
        assert solver.ansatz == "uccsd"

    def test_solver_mps_construction(self):
        from qoro_maestro_pyscf import MaestroSolver
        solver = MaestroSolver(simulation="mps", mps_bond_dim=128)
        assert solver.simulation == "mps"
        assert solver.mps_bond_dim == 128

    def test_solver_license_key(self):
        from qoro_maestro_pyscf import MaestroSolver
        solver = MaestroSolver(license_key="TEST-1234")
        assert solver.license_key == "TEST-1234"

    def test_solver_default_state(self):
        from qoro_maestro_pyscf import MaestroSolver
        solver = MaestroSolver()
        assert solver.converged is False
        assert solver.optimal_params is None
        assert solver.energy_history == []

    def test_approx_kernel_alias(self):
        from qoro_maestro_pyscf import MaestroSolver
        solver = MaestroSolver()
        assert solver.approx_kernel == solver.kernel

    def test_fix_spin_chaining(self):
        from qoro_maestro_pyscf import MaestroSolver
        solver = MaestroSolver()
        result = solver.fix_spin_(shift=0.5, ss=0.0)
        assert result is solver  # returns self for chaining
        assert solver._spin_penalty_shift == 0.5
        assert solver._spin_penalty_ss == 0.0

    def test_fix_spin_defaults(self):
        from qoro_maestro_pyscf import MaestroSolver
        solver = MaestroSolver()
        solver.fix_spin_()
        assert solver._spin_penalty_shift == 0.2
        assert solver._spin_penalty_ss == 0.0  # singlet


@pytest.mark.skipif(not _has_maestro(), reason="Requires maestro native library")
class TestPackageExports:
    """Test that all public API is exported correctly."""

    def test_maestro_solver_import(self):
        from qoro_maestro_pyscf import MaestroSolver
        assert MaestroSolver is not None

    def test_backend_config_import(self):
        from qoro_maestro_pyscf import BackendConfig, configure_backend
        assert BackendConfig is not None
        assert configure_backend is not None

    def test_set_license_key_import(self):
        from qoro_maestro_pyscf import set_license_key
        assert callable(set_license_key)

    def test_properties_import(self):
        from qoro_maestro_pyscf import compute_dipole_moment, compute_natural_orbitals
        assert callable(compute_dipole_moment)
        assert callable(compute_natural_orbitals)

    def test_version(self):
        import qoro_maestro_pyscf
        assert hasattr(qoro_maestro_pyscf, "__version__")
        assert qoro_maestro_pyscf.__version__ == "0.2.0"

    def test_all_exports(self):
        import qoro_maestro_pyscf
        expected = {
            "MaestroSolver",
            "BackendConfig",
            "configure_backend",
            "set_license_key",
            "compute_dipole_moment",
            "compute_natural_orbitals",
        }
        assert expected.issubset(set(qoro_maestro_pyscf.__all__))


# ══════════════════════════════════════════════════════════════════════════════
# End-to-End Tests — Require PySCF + Maestro + OpenFermion
# ══════════════════════════════════════════════════════════════════════════════

def _can_run_e2e() -> bool:
    """Check if all dependencies for e2e tests are available."""
    try:
        import pyscf  # noqa: F401
        import maestro  # noqa: F401
        import openfermion  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _can_run_e2e(), reason="Requires pyscf + maestro + openfermion")
class TestE2E:
    """Full VQE pipeline tests (skipped if deps not installed)."""
    pass
