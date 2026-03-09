# Copyright 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Tests for MaestroSolver fields, methods, and serialisation."""

import numpy as np
import pytest


def _has_maestro() -> bool:
    try:
        import maestro  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — no Maestro required
# ═══════════════════════════════════════════════════════════════════════════════

class TestSolverFieldsUnit:
    """Test MaestroSolver dataclass fields and defaults."""

    def test_defaults(self):
        from qoro_maestro_pyscf import MaestroSolver
        s = MaestroSolver()
        assert s.ansatz == "hardware_efficient"
        assert s.ansatz_layers == 2
        assert s.backend == "gpu"
        assert s.simulation == "statevector"
        assert s.taper is False
        assert s.vqd_penalty == 5.0
        assert s.nroots == 1
        assert s.converged is False
        assert s.optimal_params is None
        assert s.energy_history == []
        assert s.callback is None
        assert s._spin_penalty_shift == 0.0
        assert s._vqd_energies == []
        assert s._vqd_circuits == []

    def test_custom_ansatz_fields(self):
        """Setting ansatz='custom' with callable stores fields correctly."""
        from qoro_maestro_pyscf import MaestroSolver

        def my_ansatz(params, n_qubits, nelec):
            return None

        solver = MaestroSolver(
            ansatz="custom",
            custom_ansatz=my_ansatz,
            custom_ansatz_n_params=5,
        )
        assert solver.ansatz == "custom"
        assert callable(solver.custom_ansatz)
        assert solver.custom_ansatz_n_params == 5

    def test_custom_ansatz_requires_callable(self):
        """ansatz='custom' without custom_ansatz raises ValueError."""
        from qoro_maestro_pyscf import MaestroSolver
        solver = MaestroSolver(ansatz="custom")
        with pytest.raises(ValueError, match="custom_ansatz"):
            solver.kernel(
                np.zeros((2, 2)), np.zeros((2, 2, 2, 2)), 2, (1, 1)
            )

    def test_custom_ansatz_requires_n_params(self):
        """Callable custom_ansatz without n_params raises ValueError."""
        from qoro_maestro_pyscf import MaestroSolver
        solver = MaestroSolver(
            ansatz="custom",
            custom_ansatz=lambda p, n, e: None,
        )
        with pytest.raises(ValueError, match="custom_ansatz_n_params"):
            solver.kernel(
                np.zeros((2, 2)), np.zeros((2, 2, 2, 2)), 2, (1, 1)
            )

    def test_custom_ansatz_prebuilt_n_params_zero(self):
        """Pre-built circuit (non-callable) yields zero parameters."""
        from qoro_maestro_pyscf import MaestroSolver

        class FakeCircuit:
            pass

        solver = MaestroSolver(
            ansatz="custom",
            custom_ansatz=FakeCircuit(),
        )
        assert not callable(solver.custom_ansatz)
        assert solver.custom_ansatz_n_params is None

    def test_callback_field(self):
        """Callback callable is stored."""
        from qoro_maestro_pyscf import MaestroSolver
        calls = []
        solver = MaestroSolver(callback=lambda it, e, p: calls.append(it))
        assert solver.callback is not None

    def test_taper_field_set(self):
        from qoro_maestro_pyscf import MaestroSolver
        assert MaestroSolver(taper=True).taper is True

    def test_vqd_penalty_custom(self):
        from qoro_maestro_pyscf import MaestroSolver
        assert MaestroSolver(vqd_penalty=10.0).vqd_penalty == 10.0


class TestFixSpinUnit:
    """Test fix_spin_ method."""

    def test_sets_fields(self):
        from qoro_maestro_pyscf import MaestroSolver
        solver = MaestroSolver()
        ret = solver.fix_spin_(shift=0.5, ss=2.0)
        assert ret is solver
        assert solver._spin_penalty_shift == 0.5
        assert solver._spin_penalty_ss == 2.0

    def test_defaults_to_singlet(self):
        from qoro_maestro_pyscf import MaestroSolver
        solver = MaestroSolver()
        solver.fix_spin_(shift=0.3)
        assert solver._spin_penalty_ss == 0.0


class TestEvaluateCustomPaulisUnit:
    """Test validation for evaluate_custom_paulis and get_final_statevector."""

    def test_evaluate_custom_paulis_no_circuit(self):
        from qoro_maestro_pyscf import MaestroSolver
        solver = MaestroSolver()
        with pytest.raises(RuntimeError, match="No circuit available"):
            solver.evaluate_custom_paulis([])

    def test_get_final_statevector_no_circuit(self):
        from qoro_maestro_pyscf import MaestroSolver
        solver = MaestroSolver()
        with pytest.raises(RuntimeError, match="No circuit available"):
            solver.get_final_statevector()


class TestSaveLoadUnit:
    """Test save/load serialisation."""

    def test_round_trip(self, tmp_path):
        from qoro_maestro_pyscf import MaestroSolver

        solver = MaestroSolver(
            ansatz="uccsd",
            optimizer="L-BFGS-B",
            maxiter=500,
            backend="cpu",
            simulation="mps",
            mps_bond_dim=128,
        )
        solver.converged = True
        solver.vqe_time = 3.14
        solver.optimal_params = np.array([0.1, 0.2, 0.3])
        solver.energy_history = [-1.0, -1.1, -1.15]
        solver._n_qubits = 8
        solver._nelec = (2, 2)
        solver._spin_penalty_shift = 0.5
        solver._spin_penalty_ss = 0.75

        path = tmp_path / "checkpoint"
        solver.save(path)
        loaded = MaestroSolver.load(path)

        assert loaded.ansatz == "uccsd"
        assert loaded.optimizer == "L-BFGS-B"
        assert loaded.maxiter == 500
        assert loaded.backend == "cpu"
        assert loaded.simulation == "mps"
        assert loaded.mps_bond_dim == 128
        assert loaded.converged is True
        assert loaded.vqe_time == pytest.approx(3.14)
        assert loaded._n_qubits == 8
        assert loaded._nelec == (2, 2)
        assert loaded._spin_penalty_shift == 0.5
        assert loaded._spin_penalty_ss == 0.75
        np.testing.assert_array_almost_equal(
            loaded.optimal_params, [0.1, 0.2, 0.3]
        )
        assert loaded.energy_history == pytest.approx([-1.0, -1.1, -1.15])
        np.testing.assert_array_almost_equal(
            loaded.initial_point, [0.1, 0.2, 0.3]
        )

    def test_empty_solver(self, tmp_path):
        from qoro_maestro_pyscf import MaestroSolver

        solver = MaestroSolver()
        path = tmp_path / "empty"
        solver.save(path)
        loaded = MaestroSolver.load(path)
        assert loaded.optimal_params is None
        assert loaded.energy_history == []


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests — require Maestro native library
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _has_maestro(), reason="Requires maestro native library")
class TestMaestroSolverIntegration:
    """Test MaestroSolver with Maestro backend."""

    def test_solver_construction(self):
        from qoro_maestro_pyscf import MaestroSolver
        solver = MaestroSolver(
            ansatz="hardware_efficient", ansatz_layers=2, backend="cpu",
        )
        assert solver.ansatz == "hardware_efficient"

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

    def test_approx_kernel_alias(self):
        from qoro_maestro_pyscf import MaestroSolver
        solver = MaestroSolver()
        assert solver.approx_kernel == solver.kernel

    def test_fix_spin_chaining(self):
        from qoro_maestro_pyscf import MaestroSolver
        solver = MaestroSolver()
        result = solver.fix_spin_(shift=0.5, ss=0.0)
        assert result is solver
        assert solver._spin_penalty_shift == 0.5

    def test_fix_spin_defaults(self):
        from qoro_maestro_pyscf import MaestroSolver
        solver = MaestroSolver()
        solver.fix_spin_()
        assert solver._spin_penalty_shift == 0.2
        assert solver._spin_penalty_ss == 0.0
