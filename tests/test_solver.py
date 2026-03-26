# Copyright 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Tests for QoroSolver fields, methods, and serialisation."""

import numpy as np
import pytest


def _has_maestro() -> bool:
    try:
        import maestro  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests — no Qoro required
# ═══════════════════════════════════════════════════════════════════════════════

class TestSolverFieldsUnit:
    """Test QoroSolver dataclass fields and defaults."""

    def test_defaults(self):
        from qoro_pyscf import QoroSolver
        s = QoroSolver()
        assert s.ansatz == "hardware_efficient"
        assert s.ansatz_layers == 2
        assert s.backend == "cpu"
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
        from qoro_pyscf import QoroSolver

        def my_ansatz(params, n_qubits, nelec):
            return None

        solver = QoroSolver(
            ansatz="custom",
            custom_ansatz=my_ansatz,
            custom_ansatz_n_params=5,
        )
        assert solver.ansatz == "custom"
        assert callable(solver.custom_ansatz)
        assert solver.custom_ansatz_n_params == 5

    def test_custom_ansatz_requires_callable(self):
        """ansatz='custom' without custom_ansatz raises ValueError."""
        from qoro_pyscf import QoroSolver
        solver = QoroSolver(ansatz="custom")
        with pytest.raises(ValueError, match="custom_ansatz"):
            solver.kernel(
                np.zeros((2, 2)), np.zeros((2, 2, 2, 2)), 2, (1, 1)
            )

    def test_custom_ansatz_requires_n_params(self):
        """Callable custom_ansatz without n_params raises ValueError."""
        from qoro_pyscf import QoroSolver
        solver = QoroSolver(
            ansatz="custom",
            custom_ansatz=lambda p, n, e: None,
        )
        with pytest.raises(ValueError, match="custom_ansatz_n_params"):
            solver.kernel(
                np.zeros((2, 2)), np.zeros((2, 2, 2, 2)), 2, (1, 1)
            )

    def test_custom_ansatz_prebuilt_n_params_zero(self):
        """Pre-built circuit (non-callable) yields zero parameters."""
        from qoro_pyscf import QoroSolver

        class FakeCircuit:
            pass

        solver = QoroSolver(
            ansatz="custom",
            custom_ansatz=FakeCircuit(),
        )
        assert not callable(solver.custom_ansatz)
        assert solver.custom_ansatz_n_params is None

    def test_callback_field(self):
        """Callback callable is stored."""
        from qoro_pyscf import QoroSolver
        calls = []
        solver = QoroSolver(callback=lambda it, e, p: calls.append(it))
        assert solver.callback is not None

    def test_taper_field_set(self):
        from qoro_pyscf import QoroSolver
        assert QoroSolver(taper=True).taper is True

    def test_vqd_penalty_custom(self):
        from qoro_pyscf import QoroSolver
        assert QoroSolver(vqd_penalty=10.0).vqd_penalty == 10.0

    def test_learning_rate_default(self):
        from qoro_pyscf import QoroSolver
        assert QoroSolver().learning_rate == 0.01

    def test_learning_rate_custom(self):
        from qoro_pyscf import QoroSolver
        assert QoroSolver(learning_rate=0.05).learning_rate == 0.05

    def test_grad_shift_default(self):
        from qoro_pyscf import QoroSolver
        assert QoroSolver().grad_shift == pytest.approx(1e-3)

    def test_adam_optimizer_field(self):
        """optimizer='adam' is accepted without errors."""
        from qoro_pyscf import QoroSolver
        solver = QoroSolver(optimizer="adam", learning_rate=0.02)
        assert solver.optimizer == "adam"
        assert solver.learning_rate == 0.02


class TestAdamConvergenceUnit:
    """Test Adam optimizer logic (no Qoro required).

    These tests replicate the exact Adam + parameter-shift loop from
    QoroSolver.kernel() on simple known cost functions.
    """

    def test_parameter_shift_gradient_exact(self):
        """Parameter-shift at π/2 gives exact gradients for sin-based cost."""
        shift = np.pi / 2
        for theta in [0.0, 0.5, 1.0, np.pi / 4]:
            analytic = np.cos(theta)
            ps_grad = (np.sin(theta + shift) - np.sin(theta - shift)) / (2 * np.sin(shift))
            np.testing.assert_allclose(ps_grad, analytic, atol=1e-12)

    def test_adam_converges_quadratic(self):
        """Adam should minimise f(θ) = Σ(θ_i − 1)² to near-zero."""
        n_params = 4
        target = np.ones(n_params)

        def cost(params):
            return float(np.sum((params - target) ** 2))

        # Replicate the exact Adam loop from kernel()
        shift = np.pi / 2
        lr = 0.05
        params = np.zeros(n_params)
        m = np.zeros_like(params)
        v = np.zeros_like(params)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        best_energy = float('inf')
        best_params = params.copy()

        for it in range(1, 201):
            grad = np.zeros_like(params)
            for j in range(len(params)):
                p_plus = params.copy(); p_plus[j] += shift
                p_minus = params.copy(); p_minus[j] -= shift
                grad[j] = (cost(p_plus) - cost(p_minus)) / (2 * np.sin(shift))

            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            m_hat = m / (1 - beta1 ** it)
            v_hat = v / (1 - beta2 ** it)
            params -= lr * m_hat / (np.sqrt(v_hat) + eps)

            e = cost(params)
            if e < best_energy:
                best_energy = e
                best_params = params.copy()

        assert best_energy < 0.01, f"Adam didn't converge: {best_energy}"
        np.testing.assert_allclose(best_params, target, atol=0.1)


class TestFixSpinUnit:
    """Test fix_spin_ method."""

    def test_sets_fields(self):
        from qoro_pyscf import QoroSolver
        solver = QoroSolver()
        ret = solver.fix_spin_(shift=0.5, ss=2.0)
        assert ret is solver
        assert solver._spin_penalty_shift == 0.5
        assert solver._spin_penalty_ss == 2.0

    def test_defaults_to_singlet(self):
        from qoro_pyscf import QoroSolver
        solver = QoroSolver()
        solver.fix_spin_(shift=0.3)
        assert solver._spin_penalty_ss == 0.0


class TestEvaluateCustomPaulisUnit:
    """Test validation for evaluate_custom_paulis and get_final_statevector."""

    def test_evaluate_custom_paulis_no_circuit(self):
        from qoro_pyscf import QoroSolver
        solver = QoroSolver()
        with pytest.raises(RuntimeError, match="No circuit available"):
            solver.evaluate_custom_paulis([])

    def test_get_final_statevector_no_circuit(self):
        from qoro_pyscf import QoroSolver
        solver = QoroSolver()
        with pytest.raises(RuntimeError, match="No circuit available"):
            solver.get_final_statevector()


class TestSaveLoadUnit:
    """Test save/load serialisation."""

    def test_round_trip(self, tmp_path):
        from qoro_pyscf import QoroSolver

        solver = QoroSolver(
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
        loaded = QoroSolver.load(path)

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
        from qoro_pyscf import QoroSolver

        solver = QoroSolver()
        path = tmp_path / "empty"
        solver.save(path)
        loaded = QoroSolver.load(path)
        assert loaded.optimal_params is None
        assert loaded.energy_history == []


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests — require Qoro native library
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _has_maestro(), reason="Requires maestro native library")
class TestQoroSolverIntegration:
    """Test QoroSolver with Qoro backend."""

    def test_solver_construction(self):
        from qoro_pyscf import QoroSolver
        solver = QoroSolver(
            ansatz="hardware_efficient", ansatz_layers=2, backend="cpu",
        )
        assert solver.ansatz == "hardware_efficient"

    def test_solver_uccsd_construction(self):
        from qoro_pyscf import QoroSolver
        solver = QoroSolver(ansatz="uccsd", backend="gpu")
        assert solver.ansatz == "uccsd"

    def test_solver_mps_construction(self):
        from qoro_pyscf import QoroSolver
        solver = QoroSolver(simulation="mps", mps_bond_dim=128)
        assert solver.simulation == "mps"
        assert solver.mps_bond_dim == 128

    def test_solver_license_key(self):
        from qoro_pyscf import QoroSolver
        solver = QoroSolver(license_key="TEST-1234")
        assert solver.license_key == "TEST-1234"

    def test_approx_kernel_alias(self):
        from qoro_pyscf import QoroSolver
        solver = QoroSolver()
        assert solver.approx_kernel == solver.kernel

    def test_fix_spin_chaining(self):
        from qoro_pyscf import QoroSolver
        solver = QoroSolver()
        result = solver.fix_spin_(shift=0.5, ss=0.0)
        assert result is solver
        assert solver._spin_penalty_shift == 0.5

    def test_fix_spin_defaults(self):
        from qoro_pyscf import QoroSolver
        solver = QoroSolver()
        solver.fix_spin_()
        assert solver._spin_penalty_shift == 0.2
        assert solver._spin_penalty_ss == 0.0
