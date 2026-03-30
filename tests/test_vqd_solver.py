# Copyright 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Tests for VQDSolver fields, methods, and penalty weight handling."""

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

class TestVQDSolverFieldsUnit:
    """Test VQDSolver dataclass fields and defaults."""

    def test_defaults(self):
        from qoro_pyscf import VQDSolver
        v = VQDSolver()
        assert v.num_states == 2
        assert v.penalty_weights == 5.0
        assert v.converged is False
        assert v.energies is None
        assert v.optimized_states == []
        assert v.callback is None

    def test_custom_num_states(self):
        from qoro_pyscf import VQDSolver
        v = VQDSolver(num_states=5)
        assert v.num_states == 5

    def test_custom_penalty_weights_scalar(self):
        from qoro_pyscf import VQDSolver
        v = VQDSolver(penalty_weights=10.0)
        assert v.penalty_weights == 10.0

    def test_custom_penalty_weights_list(self):
        from qoro_pyscf import VQDSolver
        v = VQDSolver(penalty_weights=[3.0, 5.0, 7.0])
        assert v.penalty_weights == [3.0, 5.0, 7.0]

    def test_callback_field(self):
        from qoro_pyscf import VQDSolver
        calls = []
        v = VQDSolver(callback=lambda r, i, e, p: calls.append((r, i)))
        assert v.callback is not None

    def test_inner_solver_default(self):
        """Default inner solver is a QoroSolver with defaults."""
        from qoro_pyscf import VQDSolver, QoroSolver
        v = VQDSolver()
        assert isinstance(v.solver, QoroSolver)
        assert v.solver.ansatz == "hardware_efficient"

    def test_inner_solver_custom(self):
        from qoro_pyscf import VQDSolver, QoroSolver
        inner = QoroSolver(ansatz="uccsd", backend="gpu", maxiter=500)
        v = VQDSolver(solver=inner, num_states=3)
        assert v.solver.ansatz == "uccsd"
        assert v.solver.backend == "gpu"
        assert v.solver.maxiter == 500

    def test_approx_kernel_alias(self):
        from qoro_pyscf import VQDSolver
        v = VQDSolver()
        assert v.approx_kernel == v.kernel


class TestPenaltyWeightsResolution:
    """Test _resolve_penalty_weights helper."""

    def test_scalar_expands(self):
        from qoro_pyscf import VQDSolver
        v = VQDSolver(num_states=4, penalty_weights=10.0)
        weights = v._resolve_penalty_weights()
        assert weights == [10.0, 10.0, 10.0]

    def test_list_exact_length(self):
        from qoro_pyscf import VQDSolver
        v = VQDSolver(num_states=3, penalty_weights=[2.0, 4.0])
        weights = v._resolve_penalty_weights()
        assert weights == [2.0, 4.0]

    def test_list_shorter_pads(self):
        from qoro_pyscf import VQDSolver
        v = VQDSolver(num_states=5, penalty_weights=[3.0, 5.0])
        weights = v._resolve_penalty_weights()
        assert len(weights) == 4
        # Padded with last value
        assert weights == [3.0, 5.0, 5.0, 5.0]

    def test_list_longer_truncates(self):
        from qoro_pyscf import VQDSolver
        v = VQDSolver(num_states=2, penalty_weights=[1.0, 2.0, 3.0])
        weights = v._resolve_penalty_weights()
        assert weights == [1.0]

    def test_single_state_empty(self):
        from qoro_pyscf import VQDSolver
        v = VQDSolver(num_states=1, penalty_weights=5.0)
        weights = v._resolve_penalty_weights()
        assert weights == []


class TestVQDSolverStateAccessUnit:
    """Test state access methods raise correctly before kernel() is run."""

    def test_get_state_raises_before_kernel(self):
        from qoro_pyscf import VQDSolver
        v = VQDSolver()
        with pytest.raises(IndexError, match="Root 0 not available"):
            v.get_state(0)

    def test_get_statevector_raises_before_kernel(self):
        from qoro_pyscf import VQDSolver
        v = VQDSolver()
        with pytest.raises(IndexError, match="Root 0 not available"):
            v.get_statevector(0)

    def test_set_active_root_raises_before_kernel(self):
        from qoro_pyscf import VQDSolver
        v = VQDSolver()
        with pytest.raises(IndexError, match="Root 0 not available"):
            v.set_active_root(0)


class TestPackageExportsVQD:
    """Test that VQDSolver is exported from the package."""

    def test_vqd_solver_in_all(self):
        import qoro_pyscf
        assert "VQDSolver" in qoro_pyscf.__all__

    def test_vqd_solver_importable(self):
        from qoro_pyscf import VQDSolver
        assert VQDSolver is not None

    def test_compute_statevector_fidelity_in_all(self):
        import qoro_pyscf
        assert "compute_statevector_fidelity" in qoro_pyscf.__all__

    def test_compute_statevector_fidelity_importable(self):
        from qoro_pyscf import compute_statevector_fidelity
        assert compute_statevector_fidelity is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests — require Maestro native library
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _has_maestro(), reason="Requires maestro native library")
class TestVQDSolverIntegration:
    """Integration tests for VQDSolver with Maestro backend."""

    def test_construction_with_solver(self):
        from qoro_pyscf import QoroSolver, VQDSolver
        vqd = VQDSolver(
            solver=QoroSolver(ansatz="uccsd"),
            num_states=2,
            penalty_weights=10.0,
        )
        assert vqd.solver.ansatz == "uccsd"
        assert vqd.num_states == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Regression tests — guard against maestro.get_state_vector re-introduction
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoGetStateVectorCalls:
    """
    Regression tests ensuring maestro.get_state_vector is never called directly.

    When maestro.get_state_vector / inner_product become available in a future
    wheel, the code should use them through compute_overlap's feature-detection
    path — NOT by hard-coding a direct call that would crash on older installs.
    """

    # Files that previously contained direct maestro.get_state_vector calls.
    GUARDED_FILES = [
        "qoro_pyscf/qoro_solver.py",
        "qoro_pyscf/vqd_solver.py",
        "qoro_pyscf/expectation.py",
    ]

    def _source(self, rel_path: str) -> str:
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, rel_path)) as f:
            return f.read()

    def _code_only(self, src: str) -> str:
        """Return only the executable code lines — strips docstrings and comments."""
        lines = []
        in_docstring = False
        quote = None
        for line in src.splitlines():
            stripped = line.lstrip()
            if in_docstring:
                # Check if this line closes the docstring
                if quote in line:
                    in_docstring = False
                continue  # skip docstring body lines
            # Check for opening triple-quote
            for q in ('"""', "'''"):
                if q in stripped:
                    # Count occurrences: even = opens+closes on same line, odd = opens
                    if stripped.count(q) % 2 == 1:
                        in_docstring = True
                        quote = q
                    break
            if in_docstring:
                continue
            # Strip inline comments and collect
            import re
            lines.append(re.sub(r'#.*', '', line))
        return "\n".join(lines)

    def test_no_bare_get_state_vector_in_qoro_solver(self):
        """qoro_solver.py must not call maestro.get_state_vector directly."""
        raw_src = self._source("qoro_pyscf/qoro_solver.py")
        src = self._code_only(raw_src)
        import re
        all_calls = re.findall(r'maestro\.get_state_vector\s*\(', src)
        # Allowlist: inside hasattr() guards or inside string literals
        # (e.g. the NotImplementedError message)
        guarded = re.findall(r'hasattr\(maestro,\s*["\']get_state_vector', src)
        in_strings = re.findall(r'["\'].*maestro\.get_state_vector.*["\']', src)
        bare = len(all_calls) - len(guarded) - len(in_strings)
        assert bare <= 0, (
            f"maestro.get_state_vector called directly in qoro_solver.py "
            f"({len(all_calls)} occurrence(s), {len(guarded)} guarded, {len(in_strings)} in strings)\n"
            "Use compute_overlap() which handles API detection gracefully."
        )

    def test_no_bare_get_state_vector_in_vqd_solver(self):
        """vqd_solver.py must not call maestro.get_state_vector directly."""
        src = self._code_only(self._source("qoro_pyscf/vqd_solver.py"))
        import re
        all_calls = re.findall(r'maestro\.get_state_vector\s*\(', src)
        assert len(all_calls) == 0, (
            f"maestro.get_state_vector called directly in vqd_solver.py: {len(all_calls)} occurrence(s)"
        )

    def test_no_bare_get_state_vector_in_expectation(self):
        """expectation.py must not call maestro.get_state_vector directly."""
        src = self._code_only(self._source("qoro_pyscf/expectation.py"))
        import re
        all_calls = re.findall(r'maestro\.get_state_vector\s*\(', src)
        assert len(all_calls) == 0, (
            f"maestro.get_state_vector called directly in expectation.py: {len(all_calls)} occurrence(s)"
        )

    def test_compute_overlap_exported(self):
        """compute_overlap must be in __all__ so it's the canonical public API."""
        import qoro_pyscf
        assert "compute_overlap" in qoro_pyscf.__all__

    @pytest.mark.skipif(not _has_maestro(), reason="Requires maestro native library")
    def test_compute_overlap_works_without_inner_product(self):
        """compute_overlap must not crash even when maestro.inner_product is absent."""
        import maestro
        from maestro.circuits import QuantumCircuit
        from qoro_pyscf.backends import configure_backend
        from qoro_pyscf.expectation import compute_overlap

        config = configure_backend(use_gpu=False, simulation="statevector")
        qc = QuantumCircuit()
        qc.h(0)

        # Temporarily remove inner_product if it exists, then restore
        had_inner = hasattr(maestro, "inner_product")
        if had_inner:
            saved = maestro.inner_product
            delattr(maestro, "inner_product")
        try:
            result = compute_overlap(qc, qc, config)
        finally:
            if had_inner:
                maestro.inner_product = saved

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0 + 1e-9

    @pytest.mark.skipif(not _has_maestro(), reason="Requires maestro native library")
    def test_compute_overlap_uses_inner_product_when_available(self):
        """compute_overlap must delegate to maestro.inner_product when present."""
        import unittest.mock as mock
        import maestro
        from maestro.circuits import QuantumCircuit
        from qoro_pyscf.backends import configure_backend
        from qoro_pyscf.expectation import compute_overlap

        config = configure_backend(use_gpu=False, simulation="statevector")
        qc = QuantumCircuit()
        qc.h(0)

        sentinel = 0.42
        with mock.patch.object(maestro, "inner_product", return_value=sentinel, create=True):
            result = compute_overlap(qc, qc, config)

        assert result == pytest.approx(abs(sentinel) ** 2)

    @pytest.mark.skipif(not _has_maestro(), reason="Requires maestro native library")
    def test_get_final_statevector_raises_not_implemented(self):
        """get_final_statevector must raise NotImplementedError, not AttributeError."""
        from maestro.circuits import QuantumCircuit
        from qoro_pyscf import QoroSolver
        s = QoroSolver(ansatz="uccsd")
        # Pass a dummy circuit so we skip the 'no circuit' RuntimeError and
        # reach the NotImplementedError from the missing get_state_vector.
        qc = QuantumCircuit()
        qc.h(0)
        with pytest.raises(NotImplementedError, match="get_state_vector"):
            s.get_final_statevector(circuit=qc)

    @pytest.mark.skipif(not _has_maestro(), reason="Requires maestro native library")
    def test_compute_statevector_fidelity_works_without_get_state_vector(self):
        """compute_statevector_fidelity must not raise AttributeError."""
        from maestro.circuits import QuantumCircuit
        from qoro_pyscf.backends import configure_backend
        from qoro_pyscf.expectation import compute_statevector_fidelity

        config = configure_backend(use_gpu=False, simulation="statevector")
        qc = QuantumCircuit()
        qc.h(0)
        result = compute_statevector_fidelity(qc, qc, config)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0 + 1e-9

