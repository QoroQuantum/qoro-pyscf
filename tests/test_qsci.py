# Copyright 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the QSCI (Quantum-Selected Configuration Interaction) solver.

Unit tests run without maestro; E2E tests require maestro + pyscf + openfermion.
"""

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# Bitstring ↔ determinant helpers (unit tests, no maestro needed)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBitstringHelpers:
    """Test bitstring-to-determinant conversion functions."""

    def test_index_to_bits_zero(self):
        from qoro_maestro_pyscf.qsci_solver import _index_to_bits
        bits = _index_to_bits(0, 4)
        np.testing.assert_array_equal(bits, [0, 0, 0, 0])

    def test_index_to_bits_all_ones(self):
        from qoro_maestro_pyscf.qsci_solver import _index_to_bits
        bits = _index_to_bits(0b1111, 4)
        np.testing.assert_array_equal(bits, [1, 1, 1, 1])

    def test_index_to_bits_specific(self):
        from qoro_maestro_pyscf.qsci_solver import _index_to_bits
        # 0b0101 = 5 → LSB-first: [1, 0, 1, 0]
        bits = _index_to_bits(5, 4)
        np.testing.assert_array_equal(bits, [1, 0, 1, 0])

    def test_bits_to_int_roundtrip(self):
        from qoro_maestro_pyscf.qsci_solver import _index_to_bits, _bits_to_int
        for val in [0, 1, 5, 7, 10, 15]:
            bits = _index_to_bits(val, 4)
            assert _bits_to_int(bits, 4) == val

    def test_bits_to_int_identity(self):
        from qoro_maestro_pyscf.qsci_solver import _bits_to_int
        # [1, 1, 0, 0] → 0b0011 = 3
        assert _bits_to_int(np.array([1, 1, 0, 0]), 4) == 3

    def test_index_to_bits_6qubits(self):
        from qoro_maestro_pyscf.qsci_solver import _index_to_bits
        # 0b101010 = 42 → LSB-first: [0, 1, 0, 1, 0, 1]
        bits = _index_to_bits(42, 6)
        np.testing.assert_array_equal(bits, [0, 1, 0, 1, 0, 1])


# ═══════════════════════════════════════════════════════════════════════════════
# Determinant selection (unit tests, no maestro needed)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeterminantSelection:
    """Test probability → determinant selection logic."""

    def test_hamming_weight_postselection(self):
        """Only bitstrings with correct α/β electron counts should be kept."""
        from qoro_maestro_pyscf.qsci_solver import _probabilities_to_determinants

        # 4 qubits (2 spatial orbitals), 1α + 1β
        # JW interleaved: even = α, odd = β
        # Valid states: those with exactly 1 electron in α (even) and 1 in β (odd)
        n_qubits = 4
        nelec = (1, 1)
        probs = np.zeros(2**n_qubits)

        # |0011⟩ = index 3: bits = [1,1,0,0] → α=[1,0], β=[1,0] → 1α, 1β ✓
        probs[3] = 0.5
        # |1100⟩ = index 12: bits = [0,0,1,1] → α=[0,1], β=[0,1] → 1α, 1β ✓
        probs[12] = 0.3
        # |0001⟩ = index 1: bits = [1,0,0,0] → α=[1,0], β=[0,0] → 1α, 0β ✗
        probs[1] = 0.2

        ci_a, ci_b, sel_probs = _probabilities_to_determinants(
            probs, n_qubits, nelec, n_samples=100, probability_threshold=1e-10,
        )

        # Should have selected exactly the valid configurations
        # Probability sum should be 0.5 + 0.3 = 0.8
        assert np.isclose(np.sum(sel_probs), 0.8)

    def test_top_k_selection(self):
        """n_samples should limit the number of selected configurations."""
        from qoro_maestro_pyscf.qsci_solver import _probabilities_to_determinants

        n_qubits = 4
        nelec = (1, 1)
        probs = np.zeros(2**n_qubits)
        # Two valid states
        probs[3] = 0.6   # |0011⟩
        probs[12] = 0.4  # |1100⟩

        # Request only 1 sample — should keep the highest probability one
        ci_a, ci_b, sel_probs = _probabilities_to_determinants(
            probs, n_qubits, nelec, n_samples=1, probability_threshold=1e-10,
        )

        assert len(sel_probs) == 1
        assert np.isclose(sel_probs[0], 0.6)

    def test_threshold_filtering(self):
        """Probabilities below threshold should be excluded."""
        from qoro_maestro_pyscf.qsci_solver import _probabilities_to_determinants

        n_qubits = 4
        nelec = (1, 1)
        probs = np.zeros(2**n_qubits)
        probs[3] = 0.9
        probs[12] = 1e-12  # Below threshold

        ci_a, ci_b, sel_probs = _probabilities_to_determinants(
            probs, n_qubits, nelec, n_samples=100, probability_threshold=1e-8,
        )

        # Only the high-probability state should be selected
        assert len(sel_probs) == 1

    def test_no_valid_states_raises(self):
        """Should raise if no bitstrings have correct electron counts."""
        from qoro_maestro_pyscf.qsci_solver import _probabilities_to_determinants

        n_qubits = 4
        nelec = (1, 1)
        probs = np.zeros(2**n_qubits)
        # |0001⟩ has 1α, 0β — wrong
        probs[1] = 1.0

        with pytest.raises(ValueError, match="correct electron counts"):
            _probabilities_to_determinants(
                probs, n_qubits, nelec, n_samples=100, probability_threshold=1e-10,
            )

    def test_empty_probs_raises(self):
        """Should raise if no bitstrings above threshold."""
        from qoro_maestro_pyscf.qsci_solver import _probabilities_to_determinants

        n_qubits = 4
        nelec = (1, 1)
        probs = np.zeros(2**n_qubits)

        with pytest.raises(ValueError, match="No bitstrings above"):
            _probabilities_to_determinants(
                probs, n_qubits, nelec, n_samples=100, probability_threshold=1e-10,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Package exports
# ═══════════════════════════════════════════════════════════════════════════════

class TestQSCIExports:
    """Test that QSCISolver is properly exported."""

    def test_qsci_importable(self):
        from qoro_maestro_pyscf import QSCISolver
        assert QSCISolver is not None

    def test_qsci_in_all(self):
        import qoro_maestro_pyscf
        assert "QSCISolver" in qoro_maestro_pyscf.__all__

    def test_qsci_default_init(self):
        from qoro_maestro_pyscf import QSCISolver
        solver = QSCISolver()
        assert solver.n_samples == 500
        assert solver.probability_threshold == 1e-8
        assert solver.verbose is True


# ═══════════════════════════════════════════════════════════════════════════════
# E2E (requires maestro + pyscf + openfermion)
# ═══════════════════════════════════════════════════════════════════════════════

def _can_run_e2e() -> bool:
    try:
        import pyscf, maestro, openfermion  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _can_run_e2e(), reason="Requires pyscf + maestro + openfermion")
class TestQSCIE2E:
    """Full QSCI pipeline tests."""

    def test_h2_qsci_matches_fci(self):
        """QSCI energy for H₂/STO-3G should match FCI."""
        from pyscf import gto, scf, mcscf, fci
        from qoro_maestro_pyscf import MaestroSolver, QSCISolver

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        hf = scf.RHF(mol).run()

        # Exact FCI reference
        cisolver = fci.FCI(hf)
        e_fci = cisolver.kernel()[0]

        # QSCI
        cas = mcscf.CASCI(hf, 2, 2)
        inner = MaestroSolver(ansatz="uccsd", verbose=False)
        cas.fcisolver = QSCISolver(inner_solver=inner, verbose=False)
        e_qsci = cas.kernel()[0]

        # QSCI should match FCI to within 1e-6 Ha for this tiny system
        assert abs(e_qsci - e_fci) < 1e-6, (
            f"QSCI energy {e_qsci:.10f} differs from FCI {e_fci:.10f}"
        )

    def test_qsci_improves_on_vqe(self):
        """QSCI energy should be ≤ VQE energy (variational improvement)."""
        from pyscf import gto, scf, mcscf
        from qoro_maestro_pyscf import MaestroSolver, QSCISolver

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        hf = scf.RHF(mol).run()

        # VQE only
        cas_vqe = mcscf.CASCI(hf, 2, 2)
        cas_vqe.fcisolver = MaestroSolver(ansatz="uccsd", verbose=False)
        e_vqe = cas_vqe.kernel()[0]

        # QSCI
        cas_qsci = mcscf.CASCI(hf, 2, 2)
        inner = MaestroSolver(ansatz="uccsd", verbose=False)
        cas_qsci.fcisolver = QSCISolver(inner_solver=inner, verbose=False)
        e_qsci = cas_qsci.kernel()[0]

        assert e_qsci <= e_vqe + 1e-10, (
            f"QSCI energy {e_qsci:.10f} should be ≤ VQE energy {e_vqe:.10f}"
        )

    def test_qsci_rdm1_trace(self):
        """Trace of 1-RDM should equal number of electrons."""
        from pyscf import gto, scf, mcscf
        from qoro_maestro_pyscf import MaestroSolver, QSCISolver

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        hf = scf.RHF(mol).run()

        cas = mcscf.CASCI(hf, 2, 2)
        inner = MaestroSolver(ansatz="uccsd", verbose=False)
        solver = QSCISolver(inner_solver=inner, verbose=False)
        cas.fcisolver = solver
        cas.kernel()

        rdm1 = solver.make_rdm1(solver, 2, (1, 1))
        assert abs(np.trace(rdm1) - 2.0) < 1e-6
