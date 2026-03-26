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
    """Test bitstring-to-determinant mapping via _probabilities_to_determinants."""

    def test_single_bitstring_maps_correctly(self):
        """A single bitstring should produce the correct α/β determinants."""
        from qoro_pyscf.qsci_solver import _probabilities_to_determinants

        # 4 qubits, (1α, 1β)
        # index 3 = 0b0011, bits=[1,1,0,0], α=[1,0]=1, β=[1,0]=1
        probs = np.zeros(16)
        probs[3] = 1.0
        ci_a, ci_b, _ = _probabilities_to_determinants(
            probs, 4, (1, 1), n_samples=100, probability_threshold=1e-10,
        )
        assert 1 in ci_a   # α string = 0b01 = 1
        assert 1 in ci_b   # β string = 0b01 = 1

    def test_interleaved_mapping(self):
        """Verify JW interleaved qubit → spatial orbital mapping."""
        from qoro_pyscf.qsci_solver import _probabilities_to_determinants

        # index 12 = 0b1100, bits=[0,0,1,1], α=[0,1]=2, β=[0,1]=2
        probs = np.zeros(16)
        probs[12] = 1.0
        ci_a, ci_b, _ = _probabilities_to_determinants(
            probs, 4, (1, 1), n_samples=100, probability_threshold=1e-10,
        )
        assert 2 in ci_a   # α string = 0b10 = 2
        assert 2 in ci_b   # β string = 0b10 = 2

    def test_multiple_bitstrings_unique_dets(self):
        """Two bitstrings with different α/β parts produce distinct determinants."""
        from qoro_pyscf.qsci_solver import _probabilities_to_determinants

        probs = np.zeros(16)
        probs[3] = 0.6   # α=1, β=1
        probs[12] = 0.4  # α=2, β=2
        ci_a, ci_b, _ = _probabilities_to_determinants(
            probs, 4, (1, 1), n_samples=100, probability_threshold=1e-10,
        )
        np.testing.assert_array_equal(sorted(ci_a), [1, 2])
        np.testing.assert_array_equal(sorted(ci_b), [1, 2])

    def test_6qubit_mapping(self):
        """Test correct mapping for 6-qubit (3 spatial orbital) system."""
        from qoro_pyscf.qsci_solver import _probabilities_to_determinants

        # index 21 = 0b010101, bits=[1,0,1,0,1,0]
        # α bits (even): [1,1,1] = 0b111 = 7, β bits (odd): [0,0,0] = 0
        # This has 3α, 0β electrons
        probs = np.zeros(64)
        probs[21] = 1.0
        ci_a, ci_b, _ = _probabilities_to_determinants(
            probs, 6, (3, 0), n_samples=100, probability_threshold=1e-10,
        )
        assert 7 in ci_a   # 0b111 = 7
        assert 0 in ci_b   # 0b000 = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Determinant selection (unit tests, no maestro needed)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeterminantSelection:
    """Test probability → determinant selection logic."""

    def test_hamming_weight_postselection(self):
        """Only bitstrings with correct α/β electron counts should be kept."""
        from qoro_pyscf.qsci_solver import _probabilities_to_determinants

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
        from qoro_pyscf.qsci_solver import _probabilities_to_determinants

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
        from qoro_pyscf.qsci_solver import _probabilities_to_determinants

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
        from qoro_pyscf.qsci_solver import _probabilities_to_determinants

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
        from qoro_pyscf.qsci_solver import _probabilities_to_determinants

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
        from qoro_pyscf import QSCISolver
        assert QSCISolver is not None

    def test_qsci_in_all(self):
        import qoro_pyscf
        assert "QSCISolver" in qoro_pyscf.__all__

    def test_qsci_default_init(self):
        from qoro_pyscf import QSCISolver
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
        from qoro_pyscf import QoroSolver, QSCISolver

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        hf = scf.RHF(mol).run()

        # Exact FCI reference
        cisolver = fci.FCI(hf)
        e_fci = cisolver.kernel()[0]

        # QSCI
        cas = mcscf.CASCI(hf, 2, 2)
        inner = QoroSolver(ansatz="uccsd", verbose=False)
        cas.fcisolver = QSCISolver(inner_solver=inner, verbose=False)
        e_qsci = cas.kernel()[0]

        # QSCI should match FCI to within 1e-6 Ha for this tiny system
        assert abs(e_qsci - e_fci) < 1e-6, (
            f"QSCI energy {e_qsci:.10f} differs from FCI {e_fci:.10f}"
        )

    def test_qsci_improves_on_vqe(self):
        """QSCI energy should be ≤ VQE energy (variational improvement)."""
        from pyscf import gto, scf, mcscf
        from qoro_pyscf import QoroSolver, QSCISolver

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        hf = scf.RHF(mol).run()

        # VQE only
        cas_vqe = mcscf.CASCI(hf, 2, 2)
        cas_vqe.fcisolver = QoroSolver(ansatz="uccsd", verbose=False)
        e_vqe = cas_vqe.kernel()[0]

        # QSCI
        cas_qsci = mcscf.CASCI(hf, 2, 2)
        inner = QoroSolver(ansatz="uccsd", verbose=False)
        cas_qsci.fcisolver = QSCISolver(inner_solver=inner, verbose=False)
        e_qsci = cas_qsci.kernel()[0]

        assert e_qsci <= e_vqe + 1e-10, (
            f"QSCI energy {e_qsci:.10f} should be ≤ VQE energy {e_vqe:.10f}"
        )

    def test_qsci_rdm1_trace(self):
        """Trace of 1-RDM should equal number of electrons."""
        from pyscf import gto, scf, mcscf
        from qoro_pyscf import QoroSolver, QSCISolver

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        hf = scf.RHF(mol).run()

        cas = mcscf.CASCI(hf, 2, 2)
        inner = QoroSolver(ansatz="uccsd", verbose=False)
        solver = QSCISolver(inner_solver=inner, verbose=False)
        cas.fcisolver = solver
        cas.kernel()

        rdm1 = solver.make_rdm1(solver, 2, (1, 1))
        assert abs(np.trace(rdm1) - 2.0) < 1e-6
