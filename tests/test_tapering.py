# Copyright 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Z₂ qubit tapering."""

import numpy as np
import pytest


class TestTaperingUnit:
    """Tests for tapering.py."""

    def test_find_z2_symmetries(self):
        """find_z2_symmetries returns 2 stabilizers for H₂."""
        from qoro_maestro_pyscf.tapering import find_z2_symmetries
        from openfermion import QubitOperator

        stabilizers = find_z2_symmetries(n_qubits=4, nelec=(1, 1))
        assert len(stabilizers) == 2
        for s in stabilizers:
            assert isinstance(s, QubitOperator)

    def test_taper_hamiltonian_reduces_qubits(self):
        """Tapering should reduce from 4 qubits."""
        from openfermion import QubitOperator
        from qoro_maestro_pyscf.tapering import taper_hamiltonian

        H = (
            QubitOperator("Z0", 0.5)
            + QubitOperator("Z1", -0.3)
            + QubitOperator("Z0 Z1", 0.2)
            + QubitOperator("X0 X1", 0.1)
            + QubitOperator("", -1.0)
        )

        result = taper_hamiltonian(H, n_qubits=4, nelec=(1, 1))

        assert result.tapered_n_qubits < result.original_n_qubits
        assert result.original_n_qubits == 4
        assert len(result.removed_positions) > 0
        assert len(result.stabilizers) == 2

    def test_tapering_result_structure(self):
        """TaperingResult has correct dataclass fields."""
        from qoro_maestro_pyscf import TaperingResult
        import dataclasses

        fields = {f.name for f in dataclasses.fields(TaperingResult)}
        assert "tapered_op" in fields
        assert "original_n_qubits" in fields
        assert "tapered_n_qubits" in fields
        assert "removed_positions" in fields
        assert "stabilizers" in fields

    def test_custom_stabilizers(self):
        """Custom stabilizers are used instead of auto-detected ones."""
        from openfermion import QubitOperator
        from qoro_maestro_pyscf.tapering import taper_hamiltonian

        H = QubitOperator("Z0", 1.0) + QubitOperator("", -0.5)
        custom_stab = [QubitOperator("Z0")]

        result = taper_hamiltonian(
            H, n_qubits=2, nelec=(1, 0), stabilizers=custom_stab
        )
        assert result.tapered_n_qubits < 2
