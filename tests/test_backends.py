# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

import os

import maestro
import pytest

from qoro_pyscf import configure_backend, set_license_key


def test_rejects_unknown_simulation_mode():
    with pytest.raises(ValueError, match="invalid_mode"):
        configure_backend(simulation="invalid_mode")


def test_statevector_ignores_the_bond_dimension():
    config = configure_backend(simulation="Statevector", mps_bond_dim=32)
    assert config.simulation_type == maestro.SimulationType.Statevector
    assert config.mps_bond_dim is None
    assert config.label == "CPU (statevector)"


def test_mps_mode_keeps_the_bond_dimension():
    config = configure_backend(simulation="mps", mps_bond_dim=32)
    assert config.simulation_type == maestro.SimulationType.MatrixProductState
    assert config.mps_bond_dim == 32


def test_license_key_reaches_the_environment(monkeypatch):
    monkeypatch.setenv("MAESTRO_LICENSE_KEY", "")
    set_license_key("FIRST")
    configure_backend(license_key="SECOND")
    assert os.environ["MAESTRO_LICENSE_KEY"] == "SECOND"
