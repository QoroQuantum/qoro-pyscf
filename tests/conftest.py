# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest

from divi.backends import MaestroConfig, MaestroSimulator


@pytest.fixture
def backend() -> MaestroSimulator:
    return MaestroSimulator(shots=20000, maestro_config=MaestroConfig(seed=7))
