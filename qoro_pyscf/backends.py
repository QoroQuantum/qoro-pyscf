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
Backend configuration for Qoro quantum simulation.

Handles GPU/CPU detection, simulation type selection (Statevector / MPS),
license key management, and provides a portable configuration object used
throughout the library.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class SimulationMode(str, Enum):
    """Supported simulation modes."""
    STATEVECTOR = "statevector"
    MPS = "mps"


@dataclass
class BackendConfig:
    """
    Configuration for the Qoro simulation backend.

    Attributes
    ----------
    simulator_type : maestro.SimulatorType
        The simulator backend (GPU or QCSim CPU).
    simulation_type : maestro.SimulationType
        The simulation method (Statevector or MPS).
    label : str
        Human-readable label for logging.
    mps_bond_dim : int or None
        Bond dimension for MPS simulation (ignored for Statevector).
    """
    simulator_type: Any  # maestro.SimulatorType
    simulation_type: Any  # maestro.SimulationType
    label: str
    mps_bond_dim: Optional[int] = None


def set_license_key(key: str) -> None:
    """
    Set the Qoro GPU license key.

    The key is stored as the ``MAESTRO_LICENSE_KEY`` environment variable,
    which Qoro reads automatically during ``init_gpu()``.

    Parameters
    ----------
    key : str
        Your Qoro license key (e.g. ``"XXXX-XXXX-XXXX-XXXX"``).

    Notes
    -----
    - Get your key instantly at https://maestro.qoroquantum.net
    - First activation requires an internet connection (one-time).
    - Subsequent runs work offline for up to 30 days.
    - Contact team@qoroquantum.de for licensing issues.

    Examples
    --------
    >>> from qoro_pyscf import configure_backend
    >>> from qoro_pyscf.backends import set_license_key
    >>> set_license_key("XXXX-XXXX-XXXX-XXXX")
    >>> cfg = configure_backend()  # Will now use GPU with your license
    """
    os.environ["MAESTRO_LICENSE_KEY"] = key


def configure_backend(
    use_gpu: bool = False,
    simulation: str = "statevector",
    mps_bond_dim: int = 64,
    license_key: Optional[str] = None,
) -> BackendConfig:
    """
    Create a Qoro backend configuration with automatic fallback.

    Parameters
    ----------
    use_gpu : bool
        If True, attempt to use the GPU backend (requires a license key).
        Falls back to CPU if unavailable. Default: False (CPU).
    simulation : str
        Simulation mode: ``"statevector"`` or ``"mps"`` (Matrix Product State).
        MPS enables larger qubit counts when entanglement is bounded.
    mps_bond_dim : int
        Bond dimension for MPS simulation. Ignored for statevector mode.
        Higher values = more accurate but slower. Default: 64.
    license_key : str or None
        Qoro GPU license key. If provided, sets the ``MAESTRO_LICENSE_KEY``
        environment variable before GPU initialisation. Can also be set via
        :func:`set_license_key` or directly as an env var.

    Returns
    -------
    BackendConfig
        Configured backend ready for use with Qoro.

    Examples
    --------
    CPU statevector (default, no license needed):

    >>> cfg = configure_backend()

    GPU statevector (requires license):

    >>> cfg = configure_backend(use_gpu=True, license_key="XXXX-XXXX-XXXX-XXXX")

    GPU MPS for larger circuits:

    >>> cfg = configure_backend(use_gpu=True, simulation="mps", mps_bond_dim=128)
    """
    # --- Set license key if provided ---
    if license_key is not None:
        set_license_key(license_key)

    # --- Resolve simulation type ---
    import maestro

    mode = SimulationMode(simulation.lower())

    if mode == SimulationMode.MPS:
        sim_type = maestro.SimulationType.MatrixProductState
        bond_dim = mps_bond_dim
    else:
        sim_type = maestro.SimulationType.Statevector
        bond_dim = None

    # --- Resolve simulator backend (GPU with fallback) ---
    if use_gpu and maestro.is_gpu_available():
        maestro.init_gpu()
        sim_backend = maestro.SimulatorType.Gpu
        backend_label = f"GPU ({mode.value})"
    else:
        sim_backend = maestro.SimulatorType.QCSim
        backend_label = f"CPU ({mode.value})"

    return BackendConfig(
        simulator_type=sim_backend,
        simulation_type=sim_type,
        label=backend_label,
        mps_bond_dim=bond_dim,
    )
