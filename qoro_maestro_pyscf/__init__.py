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
qoro-maestro-pyscf
==================
PySCF integration plugin for the Maestro GPU quantum simulator (Qoro Quantum).

Enables GPU-accelerated VQE calculations within PySCF's CASCI/CASSCF
framework using Maestro's native backends.

Primary API
-----------
.. autosummary::
    MaestroSolver
    BackendConfig
    configure_backend

Quick Start
-----------
::

    from pyscf import gto, scf, mcscf
    from qoro_maestro_pyscf import MaestroSolver

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    hf  = scf.RHF(mol).run()

    cas = mcscf.CASCI(hf, 2, 2)
    cas.fcisolver = MaestroSolver(ansatz="uccsd")
    cas.run()
"""

from qoro_maestro_pyscf.maestro_solver import MaestroSolver
from qoro_maestro_pyscf.backends import BackendConfig, configure_backend, set_license_key
from qoro_maestro_pyscf.properties import (
    compute_dipole_moment,
    compute_natural_orbitals,
)

__all__ = [
    "MaestroSolver",
    "BackendConfig",
    "configure_backend",
    "set_license_key",
    "compute_dipole_moment",
    "compute_natural_orbitals",
]
__version__ = "0.1.1"
