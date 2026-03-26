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
qoro-pyscf
==================
PySCF integration plugin for the Qoro quantum simulator by Qoro Quantum.

Run VQE calculations within PySCF's CASCI/CASSCF framework — works on
CPU out of the box, with optional GPU acceleration for speed.

Primary API
-----------
.. autosummary::
    QoroSolver
    BackendConfig
    configure_backend

Quick Start
-----------
::

    from pyscf import gto, scf, mcscf
    from qoro_pyscf import QoroSolver

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    hf  = scf.RHF(mol).run()

    cas = mcscf.CASCI(hf, 2, 2)
    cas.fcisolver = QoroSolver(ansatz="uccsd")
    cas.run()
"""

from qoro_pyscf.qoro_solver import QoroSolver
from qoro_pyscf.qsci_solver import QSCISolver
from qoro_pyscf.vqd_solver import VQDSolver
from qoro_pyscf.backends import BackendConfig, configure_backend, set_license_key
from qoro_pyscf.expectation import (
    get_state_probabilities,
    compute_state_fidelity,
    compute_statevector_fidelity,
    compute_overlap,
)
from qoro_pyscf.properties import (
    compute_dipole_moment,
    compute_natural_orbitals,
)
from qoro_pyscf.active_space import (
    suggest_active_space,
    suggest_active_space_from_mp2,
)
from qoro_pyscf.tapering import taper_hamiltonian, TaperingResult

__all__ = [
    "QoroSolver",
    "QSCISolver",
    "VQDSolver",
    "BackendConfig",
    "configure_backend",
    "set_license_key",
    "get_state_probabilities",
    "compute_state_fidelity",
    "compute_statevector_fidelity",
    "compute_overlap",
    "compute_dipole_moment",
    "compute_natural_orbitals",
    "suggest_active_space",
    "suggest_active_space_from_mp2",
    "taper_hamiltonian",
    "TaperingResult",
]
__version__ = "0.6.0"
