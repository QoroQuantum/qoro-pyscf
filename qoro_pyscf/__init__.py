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
==========
PySCF tools not yet in divi: a VQD active-space solver, ADAPT-VQE, active-space
selection, properties and qubit tapering.
Ground-state VQE and QSCI solvers live in divi
(:class:`divi.qprog.algorithms.VQEFCISolver`,
:class:`divi.qprog.ensembles.qsci.QSCISolver`).
"""

from qoro_pyscf.active_space import (
    suggest_active_space,
    suggest_active_space_from_mp2,
)
from qoro_pyscf.adapt import build_pauli_terms, run_adapt_vqe
from qoro_pyscf.backends import BackendConfig, configure_backend, set_license_key
from qoro_pyscf.properties import (
    compute_dipole_moment,
    compute_mulliken_spin_population,
    compute_natural_orbitals,
)
from qoro_pyscf.tapering import TaperingResult, taper_hamiltonian
from qoro_pyscf.vqd_solver import VQDSolver

__all__ = [
    "BackendConfig",
    "TaperingResult",
    "VQDSolver",
    "build_pauli_terms",
    "compute_dipole_moment",
    "compute_mulliken_spin_population",
    "compute_natural_orbitals",
    "configure_backend",
    "run_adapt_vqe",
    "set_license_key",
    "suggest_active_space",
    "suggest_active_space_from_mp2",
    "taper_hamiltonian",
]
__version__ = "0.6.1"
