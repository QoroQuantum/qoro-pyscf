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
PySCF tools not yet in divi: a VQD active-space solver and qubit tapering.
Ground-state VQE, ADAPT-VQE and QSCI live in divi
(:class:`divi.qprog.algorithms.VQEFCISolver`,
:class:`divi.qprog.algorithms.AdaptVQE`,
:class:`divi.qprog.ensembles.qsci.QSCISolver`).
"""

from qoro_pyscf.tapering import TaperingResult, taper_hamiltonian
from qoro_pyscf.vqd_solver import VQDSolver

__all__ = [
    "TaperingResult",
    "VQDSolver",
    "taper_hamiltonian",
]
__version__ = "0.6.1"
