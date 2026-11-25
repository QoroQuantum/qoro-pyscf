# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

"""ADAPT-VQE on LiH against UCCSD.

LiH in STO-3G, four electrons in four orbitals (8 qubits). ADAPT-VQE grows its
circuit from a pool of single and double excitations, one operator at a time,
until no operator's energy gradient exceeds the threshold. UCCSD on divi's
:class:`~divi.qprog.algorithms.VQEFCISolver` uses every excitation at once.

Reference: Grimsley et al., Nature Communications 10, 3007 (2019).
"""

import time

from pyscf import gto, mcscf, scf

from divi.backends import MaestroSimulator
from divi.qprog.algorithms import UCCSDAnsatz, VQEFCISolver
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.problems import MolecularProblem
from qoro_pyscf import build_pauli_terms, configure_backend, run_adapt_vqe

N_ORBITALS, N_ELECTRONS = 4, (2, 2)

if __name__ == "__main__":
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    mean_field = scf.RHF(mol).run()
    exact = mcscf.CASCI(mean_field, N_ORBITALS, N_ELECTRONS).kernel()[0]

    start = time.perf_counter()
    problem = MolecularProblem.from_casci(mean_field, N_ORBITALS, N_ELECTRONS)
    adapt = run_adapt_vqe(
        2 * N_ORBITALS,
        N_ELECTRONS,
        *build_pauli_terms(problem.hamiltonian),
        configure_backend(),
    )
    adapt_time = time.perf_counter() - start

    start = time.perf_counter()
    casci = mcscf.CASCI(mean_field, N_ORBITALS, N_ELECTRONS)
    casci.fcisolver = VQEFCISolver(
        backend=MaestroSimulator(),
        optimizer=ScipyOptimizer(ScipyMethod.L_BFGS_B),
        ansatz=UCCSDAnsatz(),
        max_iterations=100,
    )
    casci.kernel()
    uccsd_time = time.perf_counter() - start

    print(f"{'Method':<10} {'Error (mHa)':>12} {'Parameters':>11} {'Time (s)':>9}")
    print(
        f"{'ADAPT':<10} {1e3 * (adapt['energy'] - exact):>12.4f} "
        f"{adapt['n_operators']:>11} {adapt_time:>9.1f}"
    )
    print(
        f"{'UCCSD':<10} {1e3 * (casci.e_tot - exact):>12.4f} "
        f"{casci.ci.n_params:>11} {uccsd_time:>9.1f}"
    )
