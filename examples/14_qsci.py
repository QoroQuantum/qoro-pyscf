# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

"""QSCI on N2 dissociation.

N2 in 6-31G, six electrons in eight orbitals (16 qubits, 3136 determinants).
At each bond length a UCCSD VQE from divi's
:class:`~divi.qprog.algorithms.VQEFCISolver` is sampled with 10000 shots;
:class:`~qoro_pyscf.QSCISolver` keeps the 200 most probable bitstrings and
diagonalises the Hamiltonian in the determinants their alpha and beta strings
span. Both energies are compared with exact CASCI.

Reference: Kanno et al., arXiv:2302.11320.
"""

from pyscf import gto, mcscf, scf

from divi.backends import MaestroConfig, MaestroSimulator
from divi.qprog.algorithms import UCCSDAnsatz, VQEFCISolver
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from qoro_pyscf import QSCISolver

N_ORBITALS, N_ELECTRONS = 8, 6
BOND_LENGTHS = (1.1, 1.6, 2.2, 2.7)

if __name__ == "__main__":
    backend = MaestroSimulator(shots=10000, maestro_config=MaestroConfig(seed=7))
    print(
        f"{'R (Å)':>6} {'Determinants':>13} "
        f"{'VQE - CASCI (mHa)':>18} {'QSCI - CASCI (mHa)':>19}"
    )
    for bond_length in BOND_LENGTHS:
        mol = gto.M(atom=f"N 0 0 0; N 0 0 {bond_length}", basis="6-31g", verbose=0)
        mean_field = scf.RHF(mol).run()
        exact = mcscf.CASCI(mean_field, N_ORBITALS, N_ELECTRONS).kernel()[0]

        casci = mcscf.CASCI(mean_field, N_ORBITALS, N_ELECTRONS)
        casci.fcisolver = QSCISolver(
            VQEFCISolver(
                backend=backend,
                optimizer=ScipyOptimizer(ScipyMethod.L_BFGS_B),
                ansatz=UCCSDAnsatz(),
                max_iterations=30,
            ),
            n_samples=200,
        )
        casci.kernel()
        solver = casci.fcisolver
        print(
            f"{bond_length:>6.1f} {solver.n_determinants:>13} "
            f"{1e3 * (solver.vqe_energy - exact):>18.3f} "
            f"{1e3 * (casci.e_tot - exact):>19.3f}"
        )
