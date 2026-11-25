# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

"""MPS truncation error against bond dimension.

A linear H6 chain at 1.5 Å spacing in STO-3G, six electrons in six orbitals
(12 qubits). A UCCSD VQE is trained on Maestro's statevector simulator; its
parameters are then evaluated on the matrix product state simulator at bond
dimensions 4 to 64.
"""

from pyscf import gto, mcscf, scf

from divi.backends import MaestroConfig, MaestroSimulator
from divi.qprog.algorithms import VQE, UCCSDAnsatz
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.problems import MolecularProblem

N_ORBITALS = N_ELECTRONS = 6
BOND_DIMENSIONS = (4, 8, 16, 32, 64)


def make_vqe(problem: MolecularProblem, maestro_config: MaestroConfig) -> VQE:
    return VQE(
        problem,
        ansatz=UCCSDAnsatz(),
        optimizer=ScipyOptimizer(ScipyMethod.L_BFGS_B),
        max_iterations=100,
        backend=MaestroSimulator(maestro_config=maestro_config),
    )


if __name__ == "__main__":
    atoms = "; ".join(f"H 0 0 {1.5 * i:.4f}" for i in range(N_ORBITALS))
    mean_field = scf.RHF(gto.M(atom=atoms, basis="sto-3g", verbose=0)).run()
    exact = mcscf.CASCI(mean_field, N_ORBITALS, N_ELECTRONS).kernel()[0]
    problem = MolecularProblem.from_casci(mean_field, N_ORBITALS, N_ELECTRONS)

    trained = make_vqe(problem, MaestroConfig())
    trained.run(perform_final_computation=False)
    params = trained.best_params[None]
    print(f"{'Simulation':<14} {'VQE - CASCI (mHa)':>18} {'MPS - SV (mHa)':>15}")
    print(f"{'Statevector':<14} {1e3 * (trained.best_loss - exact):>18.3f}")
    for chi in BOND_DIMENSIONS:
        vqe = make_vqe(
            problem,
            MaestroConfig(simulation_type="MatrixProductState", max_bond_dimension=chi),
        )
        energy = float(vqe.evaluate(params, vqe.cost_preprocessor())[0][0])
        print(
            f"{f'MPS chi={chi}':<14} {1e3 * (energy - exact):>18.3f} "
            f"{1e3 * (energy - trained.best_loss):>15.3f}"
        )
