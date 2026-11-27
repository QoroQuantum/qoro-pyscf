# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

"""H2 bond length from VQE nuclear gradients.

H2 in STO-3G starts stretched to 1.2 Å. At each geometry divi's
:class:`~divi.qprog.algorithms.VQEFCISolver` solves CAS(2,2) and PySCF's
CASCI gradient turns its RDMs into ``dE/dR``; SciPy's BFGS minimises the
energy over the bond length with that gradient.
"""

import numpy as np
from pyscf import gto, mcscf, scf
from pyscf.data.nist import BOHR
from scipy.optimize import minimize

from divi.backends import MaestroSimulator
from divi.qprog.algorithms import UCCSDAnsatz, VQEFCISolver
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer


def compute_energy_and_gradient(
    bond_length: np.ndarray, solver: VQEFCISolver
) -> tuple[float, np.ndarray]:
    """Energy (Ha) and ``dE/dR`` (Ha/Å) of H2 at ``bond_length`` Å."""
    (distance,) = bond_length
    mol = gto.M(atom=f"H 0 0 0; H 0 0 {distance}", basis="sto-3g", verbose=0)
    casci = mcscf.CASCI(scf.RHF(mol).run(), 2, 2)
    casci.fcisolver = solver
    casci.kernel()
    gradient = casci.nuc_grad_method().kernel()
    print(f"  R = {distance:.4f} Å  E = {casci.e_tot:.8f} Ha")
    return casci.e_tot, np.array([(gradient[1, 2] - gradient[0, 2]) / 2 / BOHR])


if __name__ == "__main__":
    solver = VQEFCISolver(
        backend=MaestroSimulator(),
        optimizer=ScipyOptimizer(ScipyMethod.L_BFGS_B),
        ansatz=UCCSDAnsatz(),
        max_iterations=50,
    )
    result = minimize(
        compute_energy_and_gradient,
        x0=[1.2],
        args=(solver,),
        jac=True,
        method="BFGS",
        options={"gtol": 1e-5},
    )
    print(f"Optimised bond length {result.x[0]:.4f} Å, experiment 0.7414 Å")
