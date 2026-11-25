# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Maestro CPU against GPU on hydrogen chains with UpCCD.

Linear H4 to H10 chains at 1.5 Å spacing in STO-3G, with every orbital active
(8 to 20 qubits), solved by divi's
:class:`~divi.qprog.algorithms.VQEFCISolver` with
:class:`~divi.qprog.algorithms.UpCCDAnsatz`. The GPU needs a Maestro GPU
licence (``MAESTRO_LICENSE_KEY``).

Usage::

    python 10_gpu_benchmark.py --both
    python 10_gpu_benchmark.py --gpu --max-atoms 8 --simulation mps --bond-dim 64
"""

import argparse
import time

from pyscf import gto, mcscf, scf

from divi.backends import MaestroConfig, MaestroSimulator
from divi.qprog.algorithms import UpCCDAnsatz, VQEFCISolver
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer

SIMULATORS = {"cpu": "QCSim", "gpu": "Gpu"}


def run_chain(
    mean_field: scf.hf.RHF, maestro_config: MaestroConfig, max_iterations: int
) -> tuple[float, float, int]:
    """Total energy, wall-clock seconds and parameter count of one VQE."""
    n_atoms = mean_field.mol.natm
    casci = mcscf.CASCI(mean_field, n_atoms, n_atoms)
    casci.fcisolver = VQEFCISolver(
        backend=MaestroSimulator(maestro_config=maestro_config),
        optimizer=ScipyOptimizer(ScipyMethod.L_BFGS_B),
        ansatz=UpCCDAnsatz(),
        max_iterations=max_iterations,
    )
    start = time.perf_counter()
    casci.kernel()
    return casci.e_tot, time.perf_counter() - start, casci.ci.n_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cpu", action="store_true")
    mode.add_argument("--gpu", action="store_true")
    mode.add_argument("--both", action="store_true")
    parser.add_argument("--max-atoms", type=int, default=10)
    parser.add_argument(
        "--simulation", choices=["statevector", "mps"], default="statevector"
    )
    parser.add_argument("--bond-dim", type=int, default=64)
    parser.add_argument("--max-iterations", type=int, default=300)
    args = parser.parse_args()

    devices = ["cpu", "gpu"] if args.both else ["gpu"] if args.gpu else ["cpu"]
    options = (
        {"simulation_type": "MatrixProductState", "max_bond_dimension": args.bond_dim}
        if args.simulation == "mps"
        else {}
    )
    print(
        f"{'Chain':<6} {'Qubits':>6} {'Params':>6}"
        + "".join(f" {device + ' s':>8} {'mHa':>8}" for device in devices)
        + (f" {'Speedup':>8}" if len(devices) == 2 else "")
    )
    for n_atoms in (n for n in (4, 6, 8, 10) if n <= args.max_atoms):
        atoms = "; ".join(f"H 0 0 {1.5 * i}" for i in range(n_atoms))
        mean_field = scf.RHF(gto.M(atom=atoms, basis="sto-3g", verbose=0)).run()
        exact = mcscf.CASCI(mean_field, n_atoms, n_atoms).kernel()[0]
        row, times = "", {}
        for device in devices:
            config = MaestroConfig(simulator_type=SIMULATORS[device], **options)
            energy, times[device], n_params = run_chain(
                mean_field, config, args.max_iterations
            )
            row += f" {times[device]:>8.2f} {1e3 * (energy - exact):>8.3f}"
        if len(devices) == 2:
            row += f" {times['cpu'] / times['gpu']:>7.1f}x"
        print(f"H{n_atoms:<5} {2 * n_atoms:>6} {n_params:>6}{row}")
