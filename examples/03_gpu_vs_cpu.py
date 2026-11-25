# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

"""The same VQE on Maestro's CPU and GPU simulators.

H2 in STO-3G, two electrons in two orbitals (4 qubits), solved by divi's
:class:`~divi.qprog.algorithms.VQEFCISolver` on each simulator, which differ
only in the ``simulator_type`` of :class:`~divi.backends.MaestroConfig`. The
GPU run needs a Maestro GPU licence (``MAESTRO_LICENSE_KEY``).
"""

import time

import maestro
from pyscf import gto, mcscf, scf

from divi.backends import MaestroConfig, MaestroSimulator
from divi.qprog.algorithms import VQEFCISolver
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer


def run_vqe(mean_field: scf.hf.RHF, simulator_type: str) -> tuple[float, float]:
    """Total energy and wall-clock seconds of one seeded VQE."""
    casci = mcscf.CASCI(mean_field, 2, 2)
    casci.fcisolver = VQEFCISolver(
        backend=MaestroSimulator(
            maestro_config=MaestroConfig(simulator_type=simulator_type)
        ),
        optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        n_layers=2,
        max_iterations=100,
        seed=42,
    )
    start = time.perf_counter()
    casci.kernel()
    return casci.e_tot, time.perf_counter() - start


if __name__ == "__main__":
    mol = gto.M(atom="H 0 0 0; H 0 0 0.7414", basis="sto-3g", verbose=0)
    mean_field = scf.RHF(mol).run()
    exact = mcscf.CASCI(mean_field, 2, 2).kernel()[0]

    cpu_energy, cpu_time = run_vqe(mean_field, "QCSim")
    print(f"CPU  {1e3 * (cpu_energy - exact):8.3f} mHa  {cpu_time:.2f} s")
    if not maestro.is_gpu_available():
        print("No Maestro GPU available.")
    else:
        gpu_energy, gpu_time = run_vqe(mean_field, "Gpu")
        print(f"GPU  {1e3 * (gpu_energy - exact):8.3f} mHa  {gpu_time:.2f} s")
        print(f"GPU speedup {cpu_time / gpu_time:.1f}x")
        print(f"|E(GPU) - E(CPU)| = {abs(gpu_energy - cpu_energy):.2e} Ha")
