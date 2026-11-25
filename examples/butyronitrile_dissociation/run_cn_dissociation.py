# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Butyronitrile C≡N dissociation with VQE on Maestro.

The nitrogen of butyronitrile (CH3CH2CH2C≡N) is pulled away from its carbon
along the bond, all other atoms fixed, from 1.157 Å to 2.749 Å.

  Stage 0: CAS(4,4)/STO-3G, 8 qubits, UCCSD with COBYLA on divi.
  Stage 1: CAS(8,8)/STO-3G, 16 qubits, greedy ADAPT-VQE over doubles.
  Stage 2: CAS(14,14)/6-31G*, 28 qubits, UCCSD with Adam on divi.

Each point also gets HF, MP2 and exact CASCI where feasible. Results are saved
to ``results/`` as JSON for ``plot_results.py``. The GPU needs a Maestro GPU
licence (``MAESTRO_LICENSE_KEY``).

Usage::

    python run_cn_dissociation.py --stage 0 --cpu --quick
    python run_cn_dissociation.py --stage 1 --both
    python run_cn_dissociation.py --stage both --gpu
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from pyscf import gto, mcscf, mp, scf

from divi.backends import MaestroConfig, MaestroSimulator
from divi.qprog.algorithms import UCCSDAnsatz, VQEFCISolver
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.problems import MolecularProblem
from qoro_pyscf import AdamOptimizer, build_pauli_terms, configure_backend, run_adapt_vqe

EQUILIBRIUM_ATOMS = [
    ("C", (0.71951, -0.72284, 0.00000)),
    ("C", (0.00000, 0.63021, 0.00000)),
    ("C", (-1.45592, 0.49995, 0.00000)),
    ("N", (-2.20171, 0.42056, 0.00000)),
    ("C", (2.23248, -0.56128, 0.00000)),
    ("H", (0.39949, -1.29704, 0.87880)),
    ("H", (0.39949, -1.29704, -0.87880)),
    ("H", (0.28619, 1.21996, 0.88101)),
    ("H", (0.28619, 1.21996, -0.88101)),
    ("H", (2.72786, -1.53832, 0.00000)),
    ("H", (2.57639, -0.01393, -0.88677)),
    ("H", (2.57639, -0.01393, 0.88677)),
]
NITROGEN = 3
CARBON = np.array(EQUILIBRIUM_ATOMS[2][1])
CN_DIRECTION = np.array(EQUILIBRIUM_ATOMS[NITROGEN][1]) - CARBON
CN_DIRECTION /= np.linalg.norm(CN_DIRECTION)
CN_DISTANCES = [1.157, 1.356, 1.555, 1.754, 1.953, 2.152, 2.351, 2.550, 2.749]

STAGES = {
    0: {
        "label": "Stage 0 — CPU Quick",
        "norb": 4,
        "nelec": 4,
        "basis": "sto-3g",
        "ansatz": "uccsd",
        "optimizer": "COBYLA",
        "maxiter": 400,
    },
    1: {
        "label": "Stage 1 — Baseline",
        "norb": 8,
        "nelec": 8,
        "basis": "sto-3g",
        "ansatz": "adapt",
        "optimizer": "ROTOSOLVE",
        "maxiter": 100,
    },
    2: {
        "label": "Stage 2 — Extended",
        "norb": 14,
        "nelec": 14,
        "basis": "6-31g*",
        "ansatz": "uccsd",
        "optimizer": "adam",
        "maxiter": 800,
    },
}
SIMULATORS = {"cpu": "QCSim", "gpu": "Gpu"}


def build_butyronitrile(cn_distance: float, basis: str) -> gto.Mole:
    """Butyronitrile with the C≡N bond at ``cn_distance`` Å."""
    atoms = []
    for index, (element, coords) in enumerate(EQUILIBRIUM_ATOMS):
        if index == NITROGEN:
            coords = CARBON + CN_DIRECTION * cn_distance
        atoms.append(f"{element} {coords[0]:.5f} {coords[1]:.5f} {coords[2]:.5f}")
    return gto.M(atom="\n".join(atoms), basis=basis, verbose=0)


def make_solver(stage: dict, device: str) -> VQEFCISolver:
    """A UCCSD solver, reused across geometries so each point warm-starts."""
    optimizer = (
        AdamOptimizer(learning_rate=0.005)
        if stage["optimizer"] == "adam"
        else ScipyOptimizer(ScipyMethod.COBYLA)
    )
    return VQEFCISolver(
        backend=MaestroSimulator(
            maestro_config=MaestroConfig(simulator_type=SIMULATORS[device])
        ),
        optimizer=optimizer,
        ansatz=UCCSDAnsatz(),
        max_iterations=stage["maxiter"],
    )


def run_adapt(mean_field: scf.hf.RHF, stage: dict, device: str) -> float:
    """Greedy ADAPT-VQE energy over the doubles pool."""
    norb, nelec = stage["norb"], stage["nelec"]
    problem = MolecularProblem.from_casci(mean_field, norb, nelec)
    result = run_adapt_vqe(
        2 * norb,
        (problem.n_alpha, problem.n_beta),
        *build_pauli_terms(problem.hamiltonian),
        configure_backend(use_gpu=device == "gpu"),
        pool="d",
        greedy=True,
        maxiter_per_step=stage["maxiter"],
    )
    return result["energy"]


def run_stage(stage_number: int, devices: list[str], quick: bool) -> dict:
    stage = STAGES[stage_number]
    distances = CN_DISTANCES[:3] if quick else CN_DISTANCES
    norb, nelec, basis = stage["norb"], stage["nelec"], stage["basis"]
    print(f"\n{stage['label']}: CAS({nelec},{norb})/{basis}, {2 * norb} qubits")

    results = {
        "metadata": {
            "stage": stage_number,
            "label": stage["label"],
            "basis": basis,
            "norb": norb,
            "nelec": nelec,
            "n_qubits": 2 * norb,
            "ansatz": stage["ansatz"],
            "optimizer": stage["optimizer"],
            "maxiter": stage["maxiter"],
        },
        "cn_distances": distances,
        "hf_energies": [],
        "fci_energies": [],
        "mp2_energies": [],
    }
    mean_fields = []
    print(f"{'d(C≡N)':>8} {'HF':>14} {'MP2':>14} {'CASCI':>14}")
    for distance in distances:
        mean_field = scf.RHF(build_butyronitrile(distance, basis)).run()
        mean_fields.append(mean_field)
        results["hf_energies"].append(mean_field.e_tot)
        results["mp2_energies"].append(mp.MP2(mean_field).run().e_tot)
        results["fci_energies"].append(mcscf.CASCI(mean_field, norb, nelec).kernel()[0])
        print(
            f"{distance:>8.3f} {mean_field.e_tot:>14.8f} "
            f"{results['mp2_energies'][-1]:>14.8f} {results['fci_energies'][-1]:>14.8f}"
        )

    for device in devices:
        energies, times = [], []
        solver = None if stage["ansatz"] == "adapt" else make_solver(stage, device)
        for distance, mean_field, exact in zip(
            distances, mean_fields, results["fci_energies"]
        ):
            start = time.perf_counter()
            if solver is None:
                energy = run_adapt(mean_field, stage, device)
            else:
                casci = mcscf.CASCI(mean_field, norb, nelec)
                casci.fcisolver = solver
                energy = casci.kernel()[0]
            times.append(time.perf_counter() - start)
            energies.append(float(energy))
            print(
                f"{device.upper()} {distance:.3f} Å: {1e3 * (energy - exact):8.2f} mHa, "
                f"{times[-1]:.1f} s"
            )
        results[f"vqe_{device}_energies"] = energies
        results[f"vqe_{device}_times"] = times

    if len(devices) == 2:
        speedup = np.mean(results["vqe_cpu_times"]) / np.mean(results["vqe_gpu_times"])
        print(f"GPU speedup {speedup:.1f}x")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    name = f"butyronitrile_stage{stage_number}_{basis.replace('*', 'star')}_cas{norb}"
    if quick:
        name += "_quick"
    out_path = out_dir / f"{name}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cpu", action="store_true")
    mode.add_argument("--gpu", action="store_true")
    mode.add_argument("--both", action="store_true")
    parser.add_argument("--stage", choices=["0", "1", "2", "both"], required=True)
    parser.add_argument("--quick", action="store_true", help="Only 3 geometries")
    args = parser.parse_args()

    devices = ["cpu", "gpu"] if args.both else ["gpu"] if args.gpu else ["cpu"]
    stages = [1, 2] if args.stage == "both" else [int(args.stage)]
    for stage_number in stages:
        run_stage(stage_number, devices, args.quick)
