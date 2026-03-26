#!/usr/bin/env python3
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
Example 10 — GPU vs CPU Benchmark (Hydrogen Chains)
====================================================

Benchmarks Qoro GPU vs CPU performance on hydrogen chains of increasing
size using the UpCCD ansatz.  The active space grows with chain length,
pushing the system into a regime where GPU acceleration matters.

This is designed to be run on a machine with an NVIDIA GPU and a Qoro
license key.

Benchmark systems
-----------------
| Chain | Active Space | Qubits | UpCCD Params |
|-------|-------------|--------|-------------|
| H₄   | CAS(4,4)    |   8    |     4       |
| H₆   | CAS(6,6)    |  12    |     9       |
| H₈   | CAS(8,8)    |  16    |    16       |
| H₁₀  | CAS(10,10)  |  20    |    25       |

At 16–20 qubits, statevector simulation involves 2^16–2^20 amplitudes per
circuit evaluation.  GPU parallelism delivers real speedups here.

Usage
-----
    # CPU only (baseline)
    python 10_gpu_benchmark.py --cpu

    # GPU only
    python 10_gpu_benchmark.py --gpu

    # Both — full comparison (recommended)
    python 10_gpu_benchmark.py --both

    # Control which chain sizes to run
    python 10_gpu_benchmark.py --both --max-atoms 8

    # MPS mode benchmark
    python 10_gpu_benchmark.py --gpu --simulation mps --bond-dim 64
"""

import argparse
import time

import numpy as np
from pyscf import gto, scf, mcscf

from qoro_pyscf import QoroSolver
from qoro_pyscf.ansatze import upccd_param_count


def build_hydrogen_chain(n_atoms: int, spacing: float = 1.5) -> gto.Mole:
    """Build a linear hydrogen chain with equal spacing (in Å)."""
    atoms = "; ".join(f"H 0 0 {i * spacing}" for i in range(n_atoms))
    return gto.M(atom=atoms, basis="sto-3g", verbose=0)


def run_benchmark(n_atoms, backend, simulation, mps_bond_dim, maxiter):
    """Run a single benchmark: returns (energy, fci_energy, time, converged)."""
    mol = build_hydrogen_chain(n_atoms)
    hf_obj = scf.RHF(mol).run()

    norb = n_atoms
    nelec = n_atoms
    n_qubits = 2 * norb

    # FCI reference
    cas_fci = mcscf.CASCI(hf_obj, norb, nelec)
    cas_fci.verbose = 0
    fci_e = cas_fci.kernel()[0]

    # VQE with UpCCD
    cas_vqe = mcscf.CASCI(hf_obj, norb, nelec)
    cas_vqe.verbose = 0
    cas_vqe.fcisolver = QoroSolver(
        ansatz="upccd",
        backend=backend,
        simulation=simulation,
        mps_bond_dim=mps_bond_dim,
        maxiter=maxiter,
        verbose=False,
    )

    t0 = time.perf_counter()
    vqe_e = cas_vqe.kernel()[0]
    elapsed = time.perf_counter() - t0

    converged = cas_vqe.fcisolver.converged
    return vqe_e, fci_e, elapsed, converged


def main():
    parser = argparse.ArgumentParser(
        description="GPU vs CPU benchmark on hydrogen chains"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--gpu", action="store_true", help="Run GPU only")
    mode.add_argument("--cpu", action="store_true", help="Run CPU only")
    mode.add_argument("--both", action="store_true", help="Run both and compare")
    parser.add_argument(
        "--max-atoms", type=int, default=10,
        help="Largest H-chain to benchmark (default: 10)"
    )
    parser.add_argument(
        "--simulation", type=str, default="statevector",
        choices=["statevector", "mps"],
        help="Simulation mode (default: statevector)"
    )
    parser.add_argument(
        "--bond-dim", type=int, default=64,
        help="MPS bond dimension (default: 64, only used with --simulation mps)"
    )
    parser.add_argument(
        "--maxiter", type=int, default=300,
        help="Max VQE iterations (default: 300)"
    )
    args = parser.parse_args()

    chain_sizes = [n for n in [4, 6, 8, 10] if n <= args.max_atoms]
    backends = []
    if args.both:
        backends = ["cpu", "gpu"]
    elif args.gpu:
        backends = ["gpu"]
    else:
        backends = ["cpu"]

    sim_label = args.simulation.upper()
    if args.simulation == "mps":
        sim_label += f" (bond_dim={args.bond_dim})"

    print("=" * 80)
    print("  QORO GPU vs CPU BENCHMARK — HYDROGEN CHAINS + UpCCD")
    print(f"  Simulation: {sim_label}")
    print(f"  Backends:   {', '.join(b.upper() for b in backends)}")
    print(f"  Chains:     {', '.join(f'H{n}' for n in chain_sizes)}")
    print("=" * 80)

    # Header
    print(f"\n  {'Chain':<6s}  {'Qubits':>6s}  {'Params':>6s}  ", end="")
    for b in backends:
        print(f"{'Time ('+b.upper()+')':>12s}  {'Error (mHa)':>11s}  ", end="")
    if len(backends) == 2:
        print(f"{'Speedup':>8s}", end="")
    print()
    print(f"  {'─' * (6 + 8 + 8 + len(backends) * 25 + (8 if len(backends) == 2 else 0))}")

    for n_atoms in chain_sizes:
        n_qubits = 2 * n_atoms
        nelec = (n_atoms // 2, n_atoms // 2)
        n_params = upccd_param_count(n_qubits, nelec)

        results = {}
        for backend in backends:
            print(f"  Running H{n_atoms} on {backend.upper()}...", end="", flush=True)
            try:
                vqe_e, fci_e, elapsed, converged = run_benchmark(
                    n_atoms, backend, args.simulation, args.bond_dim, args.maxiter
                )
                error = abs(vqe_e - fci_e) * 1000
                results[backend] = (elapsed, error, converged)
                print(f" done ({elapsed:.1f}s)")
            except Exception as e:
                results[backend] = None
                print(f" FAILED: {e}")

        # Print row
        print(f"  H{n_atoms:<5d} {n_qubits:>6d}  {n_params:>6d}  ", end="")
        for b in backends:
            if results[b] is not None:
                elapsed, error, converged = results[b]
                marker = "" if converged else " ✗"
                print(f"{elapsed:>10.2f}s  {error:>10.4f}{marker}  ", end="")
            else:
                print(f"{'FAILED':>12s}  {'—':>11s}  ", end="")

        if len(backends) == 2 and all(results[b] is not None for b in backends):
            speedup = results["cpu"][0] / results["gpu"][0]
            print(f"{speedup:>7.1f}×", end="")
        print()

    print(f"\n{'=' * 80}")
    print("  Notes:")
    print("  - Speedup = CPU_time / GPU_time")
    print("  - Error is measured against PySCF's exact FCI solver")
    print("  - Chemical accuracy threshold: 1.6 mHa")
    if args.simulation == "mps":
        print(f"  - MPS bond dimension: {args.bond_dim}")
    print()


if __name__ == "__main__":
    main()
