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
Example 3 — GPU vs CPU Benchmark
=================================

Runs the same VQE calculation on both GPU and CPU backends and compares
wall-clock performance.

Why GPU matters for VQE
-----------------------
A single VQE iteration requires evaluating ⟨ψ(θ)|H|ψ(θ)⟩, which involves:

  1. Preparing |ψ(θ)⟩ — applying the parameterised circuit
  2. Computing expectation values for each Pauli term in the Hamiltonian

For a molecule with N qubits, the statevector has 2^N amplitudes, and each
gate is a 2^N × 2^N matrix multiply (or sparse equivalent). On a GPU, these
are massively parallelised — thousands of amplitudes updated simultaneously.

A typical VQE runs 100–500 iterations, each with dozens of Pauli evaluations.
The GPU speedup compounds across all of these.

What Qoro offers
-------------------
- **CPU by default** — works out of the box, no license needed
- **GPU upgrade** — switch to GPU for dramatic speedups on larger systems
- **Same API** — switching from CPU to GPU is a single argument change
- **No code changes** — your circuit, ansatz, and optimiser are identical

What this example shows
-----------------------
- Identical VQE on H₂ with both CPU and GPU backends
- Side-by-side timing comparison
- Both yield the same energy (within optimiser tolerance)

Usage
-----
    python 03_gpu_vs_cpu.py
"""

import time

from pyscf import gto, scf, mcscf
from qoro_pyscf import QoroSolver


def main():
    print("=" * 72)
    print("  GPU vs CPU BENCHMARK")
    print("=" * 72)

    mol = gto.M(atom="H 0 0 0; H 0 0 0.7414", basis="sto-3g", verbose=0)
    hf_obj = scf.RHF(mol).run()

    # FCI baseline
    cas_fci = mcscf.CASCI(hf_obj, 2, 2)
    cas_fci.verbose = 0
    fci_e = cas_fci.kernel()[0]
    print(f"\n  FCI energy (exact): {fci_e:+.10f} Ha\n")

    results = {}

    for backend_name in ["cpu", "gpu"]:
        print(f"  Running on {backend_name.upper()}...")

        cas = mcscf.CASCI(hf_obj, 2, 2)
        cas.fcisolver = QoroSolver(
            ansatz="hardware_efficient",
            ansatz_layers=2,
            backend=backend_name,
            maxiter=100,
            verbose=False,
        )

        t0 = time.perf_counter()
        e = cas.kernel()[0]
        elapsed = time.perf_counter() - t0

        error = abs(e - fci_e) * 1000
        iters = len(cas.fcisolver.energy_history)

        results[backend_name] = {"energy": e, "time": elapsed, "iters": iters}

        print(f"    Energy  : {e:+.10f} Ha  (err: {error:.2f} mHa)")
        print(f"    Time    : {elapsed:.3f} s")
        print(f"    Iters   : {iters}")
        print()

    # Comparison
    if "gpu" in results and "cpu" in results:
        speedup = results["cpu"]["time"] / max(results["gpu"]["time"], 1e-9)
        print(f"  {'─' * 50}")
        print(f"  GPU speedup   : {speedup:.1f}×")
        print(f"  Energy match  : {'✓ YES' if abs(results['gpu']['energy'] - results['cpu']['energy']) < 1e-4 else '✗ NO'}")
        print()
        print("  Note: Speedup increases with circuit size. For 4-qubit H₂,")
        print("  GPU overhead may dominate. Try larger molecules for dramatic gains.")


if __name__ == "__main__":
    main()
