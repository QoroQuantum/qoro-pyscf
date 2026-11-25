# qoro-pyscf — Library Overview

`qoro-pyscf` holds PySCF tools that divi does not ship yet. Circuits,
Hamiltonians, RDMs, ansätze and the ground-state VQE solver
(`divi.qprog.algorithms.VQEFCISolver`) come from divi; this package builds on
them.

## Modules

| Module | Responsibility |
|--------|---------------|
| `qsci_solver.py` | `QSCISolver`: PySCF `fcisolver` that samples a divi VQE and diagonalises in the selected determinants (PySCF selected CI). |
| `adapt.py` | ADAPT-VQE on Maestro circuits, and `build_pauli_terms` to feed it a divi Hamiltonian. |
| `rotosolve.py` | Rotosolve steps and sweeps, with per-parameter frequencies. |
| `adam.py` | `AdamOptimizer`, a divi `Optimizer`. |
| `active_space.py` | AVAS and MP2-natural-orbital active-space selection. |
| `properties.py` | Dipole moments, natural orbitals, Mulliken spin populations from active-space RDMs. |
| `tapering.py` | Z₂ qubit tapering of OpenFermion `QubitOperator`s. |
| `backends.py` | Maestro CPU/GPU and statevector/MPS settings for ADAPT-VQE, and the GPU licence key. |

## Qubit ordering

Spin orbitals are interleaved, alpha on even qubits and beta on odd ones, the
same as divi. ADAPT-VQE's Pauli labels put qubit ``k`` at character ``k``,
which is Qiskit's label order reversed; `build_pauli_terms` does the reversal.
