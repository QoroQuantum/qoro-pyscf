# qoro-pyscf — Library Overview

`qoro-pyscf` holds PySCF tools that divi does not ship yet. Circuits,
Hamiltonians, RDMs, ansätze, the ground-state VQE solver
(`divi.qprog.algorithms.VQEFCISolver`) and the QSCI solver
(`divi.qprog.ensembles.qsci.QSCISolver`) come from divi; this package builds on
them.

## Modules

| Module | Responsibility |
|--------|---------------|
| `adapt.py` | ADAPT-VQE on Maestro circuits, and `build_pauli_terms` to feed it a divi Hamiltonian. |
| `tapering.py` | Z₂ qubit tapering of OpenFermion `QubitOperator`s. |
| `backends.py` | Maestro CPU/GPU and statevector/MPS settings for ADAPT-VQE, and the GPU licence key. |

## Qubit ordering

Spin orbitals are interleaved, alpha on even qubits and beta on odd ones, the
same as divi. ADAPT-VQE's Pauli labels put qubit ``k`` at character ``k``,
which is Qiskit's label order reversed; `build_pauli_terms` does the reversal.
