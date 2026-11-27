# qoro-pyscf — Library Overview

`qoro-pyscf` holds PySCF tools that divi does not ship yet. Circuits,
Hamiltonians, RDMs, ansätze, the ground-state VQE solver
(`divi.qprog.algorithms.VQEFCISolver`), ADAPT-VQE
(`divi.qprog.algorithms.AdaptVQE`) and the QSCI solver
(`divi.qprog.ensembles.qsci.QSCISolver`) come from divi; this package builds on
them.

## Modules

| Module | Responsibility |
|--------|---------------|
| `vqd_solver.py` | Ground and excited states by variational quantum deflation, as PySCF's `fcisolver`. |
| `tapering.py` | Z₂ qubit tapering of OpenFermion `QubitOperator`s. |

## Qubit ordering

Spin orbitals are interleaved, alpha on even qubits and beta on odd ones, the
same as divi.
