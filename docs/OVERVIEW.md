# qoro-maestro-pyscf — Library Overview

## What Is This?

`qoro-maestro-pyscf` is a PySCF integration plugin for the [Maestro](https://qoroquantum.github.io/maestro/) GPU quantum simulator by [Qoro Quantum](https://qoroquantum.de). It enables GPU-accelerated VQE calculations within PySCF's CASCI/CASSCF framework.

It lets you run quantum chemistry calculations (VQE) on Maestro's GPU-accelerated backends directly within PySCF's CASCI/CASSCF framework.

---

## How It Works

```
PySCF (molecule → integrals)
    ↓
qoro-maestro-pyscf (integrals → qubit Hamiltonian → VQE on Maestro → RDMs)
    ↓
PySCF (RDMs → orbital optimisation → energy)
```

### The Pipeline

1. **PySCF calls `kernel(h1, h2, norb, nelec)`** — passes one- and two-electron integrals from the active space
2. **We build the qubit Hamiltonian** — Jordan-Wigner transformation via OpenFermion
3. **We build the ansatz circuit** — hardware-efficient or UCCSD, as native Maestro `QuantumCircuit` objects
4. **We run VQE on Maestro's GPU** — SciPy optimiser + Maestro's `qc.estimate()` for expectation values
5. **We return `(energy, self)` to PySCF** — and reconstruct RDMs on demand from the optimised circuit

### Key Difference from Qiskit

In qiskit-nature-pyscf, the heavy lifting is done by `qiskit-nature` (ElectronicStructureProblem, mappers, ElectronicDensity). We replace all of that with our own lightweight implementation using OpenFermion + Maestro.

---

## Architecture

| Module | Responsibility |
|--------|---------------|
| **`maestro_solver.py`** | `MaestroSolver` — PySCF `fcisolver` drop-in. Orchestrates the VQE loop and exposes RDM methods. |
| **`hamiltonian.py`** | Converts PySCF integrals to qubit Hamiltonian (Jordan-Wigner via OpenFermion). Handles both RHF and UHF integral formats. |
| **`ansatze.py`** | Builds parameterised quantum circuits: Hartree-Fock initial state, hardware-efficient (Ry/Rz + CNOT), and UCCSD. |
| **`expectation.py`** | Wraps Maestro's `qc.estimate()` for batched Pauli expectation values. |
| **`rdm.py`** | Reconstructs 1-RDM and 2-RDM from the VQE circuit by measuring JW-mapped fermionic operators. |
| **`backends.py`** | GPU/CPU detection, statevector/MPS configuration, license key management. |

### Data Flow

```
MaestroSolver.kernel()
  │
  ├─→ hamiltonian.integrals_to_qubit_hamiltonian()  →  QubitOperator
  ├─→ hamiltonian.qubit_op_to_pauli_list()          →  Pauli labels + coeffs
  ├─→ ansatze.hardware_efficient_ansatz(params)      →  QuantumCircuit
  ├─→ expectation.compute_energy(circuit, paulis)    →  float (via Maestro GPU)
  ├─→ scipy.optimize.minimize(cost_fn)               →  optimal params
  │
  └─→ Returns (energy, self)

MaestroSolver.make_rdm1()
  └─→ rdm.compute_1rdm_spatial(optimal_circuit)  →  (rdm1_a, rdm1_b)
```

---

## Simulation Backends

| Backend | `simulation=` | Best For | Notes |
|---------|--------------|----------|-------|
| GPU Statevector | `"statevector"` | Small-medium systems (≤30 qubits) | Exact, fast on GPU |
| GPU MPS | `"mps"` | Larger systems with bounded entanglement | Tuneable bond dimension |
| CPU (fallback) | `"statevector"` | Development, no GPU available | Auto-fallback |

---

## PySCF Protocol

`MaestroSolver` implements the full PySCF `fcisolver` interface:

| Method | What PySCF Calls It For |
|--------|------------------------|
| `kernel(h1, h2, norb, nelec)` | Solve the CI problem → energy |
| `approx_kernel(...)` | Same as `kernel` (CASSCF compatibility) |
| `make_rdm1(ci, norb, nelec)` | Spin-traced 1-RDM |
| `make_rdm1s(ci, norb, nelec)` | Spin-resolved 1-RDMs (α, β) |
| `make_rdm12(ci, norb, nelec)` | Spin-traced 1-RDM + 2-RDM |
| `make_rdm12s(ci, norb, nelec)` | Spin-resolved 1-RDMs + 2-RDMs |

The `ci` argument is actually `self` — we return it from `kernel()` as the "CI vector" and use it to look up the cached optimal circuit for RDM reconstruction. This is the same pattern used by qiskit-nature-pyscf.

---

## Dependencies

| Package | Why |
|---------|-----|
| `qoro-maestro` | GPU-accelerated circuit simulation |
| `pyscf` | Molecular integrals, CASCI/CASSCF framework |
| `openfermion` | Jordan-Wigner mapping, fermionic operators |
| `scipy` | Classical VQE optimisation (COBYLA, L-BFGS-B, etc.) |
| `numpy` | Array operations |
