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
3. **We build the ansatz circuit** — hardware-efficient, UCCSD, UpCCD, ADAPT-VQE, or a **custom** user-injected circuit (e.g. QCC), as native Maestro `QuantumCircuit` objects
4. **We run VQE on Maestro's GPU** — SciPy optimiser + Maestro's `qc.estimate()` for expectation values
5. **We return `(energy, self)` to PySCF** — and reconstruct RDMs on demand from the optimised circuit

### Key Difference from Qiskit

In qiskit-nature-pyscf, the heavy lifting is done by `qiskit-nature` (ElectronicStructureProblem, mappers, ElectronicDensity). We replace all of that with our own lightweight implementation using OpenFermion + Maestro.

---

## Architecture

| Module | Responsibility |
|--------|---------------|
| **`maestro_solver.py`** | `MaestroSolver` — PySCF `fcisolver` drop-in. Orchestrates VQE/VQD loop, exposes RDM methods, custom Pauli evaluation, statevector extraction, and save/load checkpointing. |
| **`hamiltonian.py`** | Converts PySCF integrals to qubit Hamiltonian (Jordan-Wigner via OpenFermion). Handles both RHF and UHF integral formats. |
| **`ansatze.py`** | Builds parameterised quantum circuits: Hartree-Fock initial state, hardware-efficient (Ry/Rz + CNOT), UCCSD, and UpCCD (paired doubles for singlets). |
| **`adapt.py`** | ADAPT-VQE: adaptive circuit growing by operator gradient screening. Produces the most compact ansatz for a given accuracy target. |
| **`active_space.py`** | Active-space auto-selection: AVAS (from AO labels) and MP2 natural orbital analysis (automatic). Returns `(norb, nelec, mo_coeff)` ready for CASCI/CASSCF. |
| **`tapering.py`** | Z₂ symmetry-based qubit tapering. Exploits particle-number and spin parity to remove 2 qubits from the Hamiltonian. |
| **`expectation.py`** | Wraps Maestro's `qc.estimate()` for batched Pauli expectation values. |
| **`rdm.py`** | Reconstructs 1-RDM and 2-RDM from the VQE circuit by measuring JW-mapped fermionic operators. |
| **`properties.py`** | Dipole moments, natural orbital occupations, Mulliken spin populations. |
| **`backends.py`** | GPU/CPU detection, statevector/MPS configuration, license key management. |

### Data Flow

```
MaestroSolver.kernel()
  │
  ├─→ hamiltonian.integrals_to_qubit_hamiltonian()  →  QubitOperator
  ├─→ [optional] tapering.taper_hamiltonian()        →  reduced QubitOperator (−2 qubits)
  ├─→ hamiltonian.qubit_op_to_pauli_list()           →  Pauli labels + coeffs
  ├─→ ansatze / custom_ansatz(params)                →  QuantumCircuit
  ├─→ expectation.compute_energy(circuit, paulis)    →  float (via Maestro GPU)
  ├─→ [optional] spin penalty via fix_spin_()        →  ⟨S²⟩ penalty term
  ├─→ [optional] callback(iteration, energy, params) →  user logging
  ├─→ scipy.optimize.minimize(cost_fn)               →  optimal params
  ├─→ [optional] _run_vqd() for nroots > 1           →  excited state energies
  │
  └─→ Returns (energy, self) or (energies, [self]*nroots)

suggest_active_space(mf, ao_labels)     →  (norb, nelec, mo_coeff) via AVAS
suggest_active_space_from_mp2(mf)       →  (norb, nelec, mo_coeff) via MP2 NOs

MaestroSolver.make_rdm1()
  └─→ rdm.compute_1rdm_spatial(optimal_circuit)  →  (rdm1_a, rdm1_b)

MaestroSolver.evaluate_custom_paulis()
  └─→ Direct Pauli term evaluation on GPU (bypasses Hamiltonian generation)

MaestroSolver.get_final_statevector()
  └─→ Raw complex-valued amplitudes for fidelity benchmarking

MaestroSolver.save() / MaestroSolver.load()
  └─→ JSON + NPZ checkpoint for reproducibility / fault tolerance
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
| `spin_square(ci, norb, nelec)` | ⟨S²⟩ and multiplicity |
| `fix_spin_(shift, ss)` | Spin-penalty constraint for VQE |

The `ci` argument is actually `self` — we return it from `kernel()` as the "CI vector" and use it to look up the cached optimal circuit for RDM reconstruction. This is the same pattern used by qiskit-nature-pyscf.

### Advanced Methods

| Method | Purpose |
|--------|---------|
| `evaluate_custom_paulis(terms, circuit)` | Evaluate user-defined Pauli Hamiltonian on GPU |
| `get_final_statevector(circuit)` | Extract raw complex statevector for fidelity |
| `save(path)` / `load(path)` | Checkpoint solver state (JSON + NPZ) |

---

## Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ansatz` | str | `"hardware_efficient"` | Ansatz type: `hardware_efficient`, `uccsd`, `upccd`, `adapt`, `custom` |
| `ansatz_layers` | int | 2 | Layers for hardware-efficient ansatz |
| `optimizer` | str | `"COBYLA"` | SciPy optimiser method |
| `maxiter` | int | 200 | Maximum VQE iterations (0 = pre-computed mode) |
| `backend` | str | `"gpu"` | `"gpu"` or `"cpu"` |
| `simulation` | str | `"statevector"` | `"statevector"` or `"mps"` |
| `mps_bond_dim` | int | 64 | MPS bond dimension (only for `simulation="mps"`) |
| `taper` | bool | `False` | Enable Z₂ qubit tapering (saves ~2 qubits) |
| `vqd_penalty` | float | 5.0 | Overlap penalty strength β for VQD excited states |
| `nroots` | int | 1 | Number of roots (1 = ground state only, >1 = VQD) |
| `callback` | callable | None | `(iteration, energy, params) → None` hook |
| `custom_ansatz` | callable/circuit | None | User-injected ansatz for `ansatz="custom"` |
| `custom_ansatz_n_params` | int | None | Parameter count for callable custom ansatze |

---

## Dependencies

| Package | Why |
|---------|-----|
| `qoro-maestro` | GPU-accelerated circuit simulation |
| `pyscf` | Molecular integrals, CASCI/CASSCF framework |
| `openfermion` | Jordan-Wigner mapping, fermionic operators |
| `scipy` | Classical VQE optimisation (COBYLA, L-BFGS-B, etc.) |
| `numpy` | Array operations |
