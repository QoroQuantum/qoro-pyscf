# API Reference

## `MaestroSolver`

The primary class — a PySCF `fcisolver` drop-in that runs VQE on Maestro.

```python
from qoro_maestro_pyscf import MaestroSolver

solver = MaestroSolver(
    ansatz="hardware_efficient",   # "hardware_efficient", "uccsd", "upccd", or "adapt"
    ansatz_layers=2,               # layers for hardware-efficient ansatz
    optimizer="COBYLA",            # any scipy.optimize method
    maxiter=200,                   # max VQE iterations (0 = use initial_point directly)
    backend="gpu",                 # "gpu" or "cpu"
    simulation="statevector",      # "statevector" or "mps"
    mps_bond_dim=64,               # MPS bond dimension (if simulation="mps")
    license_key=None,              # Maestro GPU license key (optional)
    initial_point=None,            # initial parameter vector (optional)
    verbose=True,                  # print VQE progress
    # ADAPT-VQE parameters (only used when ansatz="adapt"):
    adapt_threshold=1e-3,          # convergence: stop when max |gradient| below this
    adapt_max_ops=50,              # max operators to add
    adapt_pool="sd",               # operator pool: "sd" (singles+doubles) or "d" (doubles)
)
```

### Usage with PySCF

```python
from pyscf import gto, scf, mcscf
from qoro_maestro_pyscf import MaestroSolver

mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
hf  = scf.RHF(mol).run()

# CASCI
cas = mcscf.CASCI(hf, 2, 2)
cas.fcisolver = MaestroSolver(ansatz="uccsd")
cas.run()

# CASSCF (orbital optimisation using RDMs)
cas = mcscf.CASSCF(hf, 2, 2)
cas.fcisolver = MaestroSolver(ansatz="hardware_efficient", ansatz_layers=3)
cas.run()
```

### Attributes (after `kernel` runs)

| Attribute | Type | Description |
|-----------|------|-------------|
| `converged` | `bool` | Whether the optimiser converged |
| `vqe_time` | `float` | Wall-clock time for VQE (seconds) |
| `energy_history` | `list[float]` | Energy at each VQE iteration |
| `optimal_params` | `np.ndarray` | Optimised variational parameters |

---

## `QSCISolver`

Quantum-Selected Configuration Interaction — wraps a `MaestroSolver` for state preparation, then classically diagonalizes in the quantum-selected subspace. Reference: [arXiv:2302.11320](https://arxiv.org/abs/2302.11320).

```python
from qoro_maestro_pyscf import QSCISolver, MaestroSolver

solver = QSCISolver(
    inner_solver=MaestroSolver(ansatz="uccsd"),  # VQE state preparation
    n_samples=500,              # max determinants in CI subspace
    probability_threshold=1e-8, # min probability for configuration selection
    verbose=True,               # print QSCI progress
)
```

### Usage with PySCF

```python
from pyscf import gto, scf, mcscf
from qoro_maestro_pyscf import MaestroSolver, QSCISolver

mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
hf = scf.RHF(mol).run()

cas = mcscf.CASCI(hf, 3, 2)
inner = MaestroSolver(ansatz="uccsd", maxiter=100)
cas.fcisolver = QSCISolver(inner_solver=inner)
e_qsci = cas.kernel()[0]
```

### Attributes (after `kernel` runs)

| Attribute | Type | Description |
|-----------|------|-------------|
| `converged` | `bool` | Whether QSCI completed successfully |
| `vqe_energy` | `float` | Inner VQE energy (before QSCI improvement) |
| `qsci_energy` | `float` | Final QSCI energy |
| `qsci_time` | `float` | Wall-clock time for full QSCI pipeline (seconds) |
| `n_determinants` | `int` | Size of the CI subspace |

### RDM Interface

`QSCISolver` implements the full PySCF `fcisolver` RDM protocol: `make_rdm1`, `make_rdm1s`, `make_rdm12`, `make_rdm12s`, `spin_square`. All RDMs are computed classically from the selected-CI eigenvector.

---

## `configure_backend`

Create a backend configuration manually (used internally by `MaestroSolver`, but available for advanced use).

```python
from qoro_maestro_pyscf import configure_backend

# GPU statevector
cfg = configure_backend(use_gpu=True, simulation="statevector")

# GPU MPS with custom bond dimension
cfg = configure_backend(use_gpu=True, simulation="mps", mps_bond_dim=128)

# With license key
cfg = configure_backend(license_key="XXXX-XXXX-XXXX-XXXX")
```

---

## `set_license_key`

Set the Maestro GPU license key programmatically.

```python
from qoro_maestro_pyscf import set_license_key

set_license_key("XXXX-XXXX-XXXX-XXXX")
# Equivalent to: os.environ["MAESTRO_LICENSE_KEY"] = "XXXX-XXXX-XXXX-XXXX"
```

---

## `BackendConfig`

Dataclass holding backend configuration (returned by `configure_backend`).

| Field | Type | Description |
|-------|------|-------------|
| `simulator_type` | `maestro.SimulatorType` | GPU or QCSim (CPU) |
| `simulation_type` | `maestro.SimulationType` | Statevector or MPS |
| `label` | `str` | Human-readable label |
| `mps_bond_dim` | `int \| None` | MPS bond dimension |

---

## Internal Modules

These are not part of the public API but are documented for contributors.

### `hamiltonian`

```python
from qoro_maestro_pyscf.hamiltonian import (
    integrals_to_qubit_hamiltonian,  # (h1, h2, norb) → (QubitOperator, offset)
    qubit_op_to_pauli_list,          # QubitOperator → (id_coeff, labels, coeffs)
)
```

### `ansatze`

```python
from qoro_maestro_pyscf.ansatze import (
    hartree_fock_circuit,             # (n_qubits, nelec) → QuantumCircuit
    hardware_efficient_ansatz,        # (params, n_qubits, n_layers) → QuantumCircuit
    hardware_efficient_param_count,   # (n_qubits, n_layers) → int
    uccsd_ansatz,                     # (params, n_qubits, nelec) → QuantumCircuit
    uccsd_param_count,                # (n_qubits, nelec) → int
    upccd_ansatz,                     # (params, n_qubits, nelec) → QuantumCircuit
    upccd_param_count,                # (n_qubits, nelec) → int
)
```

### `expectation`

```python
from qoro_maestro_pyscf.expectation import (
    evaluate_expectation,    # (circuit, pauli_labels, config) → np.ndarray
    compute_energy,          # (circuit, offset, labels, coeffs, config) → float
    get_state_probabilities, # (circuit, config) → np.ndarray
    compute_state_fidelity,  # (circuit_a, circuit_b, config) → float
)
```

### `rdm`

```python
from qoro_maestro_pyscf.rdm import (
    compute_1rdm_spatial,   # (circuit, n_qubits, config) → (rdm1_a, rdm1_b)
    compute_2rdm_spatial,   # (circuit, n_qubits, config) → (rdm2_aa, rdm2_ab, rdm2_bb)
    trace_spin_rdm1,        # (rdm1_a, rdm1_b) → rdm1
    trace_spin_rdm2,        # (rdm2_aa, rdm2_ab, rdm2_bb) → rdm2
)
```
