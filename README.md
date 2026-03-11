# qoro-maestro-pyscf

> PySCF integration plugin for the [Maestro](https://qoroquantum.github.io/maestro/) quantum simulator by [Qoro Quantum](https://qoroquantum.de).
>
> Run quantum chemistry VQE calculations — works on CPU out of the box, upgrade to GPU for speed.

## Installation

```bash
pip install qoro-maestro-pyscf
```

> **Dependencies:** PySCF, OpenFermion, SciPy, NumPy, and the Maestro runtime. No compiler needed — pre-built wheels for Linux and macOS (x86_64, arm64). Fully functional on CPU-only machines; no GPU required to install or run.

## Quick Start — CASCI with VQE

No GPU needed. Install and run:

```python
from pyscf import gto, scf, mcscf
from qoro_maestro_pyscf import MaestroSolver

mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
hf = scf.RHF(mol).run()

# Exact reference (PySCF's built-in FCI)
cas_fci = mcscf.CASCI(hf, 2, 2)
cas_fci.verbose = 0
fci_e = cas_fci.kernel()[0]

# VQE with Maestro
cas = mcscf.CASCI(hf, 2, 2)
cas.fcisolver = MaestroSolver(ansatz="uccsd", maxiter=500)
vqe_e = cas.kernel()[0]

print(f"FCI energy:  {fci_e:.8f} Ha")
print(f"VQE energy:  {vqe_e:.8f} Ha")
print(f"Error:       {abs(vqe_e - fci_e) * 1000:.4f} mHa")
```

Runs on CPU by default — no license key required. You should see ~0.01 mHa error, well within chemical accuracy (1.6 mHa).

## Running Your Own Molecule

Swap in your molecule, basis set, and active space:

### Choosing Your Setup

| Question | Guidance |
|----------|----------|
| **Basis set?** | Start with `sto-3g` for testing. Use `cc-pVDZ` or `6-31G*` for production. |
| **Active space?** | `(n_electrons, n_orbitals)` — use chemical intuition or `suggest_active_space()` for automatic selection. Each spatial orbital = 2 qubits. |
| **Which ansatz?** | Up to (6,6): use `uccsd` — chemistry-motivated, converges fast. Beyond (6,6): use `hardware_efficient` with 3–4 layers. For the most compact circuit: try `adapt`. |
| **How long?** | (2,2) on CPU: seconds. (6,6) on CPU: minutes. (10,10)+: use MPS or GPU. |

### Automatic Active Space Selection

Not sure which orbitals to include? Let PySCF + MP2 natural orbitals decide:

```python
from pyscf import gto, scf, mcscf
from qoro_maestro_pyscf import MaestroSolver, suggest_active_space_from_mp2

mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
hf = scf.RHF(mol).run()

norb, nelec, mo_coeff = suggest_active_space_from_mp2(hf)
cas = mcscf.CASCI(hf, norb, nelec)
cas.mo_coeff = mo_coeff
cas.fcisolver = MaestroSolver(ansatz="uccsd", maxiter=500)
cas.run()
```

## Scaling Up

Hitting limits on larger active spaces? Two options:

### MPS Mode (still CPU, no license needed)

Switch to Matrix Product State simulation for larger systems without needing a GPU:

```python
cas.fcisolver = MaestroSolver(
    ansatz="hardware_efficient",
    ansatz_layers=3,
    simulation="mps",
    mps_bond_dim=128,
)
```

### GPU Acceleration (fastest)

For maximum performance, add GPU support. Get your key instantly at **[maestro.qoroquantum.net](https://maestro.qoroquantum.net)**, then:

```python
cas.fcisolver = MaestroSolver(
    ansatz="uccsd",
    backend="gpu",
    license_key="XXXX-XXXX-XXXX-XXXX",
)
```

GPU + MPS for the largest active spaces:

```python
cas.fcisolver = MaestroSolver(
    ansatz="hardware_efficient",
    ansatz_layers=3,
    backend="gpu",
    simulation="mps",
    mps_bond_dim=128,
)
```

## GPU Setup & Licensing

GPU simulation requires an NVIDIA GPU and a license key. **Get your key instantly at [maestro.qoroquantum.net](https://maestro.qoroquantum.net).**

Three ways to provide your key:

**Option 1 — Pass directly to the solver:**
```python
cas.fcisolver = MaestroSolver(
    ansatz="uccsd",
    backend="gpu",
    license_key="XXXX-XXXX-XXXX-XXXX",
)
```

**Option 2 — Set it once in your script:**
```python
from qoro_maestro_pyscf import set_license_key
set_license_key("XXXX-XXXX-XXXX-XXXX")
```

**Option 3 — Environment variable (recommended for production):**
```bash
export MAESTRO_LICENSE_KEY="XXXX-XXXX-XXXX-XXXX"
```

> **Note:** First activation requires an internet connection (one-time). After that, the license is cached locally for offline use.

## Migrating from Qiskit

| qiskit-nature-pyscf | qoro-maestro-pyscf |
|---|---|
| `from qiskit_nature_pyscf import QiskitSolver` | `from qoro_maestro_pyscf import MaestroSolver` |
| `cas.fcisolver = QiskitSolver(algorithm)` | `cas.fcisolver = MaestroSolver(ansatz="uccsd")` |
| Requires Qiskit, qiskit-nature, qiskit-algorithms | Zero Qiskit dependencies |
| CPU-only estimator | GPU-accelerated (CUDA) |
| Statevector only | Statevector + MPS |

## Features

- **Works on CPU out of the box** — no GPU or license needed to get started
- **GPU-accelerated** statevector & MPS simulation via Maestro's CUDA backend (with license)
- **Drop-in PySCF solver** — implements the full `fcisolver` protocol (`kernel`, `make_rdm1`, `make_rdm1s`, `make_rdm12`, `make_rdm12s`)
- **CASCI and CASSCF** support (CASCI recommended; CASSCF works but VQE convergence can be tricky in the macro-iteration loop)
- **Multiple ansatze** — hardware-efficient, UCCSD, UpCCD, ADAPT-VQE, and **custom** (inject any `QuantumCircuit` callable, e.g. QCC)
- **Custom Pauli evaluation** — `evaluate_custom_paulis()` bypasses Hamiltonian generation for measurement-reduction research
- **Raw state extraction** — `get_final_statevector()` for exact fidelity benchmarking against exact diagonalisation
- **UHF support** — handles spin-unrestricted integrals
- **Pre-computed amplitudes** — skip VQE with `maxiter=0` + `initial_point`
- **State fidelity** — compare circuit states via `compute_state_fidelity()`

## Architecture

```
qoro_maestro_pyscf/
├── maestro_solver.py   # MaestroSolver — PySCF fcisolver drop-in
├── hamiltonian.py      # PySCF integrals → QubitOperator (Jordan-Wigner)
├── ansatze.py          # HF initial state, hardware-efficient, UCCSD, UpCCD
├── adapt.py            # ADAPT-VQE adaptive circuit growing
├── expectation.py      # Maestro circuit evaluation wrapper
├── rdm.py              # RDM reconstruction from VQE circuit
├── properties.py       # Dipole moments, natural orbitals
└── backends.py         # GPU/CPU/MPS backend configuration
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `qoro-maestro` | Quantum circuit simulation (GPU/CPU) |
| `pyscf` | Molecular integrals & classical reference |
| `openfermion` | Jordan-Wigner mapping & RDM operators |
| `scipy` | Classical parameter optimisation |

## Examples

See the [examples/](examples/) directory for 13 worked examples and a full workflow notebook covering H₂ dissociation, LiH UCCSD, GPU benchmarking, MPS bond dimensions, CASSCF, NEVPT2, dipole moments, geometry optimisation, UpCCD paired doubles, GPU benchmarks, ADAPT-VQE, custom QCC ansatze, and iterative QCC with fidelity benchmarking.

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
