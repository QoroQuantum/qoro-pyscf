# qoro-maestro-pyscf

> PySCF integration plugin for the [Maestro](https://qoroquantum.github.io/maestro/) quantum simulator by [Qoro Quantum](https://qoroquantum.de).
>
> **Drop-in replacement for [qiskit-nature-pyscf](https://github.com/qiskit-community/qiskit-nature-pyscf)** — run quantum chemistry VQE calculations on Maestro's GPU-accelerated backends with zero Qiskit dependencies.

## Installation

```bash
pip install qoro-maestro-pyscf
```

## Quick Start — CASCI with VQE

```python
from pyscf import gto, scf, mcscf
from qoro_maestro_pyscf import MaestroSolver

mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
hf  = scf.RHF(mol).run()

cas = mcscf.CASCI(hf, 2, 2)
cas.fcisolver = MaestroSolver(ansatz="uccsd")
cas.run()
```

## GPU MPS Mode (Larger Active Spaces)

```python
cas.fcisolver = MaestroSolver(
    ansatz="hardware_efficient",
    ansatz_layers=3,
    simulation="mps",         # Matrix Product State on GPU
    mps_bond_dim=128,
)
```

## GPU Setup & Licensing

Maestro GPU simulation requires an NVIDIA GPU and a license key from [Qoro Quantum](https://qoroquantum.de). Three ways to provide your key:

**Option 1 — Pass directly to the solver:**
```python
cas.fcisolver = MaestroSolver(
    ansatz="uccsd",
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

> **Note:** First activation requires an internet connection (one-time). After that, the license is cached locally for offline use. No GPU? The solver automatically falls back to CPU.

## Migrating from Qiskit

| qiskit-nature-pyscf | qoro-maestro-pyscf |
|---|---|
| `from qiskit_nature_pyscf import QiskitSolver` | `from qoro_maestro_pyscf import MaestroSolver` |
| `cas.fcisolver = QiskitSolver(algorithm)` | `cas.fcisolver = MaestroSolver(ansatz="uccsd")` |
| Requires Qiskit, qiskit-nature, qiskit-algorithms | Zero Qiskit dependencies |
| CPU-only estimator | GPU-accelerated (CUDA) |
| Statevector only | Statevector + MPS |

## Features

- **GPU-accelerated** statevector & MPS simulation via Maestro's CUDA backend
- **Automatic GPU→CPU fallback** when no GPU is available
- **Drop-in PySCF solver** — implements the full `fcisolver` protocol (`kernel`, `make_rdm1`, `make_rdm1s`, `make_rdm12`, `make_rdm12s`)
- **CASCI and CASSCF** support (CASCI recommended; CASSCF works but VQE convergence can be tricky in the macro-iteration loop)
- **Multiple ansatze** — hardware-efficient and UCCSD
- **UHF support** — handles spin-unrestricted integrals

## Architecture

```
qoro_maestro_pyscf/
├── maestro_solver.py   # MaestroSolver — PySCF fcisolver drop-in
├── hamiltonian.py      # PySCF integrals → QubitOperator (Jordan-Wigner)
├── ansatze.py          # HF initial state, hardware-efficient, UCCSD
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

See the [examples/](examples/) directory for 8 worked examples and a full workflow notebook covering H₂ dissociation, LiH UCCSD, GPU benchmarking, MPS bond dimensions, CASSCF, NEVPT2, dipole moments, and geometry optimisation.

## License

GPL-3.0 (same as Maestro)
