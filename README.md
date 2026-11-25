# qoro-pyscf

PySCF tools built on [divi](https://github.com/QoroQuantum/divi) that divi does
not ship yet: ADAPT-VQE, Rotosolve, an Adam optimiser, active-space
selection, one-electron properties and Z₂ qubit tapering. A ground-state VQE as
PySCF's `fcisolver` is divi's `divi.qprog.algorithms.VQEFCISolver`, and QSCI is
divi's `divi.qprog.ensembles.qsci.QSCISolver`.

## Installation

The project depends on a local divi checkout next to it (`../divi`):

```bash
uv sync
```

## ADAPT-VQE

`run_adapt_vqe` grows a circuit from a pool of excitations on Maestro.
`build_pauli_terms` turns a divi `MolecularProblem` Hamiltonian into its
inputs; see [examples/11_adapt_vqe.py](examples/11_adapt_vqe.py).

## Adam

`AdamOptimizer` is a divi `Optimizer` (Adam on central finite-difference
gradients, one parameter set, no checkpointing), for example in a VQE:

```python
from qoro_pyscf import AdamOptimizer

VQEFCISolver(backend=backend, optimizer=AdamOptimizer(learning_rate=0.01))
```

## Other tools

| Function | Purpose |
|----------|---------|
| `suggest_active_space`, `suggest_active_space_from_mp2` | AVAS or MP2-natural-orbital active spaces |
| `compute_dipole_moment`, `compute_natural_orbitals`, `compute_mulliken_spin_population` | Properties from active-space RDMs |
| `taper_hamiltonian` | Z₂ tapering of an OpenFermion `QubitOperator` |
| `rotosolve_step`, `rotosolve_sweep` | Analytic single-parameter minimisation |
| `configure_backend`, `set_license_key` | Maestro settings for ADAPT-VQE |

## Examples

See [examples/](examples/).

## License

Apache 2.0 — see [LICENSE](LICENSE).
