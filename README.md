# qoro-pyscf

PySCF tools built on [divi](https://github.com/QoroQuantum/divi) that divi does
not ship yet: a QSCI active-space solver, ADAPT-VQE, Rotosolve, an Adam
optimiser, active-space selection, one-electron properties and Z₂ qubit
tapering. A ground-state VQE as PySCF's `fcisolver` is divi's
`divi.qprog.algorithms.VQEFCISolver`.

## Installation

The project depends on a local divi checkout next to it (`../divi`):

```bash
uv sync
```

## QSCI

`QSCISolver` trains a VQE with divi's `VQEFCISolver`, samples it, keeps the
most probable determinants and diagonalises the Hamiltonian among them with
PySCF's selected CI.

```python
from pyscf import gto, mcscf, scf

from divi.backends import MaestroSimulator
from divi.qprog.algorithms import UCCSDAnsatz, VQEFCISolver
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from qoro_pyscf import QSCISolver

mean_field = scf.RHF(gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g")).run()
casci = mcscf.CASCI(mean_field, 3, 2)
casci.fcisolver = QSCISolver(
    VQEFCISolver(
        backend=MaestroSimulator(shots=1000),
        optimizer=ScipyOptimizer(ScipyMethod.L_BFGS_B),
        ansatz=UCCSDAnsatz(),
    ),
    n_samples=500,
)
casci.kernel()
```

Reference: Kanno et al., [arXiv:2302.11320](https://arxiv.org/abs/2302.11320).

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
