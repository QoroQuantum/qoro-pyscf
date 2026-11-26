# qoro-pyscf

PySCF tools built on [divi](https://github.com/QoroQuantum/divi) that divi does
not ship yet: ADAPT-VQE and Z₂ qubit tapering. A ground-state VQE as
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

## Other tools

| Function | Purpose |
|----------|---------|
| `taper_hamiltonian` | Z₂ tapering of an OpenFermion `QubitOperator` |
| `configure_backend`, `set_license_key` | Maestro settings for ADAPT-VQE |

## Examples

See [examples/](examples/).

## License

Apache 2.0 — see [LICENSE](LICENSE).
