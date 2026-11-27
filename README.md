# qoro-pyscf

PySCF tools built on [divi](https://github.com/QoroQuantum/divi) that divi does
not ship yet: a VQD active-space solver and Z₂ qubit tapering. A ground-state
VQE as PySCF's `fcisolver` is divi's `divi.qprog.algorithms.VQEFCISolver`,
ADAPT-VQE is divi's `divi.qprog.algorithms.AdaptVQE`, and QSCI is divi's
`divi.qprog.ensembles.qsci.QSCISolver`.

## Installation

The project depends on a local divi checkout next to it (`../divi`):

```bash
uv sync
```

## Tools

| Function | Purpose |
|----------|---------|
| `VQDSolver` | Ground and excited states by variational quantum deflation, as PySCF's `fcisolver` |
| `taper_hamiltonian` | Z₂ tapering of an OpenFermion `QubitOperator` |

## Examples

See [examples/](examples/).

## License

Apache 2.0 — see [LICENSE](LICENSE).
