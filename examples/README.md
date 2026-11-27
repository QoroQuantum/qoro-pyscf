# Examples

Worked examples on top of divi. Ground-state VQE as a PySCF solver, CASSCF,
UpCCD, QCC entanglers, N₂ dissociation and the ethylene UV-vis spectrum, QSCI
on N₂ dissociation, and ADAPT-VQE on LiH, are covered by divi's own tutorials
(`divi/tutorials/chemistry/`). The butyronitrile GPU stages need a Maestro GPU
licence (`MAESTRO_LICENSE_KEY`).

```bash
uv sync
uv run python examples/01_geometry_opt.py
```

| Script | Topic | Qubits |
|--------|-------|:------:|
| [01_geometry_opt.py](01_geometry_opt.py) | H₂ bond length from VQE nuclear gradients | 4 |
| [butyronitrile_dissociation/](butyronitrile_dissociation/) | C≡N dissociation in three active-space stages | 8–28 |
