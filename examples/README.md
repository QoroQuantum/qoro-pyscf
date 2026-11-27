# Examples

Worked examples for `qoro-pyscf` on top of divi. They run on CPU; the GPU ones
need a Maestro GPU licence (`MAESTRO_LICENSE_KEY`). Ground-state VQE as a PySCF
solver, CASSCF, UpCCD, QCC entanglers, N₂ dissociation and the ethylene UV-vis
spectrum, QSCI on N₂ dissociation, and ADAPT-VQE on LiH, are covered by divi's
own tutorials (`divi/tutorials/chemistry/`).

```bash
uv sync
uv run python examples/03_gpu_vs_cpu.py
```

| Script | Topic | Qubits |
|--------|-------|:------:|
| [03_gpu_vs_cpu.py](03_gpu_vs_cpu.py) | The same VQE on Maestro's CPU and GPU simulators | 4 |
| [04_mps_bond_dimension.py](04_mps_bond_dimension.py) | MPS truncation error against bond dimension, H₆ | 12 |
| [08_geometry_opt.py](08_geometry_opt.py) | H₂ bond length from VQE nuclear gradients | 4 |
| [10_gpu_benchmark.py](10_gpu_benchmark.py) | CPU against GPU on H₄–H₁₀ chains with UpCCD | 8–20 |
| [butyronitrile_dissociation/](butyronitrile_dissociation/) | C≡N dissociation in three active-space stages | 8–28 |

Not yet ported to divi (they still import the removed `QoroSolver`):
[06_nevpt2.py](06_nevpt2.py), [13_iterative_qcc_fidelity.py](13_iterative_qcc_fidelity.py)
and [full_workflow/](full_workflow/).
