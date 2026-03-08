# Examples

Hands-on examples demonstrating `qoro-maestro-pyscf` for quantum chemistry on Maestro's GPU-accelerated simulator.

## Quick Start

```bash
pip install qoro-maestro-pyscf
cd examples/

# Run any example
python 01_h2_dissociation.py
python 06_nevpt2.py --gpu
python full_workflow/full_workflow.py --gpu
```

## Examples

### Getting Started

| # | Script | Topic | Qubits | Key Concepts |
|---|--------|-------|:------:|-------------|
| 1 | [01_h2_dissociation.py](01_h2_dissociation.py) | H₂ bond-breaking curve | 4 | PES scan, multi-reference character, VQE vs FCI |
| 2 | [02_lih_uccsd.py](02_lih_uccsd.py) | LiH with UCCSD ansatz | 4 | Chemistry-motivated ansatz, CASCI |
| 3 | [03_gpu_vs_cpu.py](03_gpu_vs_cpu.py) | GPU acceleration benchmark | 4 | Backend switching, timing, Maestro speedup |

### Simulation Backends

| # | Script | Topic | Qubits | Key Concepts |
|---|--------|-------|:------:|-------------|
| 4 | [04_mps_bond_dimension.py](04_mps_bond_dimension.py) 🟢 | MPS bond dimension trade-off | 12 | H₆ chain, statevector vs MPS, accuracy vs scale |

### Production Chemistry

| # | Script | Topic | Qubits | Key Concepts |
|---|--------|-------|:------:|-------------|
| 5 | [05_casscf_orbital_opt.py](05_casscf_orbital_opt.py) 🟢 | CASSCF | 6 | Orbital optimisation, RDMs, CASCI vs CASSCF |
| 6 | [06_nevpt2.py](06_nevpt2.py) 🟢 | Perturbation theory | 8 | Dynamic correlation, NEVPT2 on top of CASSCF |
| 7 | [07_dipole_moment.py](07_dipole_moment.py) | Molecular properties | 4 | Dipole moments, natural orbital occupations |
| 8 | [08_geometry_opt.py](08_geometry_opt.py) | Geometry optimisation | 4 | Equilibrium bond length, PES scan |

### Ansatz Comparison

| # | Script | Topic | Qubits | Key Concepts |
|---|--------|-------|:------:|-------------|
| 9 | [09_upccd_paired_doubles.py](09_upccd_paired_doubles.py) | UpCCD vs UCCSD | 8 | Seniority-zero, paired doubles, parameter efficiency |
| 11 | [11_adapt_vqe.py](11_adapt_vqe.py) | ADAPT-VQE | 8 | Adaptive circuit growing, operator pool, compact ansatze |

### GPU Benchmarking

| # | Script | Topic | Qubits | Key Concepts |
|---|--------|-------|:------:|-------------|
| 10 | [10_gpu_benchmark.py](10_gpu_benchmark.py) | GPU vs CPU scaling | 8–20 | H-chain scaling, UpCCD, GPU speedup measurement |

### Full Pipeline

| # | Script | Topic | Qubits | Key Concepts |
|---|--------|-------|:------:|-------------|
| 10 | [full_workflow/](full_workflow/) 🟢 | Complete pipeline | 6 | BeH₂: HF → CASSCF → NEVPT2 → properties |

### 🟢 MPS-Recommended Examples

Examples marked with 🟢 benefit from Maestro's **MPS GPU mode** (`simulation="mps"`). MPS shines when:

- The active space has **6+ qubits** (3+ spatial orbitals)
- You need to scale beyond what statevector can handle (~30 qubits)
- You want to trade a small accuracy loss for much larger systems

To enable MPS on any example, change the `MaestroSolver` config:

```python
cas.fcisolver = MaestroSolver(
    simulation="mps",
    mps_bond_dim=64,   # higher = more accurate, slower
    backend="gpu",
)
```

> **Note:** Examples 1–3, 7, and 8 use only 4 qubits — statevector is faster for these. MPS overhead isn't worth it below ~6 qubits.

## GPU vs CPU

All examples support `--gpu` and `--cpu` flags:

```bash
python 06_nevpt2.py --gpu    # NVIDIA GPU (requires license)
python 06_nevpt2.py --cpu    # CPU fallback (no GPU needed)
```

## License Key

For GPU examples, set your Maestro license key:

```bash
export MAESTRO_LICENSE_KEY="XXXX-XXXX-XXXX-XXXX"
```
