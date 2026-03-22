# Butyronitrile C≡N Dissociation — VQE on Maestro GPU

> **Application:** Li-ion battery electrolyte chemistry
> **Molecule:** Butyronitrile (CH₃CH₂CH₂C≡N), 12 atoms
> **Bond:** C≡N triple bond dissociation

Computes the potential energy surface for the nitrile (C≡N) group
dissociation in butyronitrile — a co-solvent in lithium-ion battery
electrolytes. Understanding this reaction is critical for predicting
electrolyte stability and degradation pathways.

The dissociation breaks a triple bond, creating strong multireference
character that mean-field methods (HF, DFT) cannot capture — exactly
the regime where quantum algorithms like VQE excel.

## Two Stages

| Stage | Active Space | Basis | Qubits | Purpose |
|-------|-------------|-------|--------|---------|
| **1 — Baseline** | CAS(8,8) | STO-3G | 16 | Standard active space for this system |
| **2 — Extended** | CAS(14,14) | 6-31G* | 28 | Larger space + better basis for production |

## Quick Start

```bash
# Stage 1 — baseline (runs on CPU, no license needed)
python run_cn_dissociation.py --stage 1 --cpu

# Stage 1 — GPU vs CPU comparison
python run_cn_dissociation.py --stage 1 --both

# Stage 2 — extended active space (GPU recommended)
python run_cn_dissociation.py --stage 2 --gpu

# Both stages, full demo
python run_cn_dissociation.py --stage both --gpu

# Quick test (3 points only)
python run_cn_dissociation.py --stage 1 --cpu --quick
```

## What This Demonstrates

**Stage 1** — a standard CAS(8,8)/STO-3G calculation at 16 qubits,
showing Maestro's PySCF integration on a real battery-chemistry problem.
No GPU or license needed.

**Stage 2** — pushes to CAS(14,14)/6-31G* = 28 qubits with GPU
acceleration. The larger active space and polarised basis capture
significantly more electron correlation, producing a more accurate
dissociation curve than minimal-basis demonstrations.

## Output

Results are saved as JSON in `results/` for plotting. Each file contains:
- HF, MP2, and CAS-FCI reference energies
- VQE energies and wall times per geometry point
- Metadata (basis, active space, ansatz, etc.)
