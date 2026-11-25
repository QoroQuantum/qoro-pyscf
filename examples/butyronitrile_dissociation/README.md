# Butyronitrile C≡N Dissociation — VQE on Maestro

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

## Three Stages

| Stage | Active Space | Basis | Qubits | Ansatz | Backend | Purpose |
|-------|-------------|-------|--------|--------|---------|---------|
| **0 — CPU Quick** | CAS(4,4) | STO-3G | 8 | UCCSD (COBYLA) | CPU | Local demo, ~20 s |
| **1 — Baseline** | CAS(8,8) | STO-3G | 16 | ADAPT-VQE (greedy Rotosolve) | GPU recommended | Standard active space |
| **2 — Extended** | CAS(14,14) | 6-31G* | 28 | UCCSD (Adam) | GPU required | Larger space + better basis |

## Quick Start

```bash
# Stage 0 — fast CPU demo (8 qubits, ~20 s)
python run_cn_dissociation.py --stage 0 --cpu

# Stage 0 — 3-point quick test (~8 s)
python run_cn_dissociation.py --stage 0 --cpu --quick

# Stage 1 — baseline (GPU recommended)
python run_cn_dissociation.py --stage 1 --gpu

# Stage 1 — GPU vs CPU comparison
python run_cn_dissociation.py --stage 1 --both

# Stage 2 — extended active space (GPU required)
python run_cn_dissociation.py --stage 2 --gpu

# Stages 1 and 2 on GPU
python run_cn_dissociation.py --stage both --gpu
```

## Stage 0 Results

UCCSD-VQE error against exact CASCI in the same CAS(4,4), about 2 s per
point on CPU:

```
d(C≡N)   VQE error vs CASCI
──────   ──────────────────
1.157 Å   0.17 mHa
1.356 Å   1.35 mHa
1.555 Å   3.15 mHa
1.754 Å  10.44 mHa
1.953 Å  18.30 mHa
2.152 Å   7.43 mHa
2.351 Å   2.74 mHa
2.550 Å   0.90 mHa
2.749 Å   0.25 mHa
```

Stages 1 and 2 grow the active space to 16 and 28 qubits; their GPU and CPU
timings have not been measured on the current code. `--quick` writes to a
separate `*_quick.json`.

## Plotting Results

Results are saved as JSON in `results/`. Use the plotting script to
visualise dissociation curves, accuracy, and timing:

```bash
# Plot Stage 0 CPU demo
python plot_results.py results/butyronitrile_stage0_sto-3g_cas4.json

# Plot Stage 1 baseline
python plot_results.py results/butyronitrile_stage1_sto-3g_cas8.json

# Compare Stage 0 vs Stage 1 side-by-side
python plot_results.py results/butyronitrile_stage0_*.json results/butyronitrile_stage1_*.json

# Save all plots as PNG
python plot_results.py results/*.json --save --outdir figures/
```

The plotting script generates:
- **Dissociation PES** — HF, MP2, FCI, VQE curves overlaid
- **Accuracy plot** — VQE error vs FCI at each geometry (with 1.6 mHa chemical accuracy threshold)
- **Timing plot** — wall time per geometry point (with GPU speedup annotation when `--both` is used)
- **Cross-stage comparison** — overlaid accuracy curves when multiple stages are provided

## Output

Results are saved as JSON in `results/` for plotting. Each file contains:
- HF, MP2, and CAS-FCI reference energies
- VQE energies and wall times per geometry point
- Metadata (basis, active space, ansatz, etc.)

A pre-computed Stage 0 result (`results/butyronitrile_stage0_sto-3g_cas4.json`)
is included in the repository so the plot can be generated immediately without
running VQE.

## Requirements

- `qoro-pyscf` with its divi dependency (`uv sync`)
- `matplotlib` (the `notebook` extra)
- GPU licence key (for Stages 1–2 with `--gpu`)
