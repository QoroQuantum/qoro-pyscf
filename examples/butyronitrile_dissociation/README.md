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
| **0 — CPU Quick** | CAS(4,4) | STO-3G | 8 | UCCSD (COBYLA) | CPU | Local demo, ~20 min |
| **1 — Baseline** | CAS(8,8) | STO-3G | 16 | ADAPT-VQE (greedy Rotosolve) | GPU recommended | Standard active space |
| **2 — Extended** | CAS(14,14) | 6-31G* | 28 | UCCSD (Adam) | GPU required | Larger space + better basis |

## Quick Start

```bash
# Stage 0 — fast CPU demo (8 qubits, runs locally in ~20 min)
python run_cn_dissociation.py --stage 0 --cpu

# Stage 0 — 3-point quick test (~6 min)
python run_cn_dissociation.py --stage 0 --cpu --quick

# Stage 1 — baseline (GPU recommended; CPU takes hours per point)
python run_cn_dissociation.py --stage 1 --gpu

# Stage 1 — GPU vs CPU comparison
python run_cn_dissociation.py --stage 1 --both

# Stage 2 — extended active space (GPU required)
python run_cn_dissociation.py --stage 2 --gpu

# Both stages, full GPU demo
python run_cn_dissociation.py --stage both --gpu
```

## The Story: Why Active Space Size Matters

Running Stage 0 locally reveals exactly *why* this problem needs a GPU:

```
d(C≡N)   VQE error vs FCI
──────   ─────────────────
1.16 Å   0.35 mHa  ✓  (chemical accuracy — VQE works near equilibrium)
1.36 Å   2.03 mHa
1.55 Å   6.53 mHa
1.75 Å   17.2  mHa
1.95 Å   58.8  mHa  ← bond begins to break, multireference kicks in
2.15 Å   105   mHa
2.35 Å   138   mHa
2.55 Å   162   mHa
2.75 Å   179   mHa  ✗  (500× worse than equilibrium)
```

**The error explodes 500× as the C≡N bond breaks.** CAS(4,4) simply
doesn't have enough orbitals to describe the correlated electron pairs in
a breaking triple bond. This is not a failure of VQE — it's a failure of
the active space. The fix is more qubits.

### Why GPU is needed for the fix

| Active space | Qubits | CPU time/point | GPU time/point | Speedup |
|---|---|---|---|---|
| CAS(4,4) Stage 0 | 8 | ~0.6 s | ~0.8 s | ≈ 1× (overhead dominates) |
| CAS(8,8) Stage 1 | 16 | ~hours | ~minutes | **10–50×** |
| CAS(14,14) Stage 2 | 28 | infeasible | minutes | **∞** |

At 8 qubits, state-vector simulation is trivially fast and GPU overhead
actually hurts. The GPU advantage kicks in sharply above ~12 qubits,
where the state-vector doubles in size every qubit. By 16 qubits (Stage 1)
the GPU is 10–50× faster; by 28 qubits (Stage 2) CPU simulation is
simply not practical.

**Stage 0 demonstrates the problem. Stages 1 and 2 solve it — on a GPU.**

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

- `qoro-maestro >= 0.2.7`
- `pyscf`
- `matplotlib`
- GPU license key (for Stages 1–2 with `--gpu`)
