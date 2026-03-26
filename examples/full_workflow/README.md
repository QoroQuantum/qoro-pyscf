# Full Quantum Chemistry Workflow with Qoro

A complete, publication-quality quantum chemistry pipeline running on Qoro's GPU-accelerated quantum simulator.

## What This Does

This example models **BeH₂** (beryllium dihydride) — a molecule with interesting multi-reference character — through the full computational chemistry stack:

```
┌──────────────────────────────────────────────────────────────────┐
│  Step 1: Define Molecule (PySCF)                                 │
│    BeH₂ in STO-3G basis → 7 AOs, 14 spin-orbitals               │
├──────────────────────────────────────────────────────────────────┤
│  Step 2: Hartree-Fock (PySCF, classical)                         │
│    Mean-field approximation → starting orbitals                  │
├──────────────────────────────────────────────────────────────────┤
│  Step 3: CASSCF on Qoro GPU  ◀━━ THIS IS THE QUANTUM PART   │
│    VQE in a (2e, 3o) active space → 6 qubits                    │
│    Orbital optimisation using VQE density matrices               │
├──────────────────────────────────────────────────────────────────┤
│  Step 4: NEVPT2 (PySCF, classical)                               │
│    Perturbation theory on top of CASSCF → dynamic correlation    │
├──────────────────────────────────────────────────────────────────┤
│  Step 5: Properties (classical post-processing)                  │
│    Dipole moment & natural orbital occupations from VQE RDMs     │
└──────────────────────────────────────────────────────────────────┘
```

## Why This Matters

This is **exactly** how computational chemists work every day — the same pipeline they'd run with a classical FCI solver or DMRG. The only difference is that the active-space solver is a VQE running on Qoro's GPU.

If you can show a chemist this example and they see familiar output (CASSCF energies, NEVPT2 corrections, natural orbital occupations, dipole moments), they'll immediately understand how Qoro fits into their workflow.

## Running

```bash
# CPU (no GPU required)
python full_workflow.py

# GPU accelerated
python full_workflow.py --gpu

# Interactive notebook
jupyter notebook full_workflow.ipynb
```

## Expected Output

```
  ┌─ Step 1: Molecule
  │  BeH₂ (STO-3G), linear geometry
  │  Atoms: 3, AOs: 7
  │
  ├─ Step 2: Hartree-Fock
  │  E(HF) = -15.XXXXXXXXXX Ha
  │
  ├─ Step 3: CASSCF on Qoro (CPU)
  │  Active space: (2e, 3o) → 6 qubits
  │  E(CASSCF) = -15.XXXXXXXXXX Ha
  │
  ├─ Step 4: NEVPT2 (classical)
  │  ΔE(NEVPT2) = -0.XXXXXXXXXX Ha
  │  E(CASSCF+NEVPT2) = -15.XXXXXXXXXX Ha
  │
  ├─ Step 5: Properties
  │  Dipole: [+0.0000, +0.0000, +0.0000] D
  │  |μ| = 0.0000 D  (should be ~0 by symmetry)
  │  Natural occupations: [1.9XXX, 0.0XXX, 0.0XXX]
  │
  └─ Summary
     Method                  Energy (Ha)     Δ FCI (mHa)
     HF                    -15.XXXXXXXX         XX.XX
     CASSCF (Qoro)      -15.XXXXXXXX          X.XX
     CASSCF + NEVPT2       -15.XXXXXXXX          X.XX
     FCI (exact)           -15.XXXXXXXX          0.00
```

## Key Takeaways

1. **CASSCF captures static correlation** — the orbital optimisation lowers the energy beyond CASCI
2. **NEVPT2 is free** — it uses the RDMs from the VQE, computed by Qoro
3. **Properties work** — dipole moments, natural occupations, all from the VQE wavefunction
4. **Same pipeline as classical** — a chemist can drop in `QoroSolver` without changing their workflow

## Files

| File | Description |
|------|-------------|
| [full_workflow.py](full_workflow.py) | Self-contained script, runnable from CLI |
| [full_workflow.ipynb](full_workflow.ipynb) | Interactive notebook with step-by-step walkthrough |
| [README.md](README.md) | This file |
