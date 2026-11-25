# API Reference

## `QSCISolver(inner_solver, n_samples=500, probability_threshold=1e-8)`

PySCF `fcisolver`. `inner_solver` is a divi `VQEFCISolver`; its trained VQE is
sampled with `sample_solution()`, bitstrings with the wrong electron count per
spin or at or below `probability_threshold` are dropped, the `n_samples` most
probable are kept, and PySCF's `selected_ci.kernel_fixed_space` diagonalises in
their alpha × beta strings.

After `kernel`: `vqe_energy`, `qsci_energy`, `n_determinants`, `qsci_time`,
and `converged` (the inner VQE's). `approx_kernel` returns the last result. RDM methods (`make_rdm1`, `make_rdm1s`, `make_rdm12`,
`make_rdm12s`, `spin_square`) use the selected-CI eigenvector.

## `AdamOptimizer(learning_rate=0.01, grad_shift=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8)`

A divi `Optimizer`. Each iteration evaluates the `2 n` central
finite-difference points in one batch, takes an Adam step and evaluates the
new point; the best point seen is returned. No checkpointing.

## `run_adapt_vqe(n_qubits, nelec, identity_offset, pauli_labels, pauli_coeffs, config, pool="sd", gradient_threshold=1e-3, max_operators=50, optimizer="COBYLA", maxiter_per_step=100, verbose=False, greedy=False)`

ADAPT-VQE over the spin-conserving single and double excitations
(`pool="sd"`) or the doubles alone (`pool="d"`). Returns a dict with `energy`, `params`, `selected_ops`,
`converged` (True when the largest gradient fell below the threshold),
`n_operators`, `energy_history` and `circuit`. `optimizer="ROTOSOLVE"` sweeps
Rotosolve with each operator's frequency. `greedy=True` instead takes one
Rotosolve step on the newest parameter only, ignoring `optimizer` and
`maxiter_per_step`, and never reselects an operator already in the circuit.

## `build_pauli_terms(hamiltonian)`

Splits a Qiskit `SparsePauliOp` into `(identity_offset, pauli_labels,
pauli_coeffs)` for `run_adapt_vqe`, reversing each label to qubit-``k``-first.

## `rotosolve_step(cost_fn, params, index, freq=1, order=1)` / `rotosolve_sweep(cost_fn, params, max_sweeps=50, tol=1e-8, verbose=False, callback=None, freqs=None, orders=None)`

Minimisation of `a + Σ_{k ≤ order} b_k cos(k freq θ) + c_k sin(k freq θ)`
from `2 order + 1` evaluations per parameter: in closed form for `order=1`,
numerically for higher orders. The sweep returns `(params, energy,
energy_history, converged)`.

ADAPT-VQE's single and double excitations are exact Jordan-Wigner
exponentials `exp(θ (T - T†))`, built as commuting Pauli rotations; both are
Givens rotations (frequency 1, order 2).

## Active space

- `suggest_active_space(mf, ao_labels, threshold=0.2, minao="minao", verbose=None)` → `(norb, (n_alpha, n_beta), mo_coeff)` via AVAS.
- `suggest_active_space_from_mp2(mf, threshold=0.02, max_orbitals=None)` → the same from MP2 natural occupations; every orbital is kept in `mo_coeff`, ordered core, active, virtual.

## Properties

- `compute_dipole_moment(mol, mo_coeff, rdm1, ncore)` → `(dipole_debye, magnitude)`.
- `compute_natural_orbitals(rdm1)` → occupations (descending) and orbitals.
- `compute_mulliken_spin_population(mol, mo_coeff, rdm1_a, rdm1_b, ncore)` → per-atom spin population.

## Tapering

`taper_hamiltonian(qubit_op, n_qubits, nelec, stabilizers=None)` →
`TaperingResult(tapered_op, original_n_qubits, tapered_n_qubits,
removed_positions, stabilizers)`; the default stabilizers are the alpha and
beta parities.

## Backends

`configure_backend(use_gpu=False, simulation="statevector", mps_bond_dim=64, license_key=None)` → `BackendConfig` for `run_adapt_vqe`; falls back to CPU when no GPU is available. `set_license_key(key)` sets `MAESTRO_LICENSE_KEY`.
