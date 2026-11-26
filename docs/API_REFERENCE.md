# API Reference

## `run_adapt_vqe(n_qubits, nelec, identity_offset, pauli_labels, pauli_coeffs, config, pool="sd", gradient_threshold=1e-3, max_operators=50, optimizer="COBYLA", maxiter_per_step=100, verbose=False, greedy=False)`

ADAPT-VQE over the spin-conserving single and double excitations
(`pool="sd"`) or the doubles alone (`pool="d"`). Returns a dict with `energy`, `params`, `selected_ops`,
`converged` (True when the largest gradient fell below the threshold),
`n_operators`, `energy_history` and `circuit`. `optimizer="ROTOSOLVE"` sweeps
divi's `RotosolveOptimizer` until the energy changes by less than 1e-8.
`greedy=True` instead takes one Rotosolve sweep over the newest parameter only,
ignoring `optimizer` and `maxiter_per_step`, and never reselects an operator
already in the circuit.

The single and double excitations are exact Jordan-Wigner exponentials
`exp(θ (T - T†))`, built as commuting Pauli rotations; both are Givens
rotations (frequency 1, order 2).

## `build_pauli_terms(hamiltonian)`

Splits a Qiskit `SparsePauliOp` into `(identity_offset, pauli_labels,
pauli_coeffs)` for `run_adapt_vqe`, reversing each label to qubit-``k``-first.

## Tapering

`taper_hamiltonian(qubit_op, n_qubits, nelec, stabilizers=None)` →
`TaperingResult(tapered_op, original_n_qubits, tapered_n_qubits,
removed_positions, stabilizers)`; the default stabilizers are the alpha and
beta parities.

## Backends

`configure_backend(use_gpu=False, simulation="statevector", mps_bond_dim=64, license_key=None)` → `BackendConfig` for `run_adapt_vqe`; falls back to CPU when no GPU is available. `set_license_key(key)` sets `MAESTRO_LICENSE_KEY`.
