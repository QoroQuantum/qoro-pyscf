# Copyright 2026 Qoro Quantum Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
QoroSolver — PySCF FCI-solver drop-in backed by Maestro VQE.

This is the primary user-facing class of ``qoro-pyscf``.  It mirrors
the API of ``qiskit_nature_pyscf.QiskitSolver``, enabling a seamless swap:

    # Before (Qiskit):
    cas.fcisolver = QiskitSolver(algorithm)

    # After (Maestro):
    cas.fcisolver = QoroSolver(ansatz="uccsd")

The solver integrates with PySCF's CASCI and CASSCF objects by implementing
the ``fcisolver`` protocol: ``kernel``, ``make_rdm1``, ``make_rdm1s``,
``make_rdm12``, ``make_rdm12s``, and ``approx_kernel``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np
from scipy.optimize import minimize

if TYPE_CHECKING:
    from maestro.circuits import QuantumCircuit

from qoro_pyscf.backends import BackendConfig, configure_backend
from qoro_pyscf.hamiltonian import (
    integrals_to_qubit_hamiltonian,
    qubit_op_to_pauli_list,
)
from qoro_pyscf.ansatze import (
    hardware_efficient_ansatz,
    hardware_efficient_param_count,
    uccsd_ansatz,
    uccsd_param_count,
    upccd_ansatz,
    upccd_param_count,
)
from qoro_pyscf.expectation import compute_energy
from qoro_pyscf.rdm import (
    compute_1rdm_spatial,
    compute_2rdm_spatial,
    trace_spin_rdm1,
    trace_spin_rdm2,
)


logger = logging.getLogger(__name__)


@dataclass
class QoroSolver:
    """
    PySCF FCI-solver interface that runs a VQE on the Maestro simulator.

    Replaces PySCF's classical FCI solver with a quantum VQE algorithm
    executed on Maestro's CPU or GPU-accelerated backend.

    Parameters
    ----------
    ansatz : str
        Ansatz type: ``"hardware_efficient"``, ``"uccsd"``, ``"upccd"``,
        ``"adapt"``, or ``"custom"``.
    custom_ansatz : callable or QuantumCircuit or None
        Required when ``ansatz="custom"``. Either a callable with signature
        ``(params: np.ndarray, n_qubits: int, nelec: tuple[int,int]) → QuantumCircuit``
        that is invoked every VQE iteration (e.g. for iterative QCC circuits),
        or a pre-built ``QuantumCircuit`` for zero-parameter evaluation.
    custom_ansatz_n_params : int or None
        Number of variational parameters for the custom ansatz. Required
        when ``custom_ansatz`` is a callable.
    ansatz_layers : int
        Number of layers (hardware-efficient ansatz only). Default: 2.
    optimizer : str
        Optimiser method. Default: ``"COBYLA"``.

        - **Derivative-free** (SciPy): ``"COBYLA"``, ``"Nelder-Mead"``,
          ``"Powell"``
        - **Gradient-based** (SciPy): ``"L-BFGS-B"``, ``"CG"``, ``"BFGS"``
        - **Adam** (built-in): ``"adam"`` — uses the parameter-shift rule
          to compute exact quantum gradients. Controlled by
          ``learning_rate`` and ``grad_shift``.
    maxiter : int
        Maximum optimiser iterations. Default: 200.
    learning_rate : float
        Step size for the Adam optimiser. Ignored for SciPy methods.
        Default: 0.01.
    grad_shift : float
        Shift value for the parameter-shift gradient rule.
        Default: π/2 (exact for single-qubit rotation gates Rx, Ry, Rz).
        For non-standard generators, set to a smaller value.
    taper : bool
        If True, apply Z₂ qubit tapering to reduce qubit count by ~2.
        Default: False.
    vqd_penalty : float
        Overlap penalty strength β for VQD excited states. Only used
        when ``nroots > 1``. Default: 5.0.
    callback : callable or None
        If provided, called at each VQE iteration with signature
        ``(iteration: int, energy: float, params: np.ndarray) → None``.
    backend : str
        Backend selection: ``"cpu"`` or ``"gpu"``. Default: ``"cpu"``.
        CPU works out of the box with no license. Switch to ``"gpu"``
        for GPU acceleration (requires a Maestro license key).
    simulation : str
        Simulation mode: ``"statevector"`` or ``"mps"``. Default: ``"statevector"``.
    mps_bond_dim : int
        MPS bond dimension (for ``simulation="mps"``). Default: 64.
    license_key : str or None
        Maestro GPU license key (e.g. ``"XXXX-XXXX-XXXX-XXXX"``).
        If provided, sets the ``MAESTRO_LICENSE_KEY`` env var before GPU
        init. Can also be set via the env var directly or
        :func:`qoro_pyscf.backends.set_license_key`.
    initial_point : np.ndarray or None
        Initial parameter vector. If None, uses small random values.
    verbose : bool
        Print progress during optimisation. Default: True.

    Examples
    --------
    CASCI with CPU (works out of the box, no license needed):

    >>> from pyscf import gto, scf, mcscf
    >>> from qoro_pyscf import QoroSolver
    >>> mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    >>> hf = scf.RHF(mol).run()
    >>> cas = mcscf.CASCI(hf, 2, 2)
    >>> cas.fcisolver = QoroSolver(ansatz="uccsd")
    >>> cas.run()

    Upgrade to GPU for faster simulation:

    >>> cas.fcisolver = QoroSolver(
    ...     ansatz="uccsd",
    ...     backend="gpu",
    ...     license_key="XXXX-XXXX-XXXX-XXXX",  # or set MAESTRO_LICENSE_KEY env var
    ... )

    GPU MPS for larger active spaces:

    >>> cas.fcisolver = QoroSolver(
    ...     ansatz="hardware_efficient",
    ...     ansatz_layers=3,
    ...     backend="gpu",
    ...     simulation="mps",
    ...     mps_bond_dim=128,
    ... )

    CASCI with Adam optimiser (gradient-based):

    >>> cas.fcisolver = QoroSolver(
    ...     ansatz="uccsd",
    ...     optimizer="adam",
    ...     learning_rate=0.01,
    ...     maxiter=300,
    ... )
    """
    # --- User-configurable ---
    ansatz: str = "hardware_efficient"
    ansatz_layers: int = 2
    optimizer: str = "COBYLA"
    maxiter: int = 200
    learning_rate: float = 0.01
    grad_shift: float = 1e-3
    backend: str = "cpu"
    simulation: str = "statevector"
    mps_bond_dim: int = 64
    license_key: Optional[str] = None
    initial_point: Optional[np.ndarray] = None
    verbose: bool = True
    callback: Optional[Callable[[int, float, np.ndarray], None]] = None

    # --- Custom ansatz (QCC / user-defined) ---
    custom_ansatz: Optional[
        Union[Callable[[np.ndarray, int, tuple[int, int]], "QuantumCircuit"], "QuantumCircuit"]
    ] = None
    custom_ansatz_n_params: Optional[int] = None

    # --- ADAPT-VQE parameters ---
    adapt_threshold: float = 1e-3
    adapt_max_ops: int = 50
    adapt_pool: str = "sd"
    adapt_greedy: bool = False

    # --- Tapering ---
    taper: bool = False

    # --- Excited states (VQD) ---
    vqd_penalty: float = 5.0

    # --- PySCF interface attributes (set by CASCI/CASSCF) ---
    mol: object = field(default=None, repr=False)
    nroots: int = 1

    # --- Internal state (populated after kernel runs) ---
    converged: bool = field(default=False, init=False, repr=False)
    vqe_time: float = field(default=0.0, init=False, repr=False)
    energy_history: list = field(default_factory=list, init=False, repr=False)
    optimal_params: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _config: Optional[BackendConfig] = field(default=None, init=False, repr=False)
    _optimal_circuit: Optional[QuantumCircuit] = field(
        default=None, init=False, repr=False
    )
    _n_qubits: int = field(default=0, init=False, repr=False)
    _nelec: tuple = field(default=(0, 0), init=False, repr=False)
    _rdm1s_cache: Optional[tuple] = field(default=None, init=False, repr=False)
    _rdm2s_cache: Optional[tuple] = field(default=None, init=False, repr=False)
    _spin_penalty_shift: float = field(default=0.0, init=False, repr=False)
    _spin_penalty_ss: float = field(default=0.0, init=False, repr=False)
    _taper_result: object = field(default=None, init=False, repr=False)
    _vqd_energies: list = field(default_factory=list, init=False, repr=False)
    _vqd_circuits: list = field(default_factory=list, init=False, repr=False)

    def kernel(
        self,
        h1: Union[np.ndarray, tuple[np.ndarray, np.ndarray]],
        h2: Union[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]],
        norb: int,
        nelec: Union[int, tuple[int, int]],
        ci0=None,
        ecore: float = 0,
        **kwargs,
    ) -> tuple[float, "QoroSolver"]:
        """
        Find the ground-state energy via VQE.

        Implements PySCF's ``fcisolver.kernel`` protocol.

        Parameters
        ----------
        h1 : ndarray or (ndarray, ndarray)
            One-electron integrals. Tuple for UHF (alpha, beta).
        h2 : ndarray or (ndarray, ndarray, ndarray)
            Two-electron integrals (chemist notation). Tuple for UHF.
        norb : int
            Number of active spatial orbitals.
        nelec : int or (int, int)
            Number of active electrons.
        ci0 : ignored
            Placeholder for PySCF compatibility.
        ecore : float
            Core (inactive) energy. Default: 0.

        Returns
        -------
        e_tot : float
            Total energy (VQE energy + ecore).
        self : QoroSolver
            Reference to this solver (used as fake CI vector by PySCF).
        """
        # --- Resolve electron counts ---
        if isinstance(nelec, int):
            n_beta = nelec // 2
            n_alpha = nelec - n_beta
            self._nelec = (n_alpha, n_beta)
        else:
            self._nelec = nelec

        n_qubits = 2 * norb
        self._n_qubits = n_qubits

        # --- Validate custom ansatz early (before importing Maestro) ---
        if self.ansatz == "custom":
            if self.custom_ansatz is None:
                raise ValueError(
                    "ansatz='custom' requires `custom_ansatz` to be set "
                    "(a callable or QuantumCircuit)."
                )
            if callable(self.custom_ansatz) and self.custom_ansatz_n_params is None:
                raise ValueError(
                    "When `custom_ansatz` is a callable, "
                    "`custom_ansatz_n_params` must be set."
                )

        # --- Configure Maestro backend ---
        self._config = configure_backend(
            use_gpu=(self.backend == "gpu"),
            simulation=self.simulation,
            mps_bond_dim=self.mps_bond_dim,
            license_key=self.license_key,
        )

        if self.verbose:
            logger.info(
                "Maestro VQE: Backend=%s, Qubits=%d, Ansatz=%s",
                self._config.label, n_qubits, self.ansatz,
            )
            print(f"\nCASCI VQE solver (Maestro)")
            print(f"  Active space : ({sum(self._nelec)}e, {norb}o) → {n_qubits} qubits")
            print(f"  Ansatz       : {self.ansatz}")
            print(f"  Backend      : {self._config.label}")

        # --- Build qubit Hamiltonian from PySCF integrals ---
        qubit_op, _ = integrals_to_qubit_hamiltonian(h1, h2, norb)

        # --- Optional Z₂ tapering ---
        if self.taper:
            from qoro_pyscf.tapering import taper_hamiltonian
            self._taper_result = taper_hamiltonian(
                qubit_op, n_qubits, self._nelec
            )
            qubit_op = self._taper_result.tapered_op
            n_qubits = self._taper_result.tapered_n_qubits
            self._n_qubits = n_qubits
            if self.verbose:
                print(f"  Tapered      : "
                      f"{self._taper_result.original_n_qubits} → {n_qubits} qubits")

        identity_offset, pauli_labels, pauli_coeffs = qubit_op_to_pauli_list(
            qubit_op, n_qubits
        )

        # --- Determine parameter count ---
        if self.ansatz == "custom":
            if callable(self.custom_ansatz):
                n_params = self.custom_ansatz_n_params
            else:
                # Pre-built circuit — zero variational parameters
                n_params = 0
        elif self.ansatz == "uccsd":
            n_params = uccsd_param_count(n_qubits, self._nelec)
        elif self.ansatz == "upccd":
            n_params = upccd_param_count(n_qubits, self._nelec)
        else:
            n_params = hardware_efficient_param_count(n_qubits, self.ansatz_layers)

        if self.verbose:
            print(f"  Parameters   : {n_params}")
            print(f"  Pauli terms  : {len(pauli_labels)}")

        # ──────────── ADAPT-VQE branch ────────────
        if self.ansatz == "adapt":
            from qoro_pyscf.adapt import run_adapt_vqe

            adapt_result = run_adapt_vqe(
                n_qubits=n_qubits,
                nelec=self._nelec,
                identity_offset=identity_offset,
                pauli_labels=pauli_labels,
                pauli_coeffs=pauli_coeffs,
                config=self._config,
                pool=self.adapt_pool,
                gradient_threshold=self.adapt_threshold,
                max_operators=self.adapt_max_ops,
                optimizer=self.optimizer,
                maxiter_per_step=self.maxiter,
                verbose=self.verbose,
                greedy=self.adapt_greedy,
            )

            self.optimal_params = adapt_result["params"]
            self._optimal_circuit = adapt_result["circuit"]
            self.converged = adapt_result["converged"]
            self.energy_history = adapt_result["energy_history"]
            self.vqe_time = 0.0  # timing is in ADAPT loop

            e_vqe = adapt_result["energy"]
            e_tot = e_vqe + ecore

            if self.verbose:
                print(f"  ADAPT operators: {adapt_result['n_operators']}")
                print(f"  E(ADAPT) = {e_vqe:+.10f}  Ha")
                print(f"  E(CASCI) = {e_tot:+.10f}  Ha")

            return e_tot, self

        # --- Clear caches ---
        self._rdm1s_cache = None
        self._rdm2s_cache = None
        self.energy_history = []
        iteration = [0]

        # --- Progress-tracking state ---
        _best_energy = [float("inf")]
        _best_iter = [0]
        _prev_energy = [None]
        _t_start = time.perf_counter()

        # --- Cost function ---
        def cost(params):
            if self.ansatz == "custom":
                if callable(self.custom_ansatz):
                    qc = self.custom_ansatz(params, n_qubits, self._nelec)
                else:
                    qc = self.custom_ansatz  # pre-built circuit
            elif self.ansatz == "uccsd":
                qc = uccsd_ansatz(params, n_qubits, self._nelec)
            elif self.ansatz == "upccd":
                qc = upccd_ansatz(params, n_qubits, self._nelec)
            else:
                qc = hardware_efficient_ansatz(
                    params, n_qubits, self.ansatz_layers,
                    include_hf=True, nelec=self._nelec,
                )
            energy = compute_energy(
                qc, identity_offset, pauli_labels, pauli_coeffs, self._config
            )

            # --- Spin penalty (if fix_spin_ was called) ---
            if self._spin_penalty_shift > 0:
                from qoro_pyscf.rdm import compute_1rdm_spatial
                rdm1_a, rdm1_b = compute_1rdm_spatial(
                    qc, n_qubits, self._config
                )
                n_a = np.trace(rdm1_a)
                n_b = np.trace(rdm1_b)
                sz = (n_a - n_b) / 2.0
                overlap = np.trace(rdm1_a @ rdm1_b)
                ss = sz * (sz + 1.0) + n_b - overlap
                energy += self._spin_penalty_shift * (ss - self._spin_penalty_ss) ** 2

            self.energy_history.append(energy)
            iteration[0] += 1

            # Update best-energy tracking
            if energy < _best_energy[0]:
                _best_energy[0] = energy
                _best_iter[0] = iteration[0]

            if self.verbose and (iteration[0] % 10 == 0 or iteration[0] == 1):
                elapsed = time.perf_counter() - _t_start
                delta_str = ""
                if _prev_energy[0] is not None:
                    delta = energy - _prev_energy[0]
                    delta_str = f"  ΔE = {delta:+.6f}"
                stagnation = iteration[0] - _best_iter[0]
                stag_str = f"  ⚠ STAGNATED ({stagnation} evals)" if stagnation >= 50 else ""
                print(
                    f"    iter {iteration[0]:4d}  E = {energy:+.10f}  Ha"
                    f"{delta_str}  best = {_best_energy[0]:+.10f}"
                    f"  [{elapsed:.1f}s]{stag_str}",
                    flush=True,
                )
            _prev_energy[0] = energy

            if self.callback is not None:
                self.callback(iteration[0], energy, params)

            return energy

        # --- Initial point ---
        if self.initial_point is not None:
            x0 = np.asarray(self.initial_point, dtype=float)
        elif self.ansatz in ("uccsd", "upccd"):
            # Chemistry ansatze: zero amplitudes = HF state.
            # Start near HF with small noise to break symmetry —
            # the optimizer finds the correlation corrections from there.
            rng = np.random.default_rng(42)
            x0 = rng.uniform(-0.05, 0.05, size=n_params)
        else:
            # Hardware-efficient / custom: no physical starting point,
            # use wider random initialization.
            rng = np.random.default_rng(42)
            x0 = rng.uniform(-np.pi / 4, np.pi / 4, size=n_params)

        # --- Pre-computed amplitudes mode (skip VQE) ---
        if self.maxiter == 0 and self.initial_point is not None:
            if self.verbose:
                print("  maxiter=0 — using pre-computed amplitudes (no VQE)")
            self.optimal_params = x0
            e_vqe = cost(x0)
            self.converged = True
            self.vqe_time = 0.0
        elif self.optimizer.upper() == "ADAM":
            # --- Adam with parameter-shift gradients ---
            t0 = time.perf_counter()
            params = x0.copy()
            m = np.zeros_like(params)    # 1st moment
            v = np.zeros_like(params)    # 2nd moment
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            best_energy = float('inf')
            best_params = params.copy()
            shift = self.grad_shift

            for it in range(1, self.maxiter + 1):
                # Central finite-difference gradient estimation
                # (exact parameter-shift rule only works for single-qubit
                # rotation gates; for Trotterized UCC circuits we need
                # standard finite differences with a small step)
                grad = np.zeros_like(params)
                for j in range(len(params)):
                    params_plus = params.copy()
                    params_minus = params.copy()
                    params_plus[j] += shift
                    params_minus[j] -= shift
                    grad[j] = (cost(params_plus) - cost(params_minus)) / (2 * shift)

                # Adam update
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2
                m_hat = m / (1 - beta1 ** it)
                v_hat = v / (1 - beta2 ** it)
                params -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

                # Evaluate & track best
                e = cost(params)
                if e < best_energy:
                    best_energy = e
                    best_params = params.copy()

            self.vqe_time = time.perf_counter() - t0
            self.converged = True
            self.optimal_params = best_params
            e_vqe = best_energy
        elif self.optimizer.upper() == "ROTOSOLVE":
            # --- Rotosolve: analytical single-parameter optimiser ---
            from qoro_pyscf.rotosolve import rotosolve_sweep

            t0 = time.perf_counter()
            opt_params, e_vqe, rs_history, rs_converged = rotosolve_sweep(
                cost, x0,
                max_sweeps=self.maxiter,
                tol=1e-8,
                verbose=self.verbose,
                callback=self.callback,
            )
            self.vqe_time = time.perf_counter() - t0
            self.converged = rs_converged
            self.optimal_params = opt_params
            # Extend energy_history with per-sweep energies
            self.energy_history.extend(rs_history)
        else:
            # --- SciPy optimiser ---
            opts: dict = {"maxiter": self.maxiter}
            if self.optimizer.upper() == "COBYLA":
                opts["rhobeg"] = 0.3

            t0 = time.perf_counter()
            opt = minimize(
                cost, x0,
                method=self.optimizer,
                options=opts,
            )
            self.vqe_time = time.perf_counter() - t0
            self.converged = opt.success
            self.optimal_params = opt.x
            e_vqe = opt.fun

        # --- Build the final optimised circuit (for RDM reconstruction) ---
        if self.ansatz == "custom":
            if callable(self.custom_ansatz):
                self._optimal_circuit = self.custom_ansatz(
                    self.optimal_params, n_qubits, self._nelec
                )
            else:
                self._optimal_circuit = self.custom_ansatz
        elif self.ansatz == "uccsd":
            self._optimal_circuit = uccsd_ansatz(
                self.optimal_params, n_qubits, self._nelec
            )
        elif self.ansatz == "upccd":
            self._optimal_circuit = upccd_ansatz(
                self.optimal_params, n_qubits, self._nelec
            )
        else:
            self._optimal_circuit = hardware_efficient_ansatz(
                self.optimal_params, n_qubits, self.ansatz_layers,
                include_hf=True, nelec=self._nelec,
            )

        e_tot = e_vqe + ecore

        if self.verbose:
            n_iters = len(self.energy_history)
            status = "converged" if self.converged else "not converged"
            print(f"  VQE {status} in {n_iters} iterations")
            print(f"  E(VQE)   = {e_vqe:+.10f}  Ha")
            print(f"  E(CASCI) = {e_tot:+.10f}  Ha  ({self.vqe_time:.2f} s)")

        # --- Multi-root VQD (nroots > 1) ---
        if self.nroots > 1:
            return self._run_vqd(
                e_tot, n_qubits, identity_offset, pauli_labels,
                pauli_coeffs, n_params, ecore,
            )

        # Return (energy, self) — self acts as the fake CI vector,
        # matching the qiskit-nature-pyscf convention.
        return e_tot, self

    def _run_vqd(
        self,
        ground_energy: float,
        n_qubits: int,
        identity_offset: float,
        pauli_labels: list,
        pauli_coeffs: np.ndarray,
        n_params: int,
        ecore: float,
    ):
        """
        Run VQD (Variational Quantum Deflation) for excited states.

        After the ground state is computed in ``kernel()``, this method
        computes ``nroots - 1`` additional excited states sequentially.
        Each state k is found by minimising:

            E_k(θ) = ⟨ψ(θ)|H|ψ(θ)⟩ + β Σ_{j<k} |⟨ψ_j|ψ(θ)⟩|²

        where β is :attr:`vqd_penalty` and ⟨ψ_j|ψ(θ)⟩ is the overlap
        with previously found states, computed via statevector inner product.

        Reference: Higgott, Wang & Brierley, Quantum 3, 156 (2019).

        Returns
        -------
        energies : np.ndarray, shape (nroots,)
            Total energies for each root.
        ci_vecs : list of QoroSolver
            ``[self] * nroots`` for PySCF compatibility.
        """
        from qoro_pyscf.expectation import compute_energy, compute_overlap

        # Store ground state
        self._vqd_energies = [ground_energy]
        self._vqd_circuits = [self._optimal_circuit]

        # Store circuits for overlap computation (compute_overlap handles API selection)
        previous_circuits = [self._optimal_circuit]

        if self.verbose:
            print(f"\n  [VQD] Computing {self.nroots - 1} excited states "
                  f"(β = {self.vqd_penalty})...")

        for root in range(1, self.nroots):
            if self.verbose:
                print(f"\n  [VQD] ── Root {root} ─────────────────────────")

            iteration_k = [0]

            def _build_circuit_k(params):
                """Build circuit for the current root."""
                if self.ansatz == "custom":
                    if callable(self.custom_ansatz):
                        return self.custom_ansatz(params, n_qubits, self._nelec)
                    return self.custom_ansatz
                elif self.ansatz == "uccsd":
                    return uccsd_ansatz(params, n_qubits, self._nelec)
                elif self.ansatz == "upccd":
                    return upccd_ansatz(params, n_qubits, self._nelec)
                else:
                    return hardware_efficient_ansatz(
                        params, n_qubits, self.ansatz_layers,
                        include_hf=True, nelec=self._nelec,
                    )

            def cost_vqd(params):
                qc = _build_circuit_k(params)

                # Base energy
                energy = compute_energy(
                    qc, identity_offset, pauli_labels, pauli_coeffs,
                    self._config,
                )

                # Overlap penalty — delegates to compute_overlap which automatically
                # uses maestro.inner_product when available
                for circ_prev in previous_circuits:
                    overlap = compute_overlap(circ_prev, qc, self._config)
                    energy += self.vqd_penalty * overlap

                iteration_k[0] += 1
                if self.verbose and (iteration_k[0] % 20 == 0 or iteration_k[0] == 1):
                    print(f"    iter {iteration_k[0]:4d}  |  E = {energy:+.10f}")

                if self.callback is not None:
                    self.callback(iteration_k[0], energy, params)

                return energy

            # Random initial point (different seed per root)
            rng = np.random.default_rng(42 + root)
            x0_k = rng.uniform(-np.pi / 4, np.pi / 4, size=n_params)

            opts: dict = {"maxiter": self.maxiter}
            if self.optimizer.upper() == "COBYLA":
                opts["rhobeg"] = 0.3

            opt_k = minimize(cost_vqd, x0_k, method=self.optimizer, options=opts)

            # Build final circuit for this root
            qc_k = _build_circuit_k(opt_k.x)
            e_k = compute_energy(
                qc_k, identity_offset, pauli_labels, pauli_coeffs, self._config
            ) + ecore

            self._vqd_energies.append(e_k)
            self._vqd_circuits.append(qc_k)

            # Store circuit for future overlap computations
            previous_circuits.append(qc_k)

            if self.verbose:
                print(f"  [VQD] Root {root}: E = {e_k:+.10f} Ha")

        energies = np.array(self._vqd_energies)

        if self.verbose:
            print(f"\n  [VQD] All roots: {energies}")

        return energies, [self] * self.nroots

    # CASSCF compatibility
    approx_kernel = kernel

    # ──────────────────────────────────────────────────────────────────────
    # RDM interface (PySCF protocol)
    # ──────────────────────────────────────────────────────────────────────

    def _ensure_rdm1s(self, fake_ci_vec: "QoroSolver") -> tuple[np.ndarray, np.ndarray]:
        """Compute and cache spin-resolved 1-RDMs."""
        solver = fake_ci_vec if isinstance(fake_ci_vec, QoroSolver) else self
        if solver._rdm1s_cache is None:
            solver._rdm1s_cache = compute_1rdm_spatial(
                solver._optimal_circuit, solver._n_qubits, solver._config
            )
        return solver._rdm1s_cache

    def _ensure_rdm2s(
        self, fake_ci_vec: "QoroSolver"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute and cache spin-resolved 2-RDMs."""
        solver = fake_ci_vec if isinstance(fake_ci_vec, QoroSolver) else self
        if solver._rdm2s_cache is None:
            solver._rdm2s_cache = compute_2rdm_spatial(
                solver._optimal_circuit, solver._n_qubits, solver._config
            )
        return solver._rdm2s_cache

    def make_rdm1(
        self,
        fake_ci_vec: "QoroSolver",
        norb: int,
        nelec: Union[int, tuple[int, int]],
    ) -> np.ndarray:
        """
        Construct the spin-traced 1-RDM.

        Parameters
        ----------
        fake_ci_vec : QoroSolver
            Reference to self (passed by PySCF as the CI vector).
        norb : int
            Number of active spatial orbitals.
        nelec : int or (int, int)
            Number of active electrons.

        Returns
        -------
        rdm1 : np.ndarray, shape (norb, norb)
            Spin-traced one-particle reduced density matrix.
        """
        rdm1_a, rdm1_b = self._ensure_rdm1s(fake_ci_vec)
        return trace_spin_rdm1(rdm1_a, rdm1_b)

    def make_rdm1s(
        self,
        fake_ci_vec: "QoroSolver",
        norb: int,
        nelec: Union[int, tuple[int, int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Construct the alpha- and beta-spin 1-RDMs.

        Returns
        -------
        (rdm1_a, rdm1_b) : tuple of np.ndarray
            Alpha and beta spin 1-RDMs, each shape (norb, norb).
        """
        return self._ensure_rdm1s(fake_ci_vec)

    def make_rdm12(
        self,
        fake_ci_vec: "QoroSolver",
        norb: int,
        nelec: Union[int, tuple[int, int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Construct the spin-traced 1- and 2-RDMs.

        The 2-RDM is returned in chemist index ordering as PySCF expects.

        Returns
        -------
        (rdm1, rdm2) : tuple of np.ndarray
            Spin-traced 1-RDM shape (norb, norb) and
            spin-traced 2-RDM shape (norb, norb, norb, norb).
        """
        rdm1_a, rdm1_b = self._ensure_rdm1s(fake_ci_vec)
        rdm2_aa, rdm2_ab, rdm2_bb = self._ensure_rdm2s(fake_ci_vec)
        return (
            trace_spin_rdm1(rdm1_a, rdm1_b),
            trace_spin_rdm2(rdm2_aa, rdm2_ab, rdm2_bb),
        )

    def make_rdm12s(
        self,
        fake_ci_vec: "QoroSolver",
        norb: int,
        nelec: Union[int, tuple[int, int]],
    ) -> tuple[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ]:
        """
        Construct the spin-resolved 1- and 2-RDMs.

        Returns
        -------
        ((rdm1_a, rdm1_b), (rdm2_aa, rdm2_ab, rdm2_bb))
            Alpha/beta 1-RDMs and αα/αβ/ββ 2-RDMs in chemist ordering.
        """
        rdm1s = self._ensure_rdm1s(fake_ci_vec)
        rdm2s = self._ensure_rdm2s(fake_ci_vec)
        return rdm1s, rdm2s

    def spin_square(
        self,
        fake_ci_vec: "QoroSolver",
        norb: int,
        nelec: Union[int, tuple[int, int]],
    ) -> tuple[float, float]:
        """
        Compute ⟨S²⟩ and 2S+1 from the spin-resolved RDMs.

        Uses the standard decomposition:

            ⟨S²⟩ = (N_α - N_β)² / 4
                  + (N_α + N_β) / 2
                  - Tr(D_α · D_β)

        where D_α and D_β are the alpha and beta 1-RDMs and N is the
        trace (electron count).

        Required by PySCF's CASSCF for spin characterisation.

        Returns
        -------
        (ss, multip) : (float, float)
            ⟨S²⟩ and spin multiplicity 2S+1.
        """
        rdm1_a, rdm1_b = self._ensure_rdm1s(fake_ci_vec)

        n_alpha = np.trace(rdm1_a)
        n_beta = np.trace(rdm1_b)

        # ⟨S_z⟩ = (N_α - N_β) / 2
        sz = (n_alpha - n_beta) / 2.0

        # ⟨S²⟩ = ⟨S_z²⟩ + ⟨S_z⟩ + ⟨S-S+⟩
        # For a single-determinant-like state approximation:
        # ⟨S²⟩ ≈ S_z(S_z + 1) + N_β - Tr(D_α · D_β)
        overlap = np.trace(rdm1_a @ rdm1_b)
        ss = sz * (sz + 1.0) + n_beta - overlap

        multip = np.sqrt(abs(ss) + 0.25) * 2  # 2S+1
        return float(ss), float(multip)

    # ──────────────────────────────────────────────────────────────────────
    # Advanced features (Izmaylov-lab workflows)
    # ──────────────────────────────────────────────────────────────────────

    def evaluate_custom_paulis(
        self,
        pauli_terms: list[tuple[complex, str]],
        circuit: Optional["QuantumCircuit"] = None,
    ) -> float:
        """
        Evaluate a user-supplied Pauli Hamiltonian on the GPU.

        Bypasses PySCF / OpenFermion Hamiltonian generation entirely.
        Useful for measurement-reduction research (e.g. Hamiltonian
        factorisation, covariance-based grouping).

        Parameters
        ----------
        pauli_terms : list of (complex, str)
            Each entry is ``(coefficient, pauli_string)`` where
            ``pauli_string`` uses ``{I, X, Y, Z}`` characters,
            e.g. ``[(0.5, "ZZII"), (-0.3, "IXYZ")]``.
            Identity-only strings (``"IIII"``) are separated out and
            accumulated as a scalar offset.
        circuit : QuantumCircuit or None
            Circuit to evaluate on. Defaults to the optimised circuit
            from the most recent ``kernel()`` call.

        Returns
        -------
        expectation : float
            ⟨ψ|H_custom|ψ⟩ = Σ Re(cᵢ)·⟨Pᵢ⟩.

        Raises
        ------
        RuntimeError
            If no circuit is available (``kernel()`` not yet called and
            ``circuit`` not provided).
        """
        from qoro_pyscf.expectation import evaluate_expectation

        qc = circuit if circuit is not None else self._optimal_circuit
        if qc is None:
            raise RuntimeError(
                "No circuit available. Run kernel() first or pass an "
                "explicit `circuit` argument."
            )
        if self._config is None:
            self._config = configure_backend(
                use_gpu=(self.backend == "gpu"),
                simulation=self.simulation,
                mps_bond_dim=self.mps_bond_dim,
                license_key=self.license_key,
            )

        # Separate identity terms from non-identity Pauli strings
        identity_offset = 0.0
        pauli_labels: list[str] = []
        pauli_coeffs: list[float] = []

        for coeff, label in pauli_terms:
            if set(label) <= {"I"}:
                identity_offset += float(np.real(coeff))
            else:
                pauli_labels.append(label)
                pauli_coeffs.append(float(np.real(coeff)))

        if not pauli_labels:
            return identity_offset

        exp_vals = evaluate_expectation(qc, pauli_labels, self._config)
        coeffs = np.asarray(pauli_coeffs, dtype=float)
        return identity_offset + float(np.dot(coeffs, exp_vals))

    def get_final_statevector(
        self,
        circuit: Optional["QuantumCircuit"] = None,
    ) -> np.ndarray:
        """
        Retrieve the exact statevector from the Qoro simulator.

        Returns the full complex-amplitude vector (or MPS-contracted state)
        after VQE convergence.  This is intended for fidelity benchmarking
        against exact diagonalisation.

        Parameters
        ----------
        circuit : QuantumCircuit or None
            Circuit whose state to extract. Defaults to the optimised
            circuit from the most recent ``kernel()`` call.

        Returns
        -------
        statevector : np.ndarray, shape (2**n_qubits,)
            Complex-valued amplitudes of the prepared state.

        Raises
        ------
        RuntimeError
            If no circuit is available.
        """
        qc = circuit if circuit is not None else self._optimal_circuit
        if qc is None:
            raise RuntimeError(
                "No circuit available. Run kernel() first or pass an "
                "explicit `circuit` argument."
            )

        # TODO: implement using maestro.get_state_vector once available in the
        # installed wheel. Tracked in qoro-pyscf issue: get_state_vector binding.
        raise NotImplementedError(
            "get_final_statevector() requires maestro.get_state_vector(), which "
            "is not yet available in the installed qoro-maestro wheel. "
            "Rebuild Maestro from source to enable this method."
        )

    def fix_spin_(self, shift: float = 0.2, ss: float | None = None):
        """
        Apply a spin-penalty to the VQE cost function.

        Modifies the solver in-place so that subsequent ``kernel`` calls
        penalise states with ⟨S²⟩ away from the target value.  The penalty
        adds ``shift * (⟨S²⟩ - target)²`` to the energy at each iteration.

        Parameters
        ----------
        shift : float
            Penalty strength (Ha). Default: 0.2.
        ss : float or None
            Target ⟨S²⟩. None = singlet (0.0).

        Returns
        -------
        self : QoroSolver
            For chaining.
        """
        self._spin_penalty_shift = shift
        self._spin_penalty_ss = ss if ss is not None else 0.0
        return self

    # ──────────────────────────────────────────────────────────────────────
    # Serialisation — save / load
    # ──────────────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """
        Save solver state to disk for checkpointing or reproducibility.

        Stores configuration, optimal parameters, and energy history as a
        ``.npz`` + ``.json`` pair.  The circuit itself is **not** saved
        (it is rebuilt from parameters on ``load``).

        Parameters
        ----------
        path : str or Path
            Output path **without** extension.  Two files are created:
            ``<path>.json`` (config) and ``<path>.npz`` (arrays).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config = {
            "ansatz": self.ansatz,
            "ansatz_layers": self.ansatz_layers,
            "optimizer": self.optimizer,
            "maxiter": self.maxiter,
            "backend": self.backend,
            "simulation": self.simulation,
            "mps_bond_dim": self.mps_bond_dim,
            "verbose": self.verbose,
            "converged": self.converged,
            "vqe_time": self.vqe_time,
            "n_qubits": self._n_qubits,
            "nelec": list(self._nelec),
            "adapt_threshold": self.adapt_threshold,
            "adapt_max_ops": self.adapt_max_ops,
            "adapt_pool": self.adapt_pool,
            "_spin_penalty_shift": self._spin_penalty_shift,
            "_spin_penalty_ss": self._spin_penalty_ss,
        }

        with open(f"{path}.json", "w") as f:
            json.dump(config, f, indent=2)

        arrays: dict = {}
        if self.optimal_params is not None:
            arrays["optimal_params"] = self.optimal_params
        if self.energy_history:
            arrays["energy_history"] = np.array(self.energy_history)
        if self.initial_point is not None:
            arrays["initial_point"] = np.asarray(self.initial_point)

        np.savez(f"{path}.npz", **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "QoroSolver":
        """
        Restore a solver from a checkpoint saved by :meth:`save`.

        The returned solver has its parameters and energy history
        restored.  To reuse the optimised state without re-running VQE,
        pass the restored solver as the CASCI ``fcisolver`` and use
        ``maxiter=0``.

        Parameters
        ----------
        path : str or Path
            Base path (without extension) used during :meth:`save`.

        Returns
        -------
        QoroSolver
            A new solver instance with restored state.
        """
        path = Path(path)

        with open(f"{path}.json", "r") as f:
            config = json.load(f)

        data = np.load(f"{path}.npz", allow_pickle=False)

        solver = cls(
            ansatz=config["ansatz"],
            ansatz_layers=config.get("ansatz_layers", 2),
            optimizer=config.get("optimizer", "COBYLA"),
            maxiter=config.get("maxiter", 200),
            backend=config.get("backend", "gpu"),
            simulation=config.get("simulation", "statevector"),
            mps_bond_dim=config.get("mps_bond_dim", 64),
            verbose=config.get("verbose", True),
            adapt_threshold=config.get("adapt_threshold", 1e-3),
            adapt_max_ops=config.get("adapt_max_ops", 50),
            adapt_pool=config.get("adapt_pool", "sd"),
        )

        # Restore internal state
        solver.converged = config.get("converged", False)
        solver.vqe_time = config.get("vqe_time", 0.0)
        solver._n_qubits = config.get("n_qubits", 0)
        solver._nelec = tuple(config.get("nelec", (0, 0)))
        solver._spin_penalty_shift = config.get("_spin_penalty_shift", 0.0)
        solver._spin_penalty_ss = config.get("_spin_penalty_ss", 0.0)

        if "optimal_params" in data:
            solver.optimal_params = data["optimal_params"]
            solver.initial_point = data["optimal_params"].copy()
        if "energy_history" in data:
            solver.energy_history = data["energy_history"].tolist()

        return solver


