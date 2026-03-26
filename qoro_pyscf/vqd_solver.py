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
VQDSolver — Excited-state solver using Variational Quantum Deflation (VQD).

Wraps :class:`QoroSolver` to compute multiple excited states sequentially.
Each excited state *k* is found by minimising a modified cost function that
penalises overlap with all previously found states:

    C(θ) = ⟨ψ_k(θ)|H|ψ_k(θ)⟩ + Σ_{i<k} β_i |⟨ψ_i|ψ_k(θ)⟩|²

Reference: Higgott, Wang & Brierley, Quantum 3, 156 (2019).

Example
-------
::

    from pyscf import gto, scf, mcscf
    from qoro_pyscf import QoroSolver, VQDSolver

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    hf = scf.RHF(mol).run()

    cas = mcscf.CASCI(hf, 2, 2)
    vqd = VQDSolver(
        solver=QoroSolver(ansatz="uccsd"),
        num_states=3,
        penalty_weights=5.0,
    )
    cas.fcisolver = vqd
    cas.run()

    print(vqd.energies)          # [E0, E1, E2]
    print(vqd.optimized_states)  # per-state params/circuits
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np
from scipy.optimize import minimize

if TYPE_CHECKING:
    from maestro.circuits import QuantumCircuit

from qoro_pyscf.qoro_solver import QoroSolver
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
class _OptimizedState:
    """Internal record of a single optimized VQD state."""
    root: int
    energy: float
    params: np.ndarray
    circuit: "QuantumCircuit"
    probabilities: np.ndarray  # probability distribution for overlap penalty


@dataclass
class VQDSolver:
    """
    Excited-state solver using Variational Quantum Deflation (VQD).

    Wraps a :class:`QoroSolver` to compute ground and excited states
    sequentially.  For each excited state *k*, the cost function is:

        C(θ) = ⟨ψ_k(θ)|H|ψ_k(θ)⟩ + Σ_{i<k} β_i F(ψ_i, ψ_k(θ))

    where β_i are the ``penalty_weights`` and F is the fidelity between
    states.  Currently uses probability-based Bhattacharyya fidelity
    (a lower bound); will use exact statevector overlap |⟨ψ_i|ψ_k⟩|²
    once ``maestro.get_state_vector()`` is exposed.

    Parameters
    ----------
    solver : QoroSolver
        The inner ground-state VQE solver.  All ansatz, optimizer,
        backend, and simulation settings are inherited from this solver.
    num_states : int
        Total number of states to compute (1 = ground only). Default: 2.
    penalty_weights : float or list[float]
        Overlap penalty strength(s) β.  A single float applies uniformly
        to all overlap terms.  A list allows per-state tuning; element *i*
        is the penalty weight β_i for the overlap with state *i*.
        Default: 5.0.
    callback : callable or None
        If provided, called at each VQE iteration with signature
        ``(root: int, iteration: int, energy: float, params: np.ndarray)``.

    Attributes
    ----------
    energies : np.ndarray
        Total energies for each root (available after ``kernel()``).
    optimized_states : list[dict]
        Per-state optimization results with keys:
        ``root``, ``energy``, ``params``.
    converged : bool
        True if all roots converged.

    Examples
    --------
    H₂ ground + first excited state:

    >>> from pyscf import gto, scf, mcscf
    >>> from qoro_pyscf import QoroSolver, VQDSolver
    >>> mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    >>> hf = scf.RHF(mol).run()
    >>> cas = mcscf.CASCI(hf, 2, 2)
    >>> cas.fcisolver = VQDSolver(
    ...     solver=QoroSolver(ansatz="uccsd"),
    ...     num_states=2,
    ...     penalty_weights=10.0,
    ... )
    >>> cas.run()
    """

    # --- User-configurable ---
    solver: QoroSolver = field(default_factory=QoroSolver)
    num_states: int = 2
    penalty_weights: Union[float, list[float]] = 5.0
    callback: Optional[Callable[[int, int, float, np.ndarray], None]] = None

    # --- PySCF interface attributes (set by CASCI/CASSCF) ---
    mol: object = field(default=None, repr=False)
    nroots: int = 1  # PySCF may set this; we prefer num_states

    # --- Internal state (populated after kernel runs) ---
    converged: bool = field(default=False, init=False, repr=False)
    energies: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    optimized_states: list = field(default_factory=list, init=False, repr=False)
    _config: Optional[BackendConfig] = field(default=None, init=False, repr=False)
    _n_qubits: int = field(default=0, init=False, repr=False)
    _nelec: tuple = field(default=(0, 0), init=False, repr=False)
    _optimal_circuit: Optional["QuantumCircuit"] = field(
        default=None, init=False, repr=False
    )
    _rdm1s_cache: Optional[tuple] = field(default=None, init=False, repr=False)
    _rdm2s_cache: Optional[tuple] = field(default=None, init=False, repr=False)

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _resolve_penalty_weights(self) -> list[float]:
        """Expand penalty_weights to a list of length ``num_states - 1``."""
        n = max(self.num_states - 1, 0)
        if isinstance(self.penalty_weights, (int, float)):
            return [float(self.penalty_weights)] * n
        weights = [float(w) for w in self.penalty_weights]
        if len(weights) < n:
            # Pad with the last weight value
            weights.extend([weights[-1]] * (n - len(weights)))
        return weights[:n]

    def _build_circuit(
        self,
        params: np.ndarray,
        n_qubits: int,
        nelec: tuple[int, int],
    ) -> "QuantumCircuit":
        """Build a parameterised circuit using the inner solver's ansatz."""
        s = self.solver
        if s.ansatz == "custom":
            if callable(s.custom_ansatz):
                return s.custom_ansatz(params, n_qubits, nelec)
            return s.custom_ansatz
        elif s.ansatz == "uccsd":
            return uccsd_ansatz(params, n_qubits, nelec)
        elif s.ansatz == "upccd":
            return upccd_ansatz(params, n_qubits, nelec)
        else:
            return hardware_efficient_ansatz(
                params, n_qubits, s.ansatz_layers,
                include_hf=True, nelec=nelec,
            )

    def _get_probabilities(self, circuit: "QuantumCircuit") -> np.ndarray:
        """Extract the probability distribution from a circuit (for RDM/debug use)."""
        from qoro_pyscf.expectation import get_state_probabilities
        return get_state_probabilities(circuit, self._config)

    def _compute_overlap(self, circuit_a: "QuantumCircuit", circuit_b: "QuantumCircuit") -> float:
        """Compute |⟨ψ_a|ψ_b⟩|² using the best available Qoro API.

        Delegates to :func:`compute_overlap`, which uses
        ``maestro.inner_product`` when available and falls back to
        the Bhattacharyya coefficient otherwise.
        """
        from qoro_pyscf.expectation import compute_overlap
        return compute_overlap(circuit_a, circuit_b, self._config)

    def _param_count(self, n_qubits: int, nelec: tuple[int, int]) -> int:
        """Determine the number of variational parameters."""
        s = self.solver
        if s.ansatz == "custom":
            if callable(s.custom_ansatz):
                return s.custom_ansatz_n_params
            return 0
        elif s.ansatz == "uccsd":
            return uccsd_param_count(n_qubits, nelec)
        elif s.ansatz == "upccd":
            return upccd_param_count(n_qubits, nelec)
        else:
            return hardware_efficient_param_count(n_qubits, s.ansatz_layers)

    def _initial_point(
        self,
        n_params: int,
        seed: int,
        prev_params: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate an initial parameter vector.

        For excited states (root > 0), start from the previous root's
        optimal parameters plus a structured perturbation. This ensures
        the initial circuit is genuinely different in quantum state space,
        giving the Bhattacharyya penalty a non-zero gradient from iteration 1.
        """
        s = self.solver
        if s.initial_point is not None and seed == 42:
            return np.asarray(s.initial_point, dtype=float)

        if prev_params is not None:
            # For excited states: start from previous root's optimum + perturbation.
            # Using alternating +π/2 / -π/2 offset ensures the circuit is
            # far from the previous state in Hilbert space.
            rng = np.random.default_rng(seed)
            perturbation = rng.choice([-np.pi / 2, np.pi / 2], size=n_params)
            return prev_params + perturbation

        rng = np.random.default_rng(seed)
        if s.ansatz in ("uccsd", "upccd"):
            return rng.uniform(-0.05, 0.05, size=n_params)
        else:
            return rng.uniform(-np.pi / 4, np.pi / 4, size=n_params)

    # ──────────────────────────────────────────────────────────────────────
    # PySCF kernel interface
    # ──────────────────────────────────────────────────────────────────────

    def kernel(
        self,
        h1: Union[np.ndarray, tuple[np.ndarray, np.ndarray]],
        h2: Union[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]],
        norb: int,
        nelec: Union[int, tuple[int, int]],
        ci0=None,
        ecore: float = 0,
        **kwargs,
    ) -> tuple[np.ndarray, list["VQDSolver"]]:
        """
        Compute ground and excited states via VQD.

        Implements PySCF's ``fcisolver.kernel`` protocol but returns
        energies for all requested roots.

        Parameters
        ----------
        h1 : ndarray or (ndarray, ndarray)
            One-electron integrals.
        h2 : ndarray or (ndarray, ndarray, ndarray)
            Two-electron integrals (chemist notation).
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
        energies : np.ndarray, shape (num_states,)
            Total energies for each root.
        ci_vecs : list of VQDSolver
            ``[self] * num_states`` for PySCF compatibility.
        """
        # Use num_states or PySCF's nroots, whichever is larger
        total_states = max(self.num_states, self.nroots)

        # --- Resolve electron counts ---
        if isinstance(nelec, int):
            n_beta = nelec // 2
            n_alpha = nelec - n_beta
            self._nelec = (n_alpha, n_beta)
        else:
            self._nelec = nelec

        n_qubits = 2 * norb
        self._n_qubits = n_qubits

        # --- Validate custom ansatz early ---
        s = self.solver
        if s.ansatz == "custom":
            if s.custom_ansatz is None:
                raise ValueError(
                    "ansatz='custom' requires `custom_ansatz` to be set."
                )
            if callable(s.custom_ansatz) and s.custom_ansatz_n_params is None:
                raise ValueError(
                    "When `custom_ansatz` is a callable, "
                    "`custom_ansatz_n_params` must be set."
                )

        # --- Configure Qoro backend ---
        self._config = configure_backend(
            use_gpu=(s.backend == "gpu"),
            simulation=s.simulation,
            mps_bond_dim=s.mps_bond_dim,
            license_key=s.license_key,
        )

        verbose = s.verbose
        if verbose:
            logger.info(
                "VQD Solver: Backend=%s, Qubits=%d, Ansatz=%s, States=%d",
                self._config.label, n_qubits, s.ansatz, total_states,
            )
            print(f"\nVQD Solver (Qoro)")
            print(f"  Active space : ({sum(self._nelec)}e, {norb}o) → {n_qubits} qubits")
            print(f"  Ansatz       : {s.ansatz}")
            print(f"  Backend      : {self._config.label}")
            print(f"  States       : {total_states}")

        # --- Build qubit Hamiltonian ---
        qubit_op, _ = integrals_to_qubit_hamiltonian(h1, h2, norb)

        # --- Optional Z₂ tapering ---
        if s.taper:
            from qoro_pyscf.tapering import taper_hamiltonian
            taper_result = taper_hamiltonian(qubit_op, n_qubits, self._nelec)
            qubit_op = taper_result.tapered_op
            n_qubits = taper_result.tapered_n_qubits
            self._n_qubits = n_qubits
            if verbose:
                print(f"  Tapered      : "
                      f"{taper_result.original_n_qubits} → {n_qubits} qubits")

        identity_offset, pauli_labels, pauli_coeffs = qubit_op_to_pauli_list(
            qubit_op, n_qubits
        )

        # --- Determine parameter count ---
        n_params = self._param_count(n_qubits, self._nelec)

        if verbose:
            print(f"  Parameters   : {n_params}")
            print(f"  Pauli terms  : {len(pauli_labels)}")

        # --- Resolve penalty weights ---
        weights = self._resolve_penalty_weights()
        if verbose:
            print(f"  Penalty β    : {weights}")

        # --- Clear state ---
        self.optimized_states = []
        self._rdm1s_cache = None
        self._rdm2s_cache = None

        # Store circuits + optimal params for overlap and seeding
        previous_circuits: list["QuantumCircuit"] = []
        previous_params: list[np.ndarray] = []

        # ──────────── Compute each root sequentially ────────────
        all_energies = []

        for root in range(total_states):
            if verbose:
                print(f"\n  ── Root {root} {'(ground state)' if root == 0 else ''} "
                      f"──────────────────────────")

            iteration_k = [0]

            def _cost_vqd(params, _root=root, _previous_circuits=previous_circuits):
                """VQD cost function for root _root."""
                qc = self._build_circuit(params, n_qubits, self._nelec)

                # Base energy ⟨ψ(θ)|H|ψ(θ)⟩
                energy = compute_energy(
                    qc, identity_offset, pauli_labels, pauli_coeffs,
                    self._config,
                )

                # Overlap penalty — delegates to compute_overlap which automatically
                # uses maestro.inner_product when available
                if _root > 0:
                    for i, circ_prev in enumerate(_previous_circuits):
                        beta_i = weights[i] if i < len(weights) else weights[-1]
                        energy += beta_i * self._compute_overlap(circ_prev, qc)

                iteration_k[0] += 1
                if verbose and (iteration_k[0] % 20 == 0 or iteration_k[0] == 1):
                    tag = f"root {_root}" if _root > 0 else "ground"
                    print(f"    iter {iteration_k[0]:4d}  [{tag}]  "
                          f"E = {energy:+.10f}  Ha")

                if self.callback is not None:
                    self.callback(_root, iteration_k[0], energy, params)

                return energy

            # --- Initial point ---
            prev = previous_params[-1] if previous_params else None
            x0 = self._initial_point(n_params, seed=42 + root, prev_params=prev)

            # --- Optimize ---
            t0 = time.perf_counter()

            if s.optimizer.upper() == "ADAM":
                # Adam with parameter-shift gradients
                params = x0.copy()
                m = np.zeros_like(params)
                v = np.zeros_like(params)
                beta1, beta2, eps = 0.9, 0.999, 1e-8
                best_energy = float('inf')
                best_params = params.copy()
                shift = s.grad_shift

                for it in range(1, s.maxiter + 1):
                    grad = np.zeros_like(params)
                    for j in range(len(params)):
                        p_plus = params.copy()
                        p_minus = params.copy()
                        p_plus[j] += shift
                        p_minus[j] -= shift
                        grad[j] = (_cost_vqd(p_plus) - _cost_vqd(p_minus)) / (2 * shift)

                    m = beta1 * m + (1 - beta1) * grad
                    v = beta2 * v + (1 - beta2) * grad ** 2
                    m_hat = m / (1 - beta1 ** it)
                    v_hat = v / (1 - beta2 ** it)
                    params -= s.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

                    e = _cost_vqd(params)
                    if e < best_energy:
                        best_energy = e
                        best_params = params.copy()

                opt_time = time.perf_counter() - t0
                opt_params = best_params
                opt_converged = True
            else:
                # SciPy optimizer
                opts: dict = {"maxiter": s.maxiter}
                if s.optimizer.upper() == "COBYLA":
                    opts["rhobeg"] = 0.3

                opt = minimize(
                    _cost_vqd, x0,
                    method=s.optimizer,
                    options=opts,
                )
                opt_time = time.perf_counter() - t0
                opt_params = opt.x
                opt_converged = opt.success

            # --- Build final circuit and compute clean energy ---
            qc_final = self._build_circuit(opt_params, n_qubits, self._nelec)
            e_root = compute_energy(
                qc_final, identity_offset, pauli_labels, pauli_coeffs,
                self._config,
            ) + ecore

            # --- Store circuit + params for future overlap computations ---
            previous_circuits.append(qc_final)
            previous_params.append(opt_params)

            # --- Record this state ---
            state = _OptimizedState(
                root=root,
                energy=e_root,
                params=opt_params,
                circuit=qc_final,
                probabilities=self._get_probabilities(qc_final),
            )
            self.optimized_states.append(state)
            all_energies.append(e_root)

            if verbose:
                status = "converged" if opt_converged else "not converged"
                print(f"  Root {root}: E = {e_root:+.10f} Ha "
                      f"({status}, {opt_time:.2f} s)")

        # --- Finalize ---
        self.energies = np.array(all_energies)
        self.converged = True  # all roots attempted

        # Set the last root's circuit as the "active" one for RDM methods
        if self.optimized_states:
            self._optimal_circuit = self.optimized_states[-1].circuit

        if verbose:
            print(f"\n  [VQD] All roots: {self.energies}")

        return self.energies, [self] * total_states

    # CASSCF compatibility
    approx_kernel = kernel

    # ──────────────────────────────────────────────────────────────────────
    # State access
    # ──────────────────────────────────────────────────────────────────────

    def get_state(self, root: int) -> dict:
        """
        Get optimization results for a specific root.

        Parameters
        ----------
        root : int
            State index (0 = ground).

        Returns
        -------
        dict
            Keys: ``root``, ``energy``, ``params``.
        """
        if root < 0 or root >= len(self.optimized_states):
            raise IndexError(
                f"Root {root} not available. "
                f"Only {len(self.optimized_states)} states computed."
            )
        s = self.optimized_states[root]
        return {"root": s.root, "energy": s.energy, "params": s.params.copy()}

    def get_statevector(self, root: int) -> np.ndarray:
        """
        Get the probability distribution for a specific root.

        .. note::
            Returns probabilities (not complex amplitudes) until
            ``maestro.get_state_vector()`` is available.

        Parameters
        ----------
        root : int
            State index (0 = ground).

        Returns
        -------
        np.ndarray, shape (2**n_qubits,)
            Probability distribution.
        """
        if root < 0 or root >= len(self.optimized_states):
            raise IndexError(
                f"Root {root} not available. "
                f"Only {len(self.optimized_states)} states computed."
            )
        return self.optimized_states[root].probabilities.copy()

    # ──────────────────────────────────────────────────────────────────────
    # RDM interface (PySCF protocol) — operates on the last computed state
    # ──────────────────────────────────────────────────────────────────────

    def _ensure_rdm1s(
        self, fake_ci_vec: "VQDSolver"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute and cache spin-resolved 1-RDMs."""
        solver = fake_ci_vec if isinstance(fake_ci_vec, VQDSolver) else self
        if solver._rdm1s_cache is None:
            solver._rdm1s_cache = compute_1rdm_spatial(
                solver._optimal_circuit, solver._n_qubits, solver._config
            )
        return solver._rdm1s_cache

    def _ensure_rdm2s(
        self, fake_ci_vec: "VQDSolver"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute and cache spin-resolved 2-RDMs."""
        solver = fake_ci_vec if isinstance(fake_ci_vec, VQDSolver) else self
        if solver._rdm2s_cache is None:
            solver._rdm2s_cache = compute_2rdm_spatial(
                solver._optimal_circuit, solver._n_qubits, solver._config
            )
        return solver._rdm2s_cache

    def make_rdm1(
        self,
        fake_ci_vec: "VQDSolver",
        norb: int,
        nelec: Union[int, tuple[int, int]],
    ) -> np.ndarray:
        """Construct the spin-traced 1-RDM for the active state."""
        rdm1_a, rdm1_b = self._ensure_rdm1s(fake_ci_vec)
        return trace_spin_rdm1(rdm1_a, rdm1_b)

    def make_rdm1s(
        self,
        fake_ci_vec: "VQDSolver",
        norb: int,
        nelec: Union[int, tuple[int, int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct the alpha- and beta-spin 1-RDMs."""
        return self._ensure_rdm1s(fake_ci_vec)

    def make_rdm12(
        self,
        fake_ci_vec: "VQDSolver",
        norb: int,
        nelec: Union[int, tuple[int, int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct the spin-traced 1- and 2-RDMs."""
        rdm1_a, rdm1_b = self._ensure_rdm1s(fake_ci_vec)
        rdm2_aa, rdm2_ab, rdm2_bb = self._ensure_rdm2s(fake_ci_vec)
        return (
            trace_spin_rdm1(rdm1_a, rdm1_b),
            trace_spin_rdm2(rdm2_aa, rdm2_ab, rdm2_bb),
        )

    def make_rdm12s(
        self,
        fake_ci_vec: "VQDSolver",
        norb: int,
        nelec: Union[int, tuple[int, int]],
    ) -> tuple[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ]:
        """Construct the spin-resolved 1- and 2-RDMs."""
        rdm1s = self._ensure_rdm1s(fake_ci_vec)
        rdm2s = self._ensure_rdm2s(fake_ci_vec)
        return rdm1s, rdm2s

    def spin_square(
        self,
        fake_ci_vec: "VQDSolver",
        norb: int,
        nelec: Union[int, tuple[int, int]],
    ) -> tuple[float, float]:
        """Compute ⟨S²⟩ and 2S+1 for the active state."""
        rdm1_a, rdm1_b = self._ensure_rdm1s(fake_ci_vec)
        n_alpha = np.trace(rdm1_a)
        n_beta = np.trace(rdm1_b)
        sz = (n_alpha - n_beta) / 2.0
        overlap = np.trace(rdm1_a @ rdm1_b)
        ss = sz * (sz + 1.0) + n_beta - overlap
        multip = np.sqrt(abs(ss) + 0.25) * 2
        return float(ss), float(multip)

    def set_active_root(self, root: int) -> None:
        """
        Set which root is used for RDM methods and property evaluation.

        Parameters
        ----------
        root : int
            State index (0 = ground).
        """
        if root < 0 or root >= len(self.optimized_states):
            raise IndexError(
                f"Root {root} not available. "
                f"Only {len(self.optimized_states)} states computed."
            )
        state = self.optimized_states[root]
        self._optimal_circuit = state.circuit
        self._rdm1s_cache = None
        self._rdm2s_cache = None
