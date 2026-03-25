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
QSCISolver — PySCF FCI-solver drop-in backed by Quantum-Selected CI.

Implements the QSCI algorithm (arXiv:2302.11320): a hybrid quantum-classical
method that uses a quantum computer (via VQE on Maestro) to *select* important
electron configurations, then classically diagonalizes the Hamiltonian in the
subspace spanned by those configurations.

Usage::

    from qoro_maestro_pyscf import MaestroSolver, QSCISolver

    inner = MaestroSolver(ansatz="uccsd", backend="gpu")
    cas.fcisolver = QSCISolver(inner_solver=inner)
    cas.run()

References
----------
[1] Kanno et al., arXiv:2302.11320 (QSCI)
[2] Robledo-Moreno et al., arXiv:2405.05068 (SQD at scale)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    from maestro.circuits import QuantumCircuit

from qoro_maestro_pyscf.maestro_solver import MaestroSolver

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Probability extraction via Z-projectors
# ──────────────────────────────────────────────────────────────────────────────

def _compute_probabilities_via_z_projectors(
    circuit: "QuantumCircuit",
    n_qubits: int,
    config,
) -> np.ndarray:
    """
    Compute the full probability distribution P(|k⟩) from Z-string expectations.

    Uses the identity:

        P(|k⟩) = (1/2^n) Σ_{z} (-1)^{popcount(k & z)} ⟨Z_z⟩

    where z iterates over all 2^n bitmasks selecting which qubits get Z vs I,
    and ⟨Z_z⟩ is the expectation value of the corresponding Z-string operator.

    All Z-strings are batched into a single ``circuit.estimate()`` call,
    so Maestro evaluates them in one statevector/MPS pass.

    Parameters
    ----------
    circuit : QuantumCircuit
        The prepared (optimised) circuit.
    n_qubits : int
        Number of qubits (spin-orbitals).
    config : BackendConfig
        Maestro backend configuration.

    Returns
    -------
    probabilities : ndarray, shape (2**n_qubits,)
        Probability of each computational basis state.
    """
    from qoro_maestro_pyscf.expectation import evaluate_expectation

    n_states = 2 ** n_qubits

    # Build all Z-string observables (excluding identity at mask=0)
    z_labels = []
    for mask in range(1, n_states):
        label = ['I'] * n_qubits
        for i in range(n_qubits):
            if (mask >> i) & 1:
                label[i] = 'Z'
        z_labels.append(''.join(label))

    # Single batched estimate call
    z_exp_vals = evaluate_expectation(circuit, z_labels, config)

    # Prepend identity expectation (always 1.0)
    all_exp_vals = np.empty(n_states)
    all_exp_vals[0] = 1.0
    all_exp_vals[1:] = z_exp_vals

    # Compute probabilities via Walsh-Hadamard-like transform
    probs = np.zeros(n_states)
    for k in range(n_states):
        val = 0.0
        for mask in range(n_states):
            # (-1)^{popcount(k & mask)}
            sign = (-1) ** bin(k & mask).count('1')
            val += sign * all_exp_vals[mask]
        probs[k] = val / n_states

    # Clamp small numerical noise to zero
    probs = np.maximum(probs, 0.0)

    return probs


# ──────────────────────────────────────────────────────────────────────────────
# Bitstring ↔ determinant helpers
# ──────────────────────────────────────────────────────────────────────────────

def _probabilities_to_determinants(
    probabilities: np.ndarray,
    n_qubits: int,
    nelec: tuple[int, int],
    n_samples: int,
    probability_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a probability vector to selected α/β determinant strings.

    Each index *k* in the probability vector corresponds to a computational
    basis state |k⟩.  In the Jordan-Wigner mapping used by maestro-pyscf,
    even-indexed qubits are α spin-orbitals and odd-indexed are β.

    This function:
    1. Filters states below ``probability_threshold``
    2. Postselects states with the correct electron number per spin sector
    3. Keeps the top ``n_samples`` states by probability
    4. Splits each bitstring into α and β parts and converts to PySCF's
       determinant integer encoding

    Parameters
    ----------
    probabilities : ndarray, shape (2**n_qubits,)
        Probability distribution over computational basis states.
    n_qubits : int
        Number of qubits (spin-orbitals).
    nelec : (int, int)
        (n_alpha, n_beta) electron counts.
    n_samples : int
        Maximum number of determinants to keep.
    probability_threshold : float
        Minimum probability to be considered.

    Returns
    -------
    ci_strs_a : ndarray of int64
        Unique α determinant strings (integer-encoded).
    ci_strs_b : ndarray of int64
        Unique β determinant strings (integer-encoded).
    selected_probs : ndarray
        Probabilities of the selected bitstrings.
    """
    n_alpha, n_beta = nelec
    n_spatial = n_qubits // 2

    # Step 1: threshold filter
    candidates = np.where(probabilities > probability_threshold)[0]
    if len(candidates) == 0:
        raise ValueError(
            f"No bitstrings above probability threshold {probability_threshold}. "
            "Try lowering the threshold or improving the VQE result."
        )

    # Step 2: postselect by Hamming weight on α/β sectors
    valid_indices = []
    for idx in candidates:
        bits = _index_to_bits(idx, n_qubits)
        # JW interleaved: even qubits = α, odd qubits = β
        alpha_bits = bits[0::2]  # positions 0, 2, 4, ...
        beta_bits = bits[1::2]   # positions 1, 3, 5, ...
        if np.sum(alpha_bits) == n_alpha and np.sum(beta_bits) == n_beta:
            valid_indices.append(idx)

    if len(valid_indices) == 0:
        raise ValueError(
            f"No bitstrings with correct electron counts "
            f"(n_α={n_alpha}, n_β={n_beta}) found above threshold. "
            "The VQE state may not preserve particle number."
        )

    valid_indices = np.array(valid_indices)
    valid_probs = probabilities[valid_indices]

    # Step 3: sort by probability and take top n_samples
    order = np.argsort(valid_probs)[::-1]
    n_keep = min(n_samples, len(order))
    top_indices = valid_indices[order[:n_keep]]
    selected_probs = valid_probs[order[:n_keep]]

    # Step 4: convert to α/β determinant integers
    alpha_strs_set = set()
    beta_strs_set = set()

    for idx in top_indices:
        bits = _index_to_bits(idx, n_qubits)
        alpha_bits = bits[0::2]
        beta_bits = bits[1::2]
        alpha_strs_set.add(_bits_to_int(alpha_bits, n_spatial))
        beta_strs_set.add(_bits_to_int(beta_bits, n_spatial))

    ci_strs_a = np.array(sorted(alpha_strs_set), dtype=np.int64)
    ci_strs_b = np.array(sorted(beta_strs_set), dtype=np.int64)

    return ci_strs_a, ci_strs_b, selected_probs


def _index_to_bits(index: int, n_qubits: int) -> np.ndarray:
    """Convert a basis state index to a binary array (LSB-first)."""
    return np.array([(index >> i) & 1 for i in range(n_qubits)], dtype=int)


def _bits_to_int(bits: np.ndarray, n_bits: int) -> int:
    """Convert a binary array (LSB-first) to an integer."""
    result = 0
    for i in range(n_bits):
        if bits[i]:
            result |= (1 << i)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# QSCISolver
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QSCISolver:
    """
    PySCF FCI-solver that runs QSCI (Quantum-Selected Configuration Interaction).

    Uses an inner ``MaestroSolver`` to prepare a VQE state, samples it in the
    computational basis, selects the most important electron configurations,
    and classically diagonalizes the Hamiltonian in that subspace using
    PySCF's selected CI machinery.

    The result is **variational** (always ≥ true ground-state energy) and
    **robust to noise** because the quantum device is used only for
    configuration selection, not energy evaluation.

    Parameters
    ----------
    inner_solver : MaestroSolver
        VQE solver for state preparation. Any ansatz is supported.
    n_samples : int
        Maximum number of determinants to include in the CI subspace.
        Default: 500. Larger values give more accurate results but
        increase classical diagonalization cost.
    probability_threshold : float
        Minimum probability for a configuration to be considered.
        Default: 1e-8.
    sci_max_iterations : int
        Maximum self-consistent configuration recovery iterations.
        Default: 1 (no self-consistency; single-shot QSCI).
    sci_convergence_tol : float
        Energy convergence tolerance for the self-consistent loop.
        Default: 1e-6 Ha.
    verbose : bool
        Print progress. Default: True.

    Examples
    --------
    CASCI with QSCI (CPU, works out of the box):

    >>> from pyscf import gto, scf, mcscf
    >>> from qoro_maestro_pyscf import MaestroSolver, QSCISolver
    >>> mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    >>> hf = scf.RHF(mol).run()
    >>> cas = mcscf.CASCI(hf, 2, 2)
    >>> inner = MaestroSolver(ansatz="uccsd")
    >>> cas.fcisolver = QSCISolver(inner_solver=inner)
    >>> cas.run()

    GPU-accelerated QSCI with larger subspace:

    >>> inner = MaestroSolver(ansatz="uccsd", backend="gpu")
    >>> cas.fcisolver = QSCISolver(inner_solver=inner, n_samples=2000)
    """

    # --- User-configurable ---
    inner_solver: MaestroSolver = field(default_factory=MaestroSolver)
    n_samples: int = 500
    probability_threshold: float = 1e-8
    sci_max_iterations: int = 1
    sci_convergence_tol: float = 1e-6
    verbose: bool = True

    # --- PySCF interface attributes (set by CASCI/CASSCF) ---
    mol: object = field(default=None, repr=False)
    nroots: int = 1

    # --- Internal state (populated after kernel runs) ---
    converged: bool = field(default=False, init=False, repr=False)
    qsci_time: float = field(default=0.0, init=False, repr=False)
    vqe_energy: float = field(default=0.0, init=False, repr=False)
    qsci_energy: float = field(default=0.0, init=False, repr=False)
    n_determinants: int = field(default=0, init=False, repr=False)
    _n_qubits: int = field(default=0, init=False, repr=False)
    _nelec: tuple = field(default=(0, 0), init=False, repr=False)
    _norb: int = field(default=0, init=False, repr=False)
    _ci_vector: object = field(default=None, init=False, repr=False)
    _ci_strs: object = field(default=None, init=False, repr=False)

    def kernel(
        self,
        h1: Union[np.ndarray, tuple[np.ndarray, np.ndarray]],
        h2: Union[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]],
        norb: int,
        nelec: Union[int, tuple[int, int]],
        ci0=None,
        ecore: float = 0,
        **kwargs,
    ) -> tuple[float, "QSCISolver"]:
        """
        Find the ground-state energy via QSCI.

        Implements PySCF's ``fcisolver.kernel`` protocol.

        Flow:
        1. Run inner VQE solver → optimized circuit
        2. Extract probability distribution → select determinants
        3. Classically diagonalize in the selected subspace

        Parameters
        ----------
        h1, h2, norb, nelec, ci0, ecore
            Standard PySCF fcisolver.kernel arguments.

        Returns
        -------
        e_tot : float
            Total QSCI energy (QSCI eigenvalue + ecore).
        self : QSCISolver
            Reference to this solver (acts as CI vector for PySCF).
        """
        from pyscf import ao2mo
        from pyscf.fci import selected_ci

        t0 = time.perf_counter()

        # --- Resolve electron counts ---
        if isinstance(nelec, int):
            n_beta = nelec // 2
            n_alpha = nelec - n_beta
            self._nelec = (n_alpha, n_beta)
        else:
            self._nelec = nelec

        self._norb = norb
        self._n_qubits = 2 * norb
        n_alpha, n_beta = self._nelec

        if self.verbose:
            print(f"\n╔══ QSCI Solver (Maestro) ══════════════════════════")
            print(f"║  Active space : ({sum(self._nelec)}e, {norb}o) → "
                  f"{self._n_qubits} qubits")
            print(f"║  Max samples  : {self.n_samples}")
            print(f"║  Inner ansatz : {self.inner_solver.ansatz}")
            print(f"║  Backend      : {self.inner_solver.backend}")
            print(f"╠══ Step 1: VQE state preparation ══════════════════")

        # ── Step 1: Run inner VQE ────────────────────────────────────
        self.inner_solver.mol = self.mol
        e_vqe, _ = self.inner_solver.kernel(h1, h2, norb, nelec, ci0, ecore, **kwargs)
        self.vqe_energy = e_vqe

        if self.verbose:
            print(f"║  E(VQE) = {e_vqe:+.10f} Ha")
            print(f"╠══ Step 2: Sampling & configuration selection ═════")

        # ── Step 2: Extract probabilities and select determinants ────
        circuit = self.inner_solver._optimal_circuit
        config = self.inner_solver._config

        probabilities = _compute_probabilities_via_z_projectors(
            circuit, self._n_qubits, config,
        )

        ci_strs_a, ci_strs_b, selected_probs = _probabilities_to_determinants(
            probabilities,
            self._n_qubits,
            self._nelec,
            self.n_samples,
            self.probability_threshold,
        )

        subspace_dim = len(ci_strs_a) * len(ci_strs_b)

        if self.verbose:
            print(f"║  Selected dets: {len(ci_strs_a)} α × {len(ci_strs_b)} β "
                  f"= {subspace_dim} subspace")
            print(f"║  Prob coverage: {np.sum(selected_probs):.6f}")
            print(f"╠══ Step 3: Classical diagonalization ══════════════")

        # ── Step 3: Subspace diagonalization via PySCF selected_ci ───
        # Restore h2 to full 4D tensor if compressed
        if isinstance(h2, tuple):
            h2_full = h2  # UHF — pass through
        else:
            h2_full = ao2mo.restore(1, h2, norb) if h2.ndim != 4 else h2

        # Build the selected CI Hamiltonian and diagonalize
        ci_strs = (ci_strs_a, ci_strs_b)

        # Create a SelectedCI solver and set the determinant strings
        sci_solver = selected_ci.SelectedCI()
        sci_solver._strs = ci_strs
        e_sci, ci_vec = sci_solver.kernel(h1, h2_full, norb, self._nelec)

        e_tot = e_sci + ecore
        self.qsci_energy = e_tot
        self._ci_vector = ci_vec
        self._ci_strs = ci_strs
        self.n_determinants = subspace_dim
        self.converged = True
        self.qsci_time = time.perf_counter() - t0

        if self.verbose:
            improvement = self.vqe_energy - e_tot
            print(f"║  E(QSCI)      = {e_tot:+.10f} Ha")
            print(f"║  Improvement   = {improvement:+.2e} Ha vs VQE")
            print(f"║  Total time    = {self.qsci_time:.2f} s")
            print(f"╚══════════════════════════════════════════════════")

        return e_tot, self

    # CASSCF compatibility
    approx_kernel = kernel

    # ──────────────────────────────────────────────────────────────────
    # RDM interface (PySCF protocol)
    # ──────────────────────────────────────────────────────────────────

    def make_rdm1(
        self,
        fake_ci_vec: "QSCISolver",
        norb: int,
        nelec: Union[int, tuple[int, int]],
    ) -> np.ndarray:
        """
        Spin-traced 1-RDM from the QSCI eigenvector.

        Uses PySCF's ``selected_ci.make_rdm1`` — no quantum resources needed.
        """
        from pyscf.fci.selected_ci import make_rdm1, _as_SCIvector

        solver = fake_ci_vec if isinstance(fake_ci_vec, QSCISolver) else self
        sci_vec = _as_SCIvector(solver._ci_vector, solver._ci_strs)
        return make_rdm1(sci_vec, solver._norb, solver._nelec)

    def make_rdm1s(
        self,
        fake_ci_vec: "QSCISolver",
        norb: int,
        nelec: Union[int, tuple[int, int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Alpha and beta 1-RDMs from the QSCI eigenvector."""
        from pyscf.fci.selected_ci import make_rdm1s, _as_SCIvector

        solver = fake_ci_vec if isinstance(fake_ci_vec, QSCISolver) else self
        sci_vec = _as_SCIvector(solver._ci_vector, solver._ci_strs)
        return make_rdm1s(sci_vec, solver._norb, solver._nelec)

    def make_rdm12(
        self,
        fake_ci_vec: "QSCISolver",
        norb: int,
        nelec: Union[int, tuple[int, int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Spin-traced 1-RDM and 2-RDM from the QSCI eigenvector."""
        from pyscf.fci.selected_ci import make_rdm1, make_rdm2, _as_SCIvector

        solver = fake_ci_vec if isinstance(fake_ci_vec, QSCISolver) else self
        sci_vec = _as_SCIvector(solver._ci_vector, solver._ci_strs)
        rdm1 = make_rdm1(sci_vec, solver._norb, solver._nelec)
        rdm2 = make_rdm2(sci_vec, solver._norb, solver._nelec)
        return rdm1, rdm2

    def make_rdm12s(
        self,
        fake_ci_vec: "QSCISolver",
        norb: int,
        nelec: Union[int, tuple[int, int]],
    ) -> tuple[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ]:
        """Spin-resolved 1-RDMs and 2-RDMs."""
        from pyscf.fci.selected_ci import (
            make_rdm1s, make_rdm2s, _as_SCIvector,
        )

        solver = fake_ci_vec if isinstance(fake_ci_vec, QSCISolver) else self
        sci_vec = _as_SCIvector(solver._ci_vector, solver._ci_strs)
        rdm1s = make_rdm1s(sci_vec, solver._norb, solver._nelec)
        rdm2s = make_rdm2s(sci_vec, solver._norb, solver._nelec)
        return rdm1s, rdm2s

    def spin_square(
        self,
        fake_ci_vec: "QSCISolver",
        norb: int,
        nelec: Union[int, tuple[int, int]],
    ) -> tuple[float, float]:
        """Compute ⟨S²⟩ and 2S+1 from the QSCI eigenvector."""
        from pyscf.fci.selected_ci import spin_square, _as_SCIvector

        solver = fake_ci_vec if isinstance(fake_ci_vec, QSCISolver) else self
        sci_vec = _as_SCIvector(solver._ci_vector, solver._ci_strs)
        ss, multip = spin_square(sci_vec, solver._norb, solver._nelec)
        return float(ss), float(multip)
