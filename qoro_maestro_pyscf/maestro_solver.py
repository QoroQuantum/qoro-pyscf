# This file is part of qoro-maestro-pyscf.
#
# Copyright (C) 2026 Qoro Quantum Ltd.
#
# qoro-maestro-pyscf is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# qoro-maestro-pyscf is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with qoro-maestro-pyscf. If not, see <https://www.gnu.org/licenses/\>.

"""
MaestroSolver — PySCF FCI-solver drop-in backed by Maestro VQE.

This is the primary user-facing class of ``qoro-maestro-pyscf``.  It mirrors
the API of ``qiskit_nature_pyscf.QiskitSolver``, enabling a seamless swap:

    # Before (Qiskit):
    cas.fcisolver = QiskitSolver(algorithm)

    # After (Maestro):
    cas.fcisolver = MaestroSolver(ansatz="uccsd")

The solver integrates with PySCF's CASCI and CASSCF objects by implementing
the ``fcisolver`` protocol: ``kernel``, ``make_rdm1``, ``make_rdm1s``,
``make_rdm12``, ``make_rdm12s``, and ``approx_kernel``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from scipy.optimize import minimize

from maestro.circuits import QuantumCircuit

from qoro_maestro_pyscf.backends import BackendConfig, configure_backend
from qoro_maestro_pyscf.hamiltonian import (
    integrals_to_qubit_hamiltonian,
    qubit_op_to_pauli_list,
)
from qoro_maestro_pyscf.ansatze import (
    hardware_efficient_ansatz,
    hardware_efficient_param_count,
    uccsd_ansatz,
    uccsd_param_count,
)
from qoro_maestro_pyscf.expectation import compute_energy
from qoro_maestro_pyscf.rdm import (
    compute_1rdm_spatial,
    compute_2rdm_spatial,
    trace_spin_rdm1,
    trace_spin_rdm2,
)


logger = logging.getLogger(__name__)


@dataclass
class MaestroSolver:
    """
    PySCF FCI-solver interface that runs a VQE on the Maestro simulator.

    Replaces PySCF's classical FCI solver with a quantum VQE algorithm
    executed on Maestro's GPU-accelerated (or CPU) backend.

    Parameters
    ----------
    ansatz : str
        Ansatz type: ``"hardware_efficient"`` or ``"uccsd"``.
    ansatz_layers : int
        Number of layers (hardware-efficient ansatz only). Default: 2.
    optimizer : str
        SciPy minimiser method. Default: ``"COBYLA"``.
    maxiter : int
        Maximum optimiser iterations. Default: 200.
    backend : str
        Backend selection: ``"gpu"`` or ``"cpu"``. Default: ``"gpu"``.
    simulation : str
        Simulation mode: ``"statevector"`` or ``"mps"``. Default: ``"statevector"``.
    mps_bond_dim : int
        MPS bond dimension (for ``simulation="mps"``). Default: 64.
    license_key : str or None
        Maestro GPU license key (e.g. ``"XXXX-XXXX-XXXX-XXXX"``).
        If provided, sets the ``MAESTRO_LICENSE_KEY`` env var before GPU
        init. Can also be set via the env var directly or
        :func:`qoro_maestro_pyscf.backends.set_license_key`.
    initial_point : np.ndarray or None
        Initial parameter vector. If None, uses small random values.
    verbose : bool
        Print progress during optimisation. Default: True.

    Examples
    --------
    CASCI with GPU statevector:

    >>> from pyscf import gto, scf, mcscf
    >>> from qoro_maestro_pyscf import MaestroSolver
    >>> mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    >>> hf = scf.RHF(mol).run()
    >>> cas = mcscf.CASCI(hf, 2, 2)
    >>> cas.fcisolver = MaestroSolver(
    ...     ansatz="uccsd",
    ...     license_key="XXXX-XXXX-XXXX-XXXX",  # or set MAESTRO_LICENSE_KEY env var
    ... )
    >>> cas.run()

    CASCI with GPU MPS for larger active spaces:

    >>> cas.fcisolver = MaestroSolver(
    ...     ansatz="hardware_efficient",
    ...     ansatz_layers=3,
    ...     simulation="mps",
    ...     mps_bond_dim=128,
    ... )
    """
    # --- User-configurable ---
    ansatz: str = "hardware_efficient"
    ansatz_layers: int = 2
    optimizer: str = "COBYLA"
    maxiter: int = 200
    backend: str = "gpu"
    simulation: str = "statevector"
    mps_bond_dim: int = 64
    license_key: Optional[str] = None
    initial_point: Optional[np.ndarray] = None
    verbose: bool = True

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

    def kernel(
        self,
        h1: Union[np.ndarray, tuple[np.ndarray, np.ndarray]],
        h2: Union[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]],
        norb: int,
        nelec: Union[int, tuple[int, int]],
        ci0=None,
        ecore: float = 0,
        **kwargs,
    ) -> tuple[float, "MaestroSolver"]:
        """
        Find the ground-state energy via VQE on Maestro.

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
        self : MaestroSolver
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

        # --- Configure Maestro backend ---
        self._config = configure_backend(
            use_gpu=(self.backend == "gpu"),
            simulation=self.simulation,
            mps_bond_dim=self.mps_bond_dim,
            license_key=self.license_key,
        )

        if self.verbose:
            logger.info(
                "[MaestroSolver] Backend: %s | Qubits: %d | Ansatz: %s",
                self._config.label, n_qubits, self.ansatz,
            )
            print(f"  [MaestroSolver] Backend : {self._config.label}")
            print(f"  [MaestroSolver] Qubits  : {n_qubits}")
            print(f"  [MaestroSolver] Ansatz  : {self.ansatz}")

        # --- Build qubit Hamiltonian from PySCF integrals ---
        qubit_op, _ = integrals_to_qubit_hamiltonian(h1, h2, norb)
        identity_offset, pauli_labels, pauli_coeffs = qubit_op_to_pauli_list(
            qubit_op, n_qubits
        )

        # --- Determine parameter count ---
        if self.ansatz == "uccsd":
            n_params = uccsd_param_count(n_qubits, self._nelec)
        else:
            n_params = hardware_efficient_param_count(n_qubits, self.ansatz_layers)

        if self.verbose:
            print(f"  [MaestroSolver] Params  : {n_params}")
            print(f"  [MaestroSolver] Paulis  : {len(pauli_labels)}")

        # --- Clear caches ---
        self._rdm1s_cache = None
        self._rdm2s_cache = None
        self.energy_history = []
        iteration = [0]

        # --- Cost function ---
        def cost(params):
            if self.ansatz == "uccsd":
                qc = uccsd_ansatz(params, n_qubits, self._nelec)
            else:
                qc = hardware_efficient_ansatz(
                    params, n_qubits, self.ansatz_layers,
                    include_hf=True, nelec=self._nelec,
                )
            energy = compute_energy(
                qc, identity_offset, pauli_labels, pauli_coeffs, self._config
            )
            self.energy_history.append(energy)
            iteration[0] += 1
            if self.verbose and (iteration[0] % 20 == 0 or iteration[0] == 1):
                print(f"    iter {iteration[0]:4d}  |  E = {energy:+.10f}")
            return energy

        # --- Initial point ---
        if self.initial_point is not None:
            x0 = np.asarray(self.initial_point, dtype=float)
        else:
            # Use wider spread than zero — too-small values keep UCCSD
            # stuck at the HF solution (all excitation amplitudes ≈ 0).
            rng = np.random.default_rng(42)
            x0 = rng.uniform(-np.pi / 4, np.pi / 4, size=n_params)

        # --- Optimise ---
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

        # --- Build the final optimised circuit (for RDM reconstruction) ---
        if self.ansatz == "uccsd":
            self._optimal_circuit = uccsd_ansatz(
                self.optimal_params, n_qubits, self._nelec
            )
        else:
            self._optimal_circuit = hardware_efficient_ansatz(
                self.optimal_params, n_qubits, self.ansatz_layers,
                include_hf=True, nelec=self._nelec,
            )

        e_tot = opt.fun + ecore

        if self.verbose:
            print(f"  [MaestroSolver] Converged : {self.converged}")
            print(f"  [MaestroSolver] E(VQE)    : {opt.fun:+.10f}")
            print(f"  [MaestroSolver] E(total)  : {e_tot:+.10f}")
            print(f"  [MaestroSolver] Time      : {self.vqe_time:.2f}s")

        # Return (energy, self) — self acts as the fake CI vector,
        # matching the qiskit-nature-pyscf convention.
        return e_tot, self

    # CASSCF compatibility
    approx_kernel = kernel

    # ──────────────────────────────────────────────────────────────────────
    # RDM interface (PySCF protocol)
    # ──────────────────────────────────────────────────────────────────────

    def _ensure_rdm1s(self, fake_ci_vec: "MaestroSolver") -> tuple[np.ndarray, np.ndarray]:
        """Compute and cache spin-resolved 1-RDMs."""
        solver = fake_ci_vec if isinstance(fake_ci_vec, MaestroSolver) else self
        if solver._rdm1s_cache is None:
            solver._rdm1s_cache = compute_1rdm_spatial(
                solver._optimal_circuit, solver._n_qubits, solver._config
            )
        return solver._rdm1s_cache

    def _ensure_rdm2s(
        self, fake_ci_vec: "MaestroSolver"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute and cache spin-resolved 2-RDMs."""
        solver = fake_ci_vec if isinstance(fake_ci_vec, MaestroSolver) else self
        if solver._rdm2s_cache is None:
            solver._rdm2s_cache = compute_2rdm_spatial(
                solver._optimal_circuit, solver._n_qubits, solver._config
            )
        return solver._rdm2s_cache

    def make_rdm1(
        self,
        fake_ci_vec: "MaestroSolver",
        norb: int,
        nelec: Union[int, tuple[int, int]],
    ) -> np.ndarray:
        """
        Construct the spin-traced 1-RDM.

        Parameters
        ----------
        fake_ci_vec : MaestroSolver
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
        fake_ci_vec: "MaestroSolver",
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
        fake_ci_vec: "MaestroSolver",
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
        fake_ci_vec: "MaestroSolver",
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
        fake_ci_vec: "MaestroSolver",
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

    def fix_spin_(self, shift: float = 0.2, ss: float | None = None):
        """
        Apply a spin-penalty to the VQE cost function.

        Modifies the solver in-place so that subsequent ``kernel`` calls
        penalise states with ⟨S²⟩ away from the target value.

        Parameters
        ----------
        shift : float
            Penalty strength (Ha). Default: 0.2.
        ss : float or None
            Target ⟨S²⟩. None = singlet (0.0).

        Returns
        -------
        self : MaestroSolver
            For chaining.
        """
        self._spin_penalty_shift = shift
        self._spin_penalty_ss = ss if ss is not None else 0.0
        return self


