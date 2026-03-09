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
Active-space selection helpers.

Provides convenience wrappers around PySCF's active-space selection tools
so that users can determine ``(norb, nelec, mo_coeff)`` automatically
instead of guessing.

Two strategies are offered:

1. **AVAS** (Atomic Valence Active Space) — the user specifies which
   atomic orbitals matter (e.g. ``"Fe 3d"``), and AVAS projects the
   molecular orbitals onto that AO subspace.
2. **MP2 natural orbitals** — for cases where the user doesn't know
   which AOs matter.  Fractional occupations from MP2 flag the
   strongly-correlated orbitals.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def suggest_active_space(
    mf,
    ao_labels: str | list[str],
    threshold: float = 0.2,
    minao: str = "minao",
    verbose: Optional[int] = None,
) -> tuple[int, tuple[int, int], np.ndarray]:
    """
    Select an active space using AVAS (Atomic Valence Active Space).

    Wraps PySCF's ``mcscf.avas.avas()`` with sensible defaults and a
    cleaner return signature.

    Parameters
    ----------
    mf : pyscf.scf.hf.SCF
        A converged HF (or DFT) object.
    ao_labels : str or list of str
        Atomic orbital labels defining the valence space, e.g.
        ``"Fe 3d"`` or ``["C 2pz", "N 2pz"]``.
    threshold : float
        AVAS truncation threshold above which AOs are kept.
        Default: 0.2 (keeps orbitals with ≥20% valence character).
    minao : str
        Reference minimal AO basis for projection. Default: ``"minao"``.
    verbose : int or None
        PySCF verbosity level. None = use molecule default.

    Returns
    -------
    norb : int
        Number of active spatial orbitals.
    nelec : (int, int)
        Number of active (alpha, beta) electrons.
    mo_coeff : np.ndarray
        Reordered MO coefficients for CASCI/CASSCF.

    Examples
    --------
    >>> from pyscf import gto, scf
    >>> from qoro_maestro_pyscf import suggest_active_space, MaestroSolver
    >>> mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="cc-pvdz")
    >>> mf = scf.RHF(mol).run()
    >>> norb, nelec, mo = suggest_active_space(mf, "N 2p")
    >>> cas = mcscf.CASCI(mf, norb, nelec)
    >>> cas.fcisolver = MaestroSolver(ansatz="uccsd")
    >>> cas.run(mo)
    """
    from pyscf.mcscf import avas

    if isinstance(ao_labels, str):
        ao_labels = [ao_labels]

    kwargs = {"threshold": threshold, "minao": minao}
    if verbose is not None:
        kwargs["verbose"] = verbose

    norb, nelec_total, mo_coeff = avas.avas(mf, ao_labels, **kwargs)

    # avas returns nelec as int or numpy scalar; convert to (alpha, beta)
    if isinstance(nelec_total, (int, np.integer)):
        n_beta = int(nelec_total) // 2
        n_alpha = int(nelec_total) - n_beta
        nelec = (n_alpha, n_beta)
    else:
        nelec = tuple(int(x) for x in nelec_total)

    logger.info(
        "AVAS active space: (%de, %do) → %d qubits",
        sum(nelec), norb, 2 * norb,
    )

    return norb, nelec, mo_coeff


def suggest_active_space_from_mp2(
    mf,
    threshold: float = 0.02,
    max_orbitals: Optional[int] = None,
) -> tuple[int, tuple[int, int], np.ndarray]:
    """
    Select an active space from MP2 natural orbital occupations.

    Orbitals with fractional occupations (significantly different from
    0 or 2) indicate strong correlation.  This method identifies them
    without requiring the user to specify atomic orbital labels.

    Parameters
    ----------
    mf : pyscf.scf.hf.SCF
        A converged RHF object.
    threshold : float
        Orbitals with ``threshold < occupation < 2 - threshold`` are
        included in the active space. Default: 0.02.
    max_orbitals : int or None
        Cap the number of active orbitals. None = no cap.

    Returns
    -------
    norb : int
        Number of active spatial orbitals.
    nelec : (int, int)
        Number of active (alpha, beta) electrons.
    mo_coeff : np.ndarray
        Natural orbital coefficients (reordered: core / active / virtual).

    Examples
    --------
    >>> from pyscf import gto, scf
    >>> from qoro_maestro_pyscf import suggest_active_space_from_mp2
    >>> mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g")
    >>> mf = scf.RHF(mol).run()
    >>> norb, nelec, mo = suggest_active_space_from_mp2(mf)
    """
    from pyscf import mp as pyscf_mp

    # Run MP2
    mp2 = pyscf_mp.MP2(mf).run()

    # Compute MP2 natural orbital occupations
    # make_rdm1() returns the 1-RDM in the MO basis
    rdm1_mo = mp2.make_rdm1()
    occ, nat_coeffs = np.linalg.eigh(rdm1_mo)

    # Sort descending
    idx = np.argsort(occ)[::-1]
    occ = occ[idx]
    nat_coeffs = nat_coeffs[:, idx]

    # Natural orbital coefficients in AO basis
    no_coeff = mf.mo_coeff @ nat_coeffs

    # Identify active orbitals: fractional occupation
    active_mask = (occ > threshold) & (occ < 2.0 - threshold)

    if not np.any(active_mask):
        # Fallback: all orbitals with occupation between 0.01 and 1.99
        active_mask = (occ > 0.01) & (occ < 1.99)

    if not np.any(active_mask):
        raise ValueError(
            "No fractionally-occupied orbitals found. "
            "The system may not have significant static correlation. "
            "Try lowering the threshold or using suggest_active_space() "
            "with explicit AO labels."
        )

    active_indices = np.where(active_mask)[0]

    if max_orbitals is not None and len(active_indices) > max_orbitals:
        # Keep the most fractional orbitals
        deviations = np.abs(occ[active_indices] - 1.0)
        keep = np.argsort(deviations)[:max_orbitals]
        active_indices = np.sort(active_indices[keep])

    norb = len(active_indices)

    # Count active electrons from rounded occupations
    n_active_electrons = int(round(np.sum(occ[active_indices])))
    n_beta = n_active_electrons // 2
    n_alpha = n_active_electrons - n_beta
    nelec = (n_alpha, n_beta)

    # Reorder MOs: core (doubly occupied) | active | virtual (empty)
    core_indices = np.where(occ >= 2.0 - threshold)[0]
    virt_indices = np.where(occ <= threshold)[0]

    reorder = np.concatenate([core_indices, active_indices, virt_indices])
    mo_coeff = no_coeff[:, reorder]

    logger.info(
        "MP2 active space: (%de, %do) → %d qubits  "
        "[occupations: %s]",
        sum(nelec), norb, 2 * norb,
        ", ".join(f"{o:.3f}" for o in occ[active_indices]),
    )

    return norb, nelec, mo_coeff
