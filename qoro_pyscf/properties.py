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
Molecular property computation from VQE results.

Computes dipole moments, natural orbital occupations, and other one-electron
properties from the RDMs produced by QoroSolver.
"""

from __future__ import annotations

import numpy as np


def compute_dipole_moment(
    mol,
    mo_coeff: np.ndarray,
    rdm1: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Compute the electronic + nuclear dipole moment.

    Uses the spin-traced 1-RDM from the VQE and PySCF's integral engine
    for the dipole integrals.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        The PySCF molecule object.
    mo_coeff : np.ndarray, shape (nao, nmo)
        MO coefficient matrix (e.g. from ``cas.mo_coeff``).
    rdm1 : np.ndarray, shape (norb, norb)
        Spin-traced 1-RDM in the active-space MO basis.

    Returns
    -------
    dipole_vec : np.ndarray, shape (3,)
        Dipole moment vector [x, y, z] in Debye.
    dipole_mag : float
        Total dipole moment magnitude in Debye.
    """
    AU_TO_DEBYE = 2.541746473

    # Nuclear contribution
    charges = mol.atom_charges()
    coords = mol.atom_coords()
    nuc_dipole = np.einsum("i,ix->x", charges, coords)

    # Electronic contribution: expand active-space RDM to full AO basis
    norb = rdm1.shape[0]
    nmo = mo_coeff.shape[1]

    # Build full MO 1-RDM (core + active)
    ncore = (nmo - norb) // 2 if nmo > norb else 0
    full_rdm1_mo = np.zeros((nmo, nmo))

    # Core orbitals are doubly occupied
    for i in range(ncore):
        full_rdm1_mo[i, i] = 2.0

    # Active space
    full_rdm1_mo[ncore:ncore + norb, ncore:ncore + norb] = rdm1

    # Transform to AO basis
    rdm1_ao = mo_coeff @ full_rdm1_mo @ mo_coeff.T

    # Dipole integrals in AO basis
    with mol.with_common_orig((0, 0, 0)):
        dip_ints = mol.intor_symmetric("int1e_r", comp=3)  # (3, nao, nao)

    elec_dipole = -np.einsum("xij,ji->x", dip_ints, rdm1_ao)

    dipole = (nuc_dipole + elec_dipole) * AU_TO_DEBYE
    magnitude = np.linalg.norm(dipole)

    return dipole, magnitude


def compute_natural_orbitals(
    rdm1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute natural orbital occupations from the 1-RDM.

    Natural orbitals are the eigenvectors of the 1-RDM, and their
    eigenvalues give the occupation numbers. Fractional occupations
    indicate multi-reference / strongly correlated character.

    Parameters
    ----------
    rdm1 : np.ndarray, shape (norb, norb)
        Spin-traced 1-RDM.

    Returns
    -------
    occupations : np.ndarray, shape (norb,)
        Natural orbital occupations (sorted descending).
    nat_orb_coeffs : np.ndarray, shape (norb, norb)
        Natural orbital coefficients (columns are orbitals).
    """
    occupations, nat_orb_coeffs = np.linalg.eigh(rdm1)
    # Sort descending by occupation
    idx = np.argsort(occupations)[::-1]
    return occupations[idx], nat_orb_coeffs[:, idx]


def compute_mulliken_spin_population(
    mol,
    mo_coeff: np.ndarray,
    rdm1_a: np.ndarray,
    rdm1_b: np.ndarray,
) -> np.ndarray:
    """
    Compute Mulliken spin population on each atom.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        The PySCF molecule object.
    mo_coeff : np.ndarray
        MO coefficient matrix.
    rdm1_a, rdm1_b : np.ndarray
        Alpha and beta 1-RDMs in the active-space MO basis.

    Returns
    -------
    spin_pop : np.ndarray, shape (natom,)
        Spin population (N_alpha - N_beta) on each atom.
    """
    spin_rdm1 = rdm1_a - rdm1_b
    norb = spin_rdm1.shape[0]
    nmo = mo_coeff.shape[1]
    ncore = (nmo - norb) // 2 if nmo > norb else 0

    # Spin RDM in AO basis (core contributes zero net spin)
    full_spin_mo = np.zeros((nmo, nmo))
    full_spin_mo[ncore:ncore + norb, ncore:ncore + norb] = spin_rdm1
    spin_ao = mo_coeff @ full_spin_mo @ mo_coeff.T

    # Overlap matrix
    ovlp = mol.intor_symmetric("int1e_ovlp")

    # Mulliken population per AO
    pop_ao = np.diag(spin_ao @ ovlp)

    # Sum over AOs belonging to each atom
    natom = mol.natm
    spin_pop = np.zeros(natom)
    ao_labels = mol.ao_labels(fmt=False)
    for i, (atom_idx, *_) in enumerate(ao_labels):
        spin_pop[atom_idx] += pop_ao[i]

    return spin_pop
