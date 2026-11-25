# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from pyscf import ao2mo, fci, gto, mcscf, scf

H2 = "H 0 0 0; H 0 0 0.74"
LIH = "Li 0 0 0; H 0 0 1.6"


def run_mean_field(atom: str) -> scf.hf.RHF:
    return scf.RHF(gto.M(atom=atom, basis="sto-3g", verbose=0)).run()


def compute_exact_roots(casci: mcscf.casci.CASCI, nroots: int) -> np.ndarray:
    """Lowest ``nroots`` active-space energies of ``casci``, core energy included."""
    h1, ecore = casci.get_h1eff()
    eri = ao2mo.restore(1, casci.get_h2eff(), casci.ncas)
    energies, _ = fci.direct_spin1.kernel(
        h1, eri, casci.ncas, casci.nelecas, nroots=nroots
    )
    return np.atleast_1d(energies) + ecore
