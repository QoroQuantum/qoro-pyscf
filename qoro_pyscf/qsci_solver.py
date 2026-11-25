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

A VQE trained by divi's :class:`~divi.qprog.algorithms.VQEFCISolver` is
sampled in the computational basis; the most probable determinants span a
subspace in which PySCF's selected CI diagonalises the Hamiltonian.

References
----------
[1] Kanno et al., arXiv:2302.11320 (QSCI)
[2] Robledo-Moreno et al., arXiv:2405.05068 (SQD at scale)
"""

import time
from typing import Any

import numpy as np
import numpy.typing as npt
from pyscf import ao2mo
from pyscf.fci import selected_ci

from divi.qprog.algorithms import VQEFCISolver


def _select_determinants(
    probabilities: dict[str, float],
    nelec: tuple[int, int],
    n_samples: int,
    probability_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alpha and beta determinant strings of the most probable bitstrings.

    Bitstring character ``k`` is qubit ``k``: alpha orbital ``s`` on qubit
    ``2 s``, beta on ``2 s + 1``. Bitstrings below ``probability_threshold`` or
    with the wrong electron count per spin are dropped, and the ``n_samples``
    most probable kept.

    Returns:
        Sorted unique alpha strings, beta strings (PySCF's integer encoding),
        and the probabilities of the kept bitstrings.
    """
    n_alpha, n_beta = nelec
    valid = sorted(
        (
            (prob, bits)
            for bits, prob in probabilities.items()
            if prob > probability_threshold
            and bits[0::2].count("1") == n_alpha
            and bits[1::2].count("1") == n_beta
        ),
        reverse=True,
    )[:n_samples]
    if not valid:
        raise ValueError(
            f"No bitstrings above probability {probability_threshold} with "
            f"(n_alpha, n_beta) = {nelec}. Lower the threshold or improve the VQE."
        )

    def to_string(occupations: str) -> int:
        return sum(1 << s for s, bit in enumerate(occupations) if bit == "1")

    alpha = sorted({to_string(bits[0::2]) for _, bits in valid})
    beta = sorted({to_string(bits[1::2]) for _, bits in valid})
    return (
        np.array(alpha, dtype=np.int64),
        np.array(beta, dtype=np.int64),
        np.array([prob for prob, _ in valid]),
    )


class QSCISolver:
    """PySCF FCI-solver running Quantum-Selected Configuration Interaction.

    Args:
        inner_solver: Trains the VQE whose samples select the determinants.
        n_samples: Most probable bitstrings kept.
        probability_threshold: Bitstrings at or below it are dropped.

    Attributes:
        vqe_energy: Energy of the inner VQE.
        qsci_energy: Selected-CI energy, ``ecore`` included.
        n_determinants: Dimension of the selected subspace.
        qsci_time: Wall-clock seconds of the last :meth:`kernel`.
        converged: Whether the inner VQE's optimiser reported success.
    """

    nroots: int = 1

    def __init__(
        self,
        inner_solver: VQEFCISolver,
        n_samples: int = 500,
        probability_threshold: float = 1e-8,
    ) -> None:
        self.inner_solver = inner_solver
        self.n_samples = n_samples
        self.probability_threshold = probability_threshold
        self.converged = False
        self.vqe_energy = 0.0
        self.qsci_energy = 0.0
        self.n_determinants = 0
        self.qsci_time = 0.0
        self._norb = 0
        self._nelec = (0, 0)
        self._ci_vector: Any = None

    def kernel(
        self,
        h1e: npt.ArrayLike,
        eri: npt.ArrayLike,
        norb: int,
        nelec: int | tuple[int, int],
        ci0: Any = None,
        ecore: float = 0.0,
        **kwargs: Any,
    ) -> tuple[float, "QSCISolver"]:
        """Train the VQE, sample it, and diagonalise in the selected subspace.

        Returns:
            The QSCI energy with ``ecore``, and this solver as PySCF's CI vector.
        """
        start = time.perf_counter()
        if isinstance(nelec, (int, np.integer)):
            nelec = (nelec - nelec // 2, nelec // 2)
        nelec = (int(nelec[0]), int(nelec[1]))
        self._norb, self._nelec = norb, nelec

        self.vqe_energy, vqe = self.inner_solver.kernel(
            h1e, eri, norb, nelec, ci0, ecore, **kwargs
        )
        probabilities = vqe.sample_solution().best_probs[0]
        alpha, beta, _ = _select_determinants(
            probabilities, nelec, self.n_samples, self.probability_threshold
        )

        energy, self._ci_vector = selected_ci.kernel_fixed_space(
            selected_ci.SelectedCI(),
            np.asarray(h1e),
            ao2mo.restore(1, np.asarray(eri), norb),
            norb,
            nelec,
            (alpha, beta),
        )
        self.qsci_energy = float(energy) + ecore
        self.n_determinants = len(alpha) * len(beta)
        self.converged = self.inner_solver.converged
        self.qsci_time = time.perf_counter() - start
        return self.qsci_energy, self

    def approx_kernel(
        self,
        h1e: npt.ArrayLike,
        eri: npt.ArrayLike,
        norb: int,
        nelec: int | tuple[int, int],
        ci0: Any = None,
        ecore: float = 0.0,
        **kwargs: Any,
    ) -> tuple[float, "QSCISolver"]:
        """The last :meth:`kernel` result, for CASSCF's approximate CI steps.

        Runs :meth:`kernel` when there is no last result.
        """
        if self._ci_vector is None:
            return self.kernel(h1e, eri, norb, nelec, ci0, ecore, **kwargs)
        return self.qsci_energy, self

    def make_rdm1(self, ci: Any, norb: int, nelec: int | tuple[int, int]) -> np.ndarray:
        """Spin-summed 1-RDM of the QSCI eigenvector."""
        return selected_ci.make_rdm1(self._ci_vector, self._norb, self._nelec)

    def make_rdm1s(
        self, ci: Any, norb: int, nelec: int | tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Alpha and beta 1-RDMs of the QSCI eigenvector."""
        return selected_ci.make_rdm1s(self._ci_vector, self._norb, self._nelec)

    def make_rdm12(
        self, ci: Any, norb: int, nelec: int | tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Spin-summed 1- and 2-RDMs of the QSCI eigenvector."""
        return selected_ci.make_rdm12(self._ci_vector, self._norb, self._nelec)

    def make_rdm12s(self, ci: Any, norb: int, nelec: int | tuple[int, int]) -> tuple:
        """Spin-resolved 1- and 2-RDMs of the QSCI eigenvector."""
        return selected_ci.make_rdm12s(self._ci_vector, self._norb, self._nelec)

    def spin_square(
        self, ci: Any, norb: int, nelec: int | tuple[int, int]
    ) -> tuple[float, float]:
        """``<S^2>`` and ``2S + 1`` of the QSCI eigenvector."""
        ss, multiplicity = selected_ci.spin_square(
            self._ci_vector, self._norb, self._nelec
        )
        return float(ss), float(multiplicity)
