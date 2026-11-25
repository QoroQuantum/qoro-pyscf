# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Adam with central finite-difference gradients, as a divi optimizer."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult

from divi.qprog.optimizers import Optimizer


class AdamOptimizer(Optimizer):
    """Adam on central finite-difference gradients of the cost.

    Each iteration evaluates the ``2 n`` shifted points in one batch, takes an
    Adam step and evaluates the new point; the best point seen is returned.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        grad_shift: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.learning_rate = learning_rate
        self.grad_shift = grad_shift
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    @property
    def n_param_sets(self) -> int:
        return 1

    @property
    def supports_checkpointing(self) -> bool:
        return False

    def optimize(
        self,
        cost_fn: Callable[[npt.NDArray[np.float64]], float | npt.NDArray[np.float64]],
        initial_params: npt.NDArray[np.float64] | None = None,
        callback_fn: Callable[[OptimizeResult], Any] | None = None,
        **kwargs,
    ) -> OptimizeResult:
        if initial_params is None:
            raise ValueError("AdamOptimizer requires initial_params.")
        max_iterations = self._resolve_max_iterations(kwargs)
        params = np.asarray(initial_params, dtype=np.float64).reshape(-1).copy()
        n_params = len(params)
        shifts = self.grad_shift * np.eye(n_params)
        first_moment = np.zeros(n_params)
        second_moment = np.zeros(n_params)
        best_loss, best_params = np.inf, params.copy()

        for iteration in range(1, max_iterations + 1):
            shifted = np.vstack([params + shifts, params - shifts])
            values = np.ravel(cost_fn(shifted))
            grad = (values[:n_params] - values[n_params:]) / (2 * self.grad_shift)

            first_moment = self.beta1 * first_moment + (1 - self.beta1) * grad
            second_moment = self.beta2 * second_moment + (1 - self.beta2) * grad**2
            m_hat = first_moment / (1 - self.beta1**iteration)
            v_hat = second_moment / (1 - self.beta2**iteration)
            params = params - self.learning_rate * m_hat / (
                np.sqrt(v_hat) + self.epsilon
            )

            loss = float(np.ravel(cost_fn(params[np.newaxis]))[0])
            if loss < best_loss:
                best_loss, best_params = loss, params.copy()
            if callback_fn is not None:
                callback_fn(
                    OptimizeResult(x=params[np.newaxis].copy(), fun=np.array([loss]))
                )

        return OptimizeResult(x=best_params, fun=best_loss, nit=max_iterations)

    def get_config(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "learning_rate": self.learning_rate,
            "grad_shift": self.grad_shift,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
        }

    def save_state(self, checkpoint_dir: Path | str) -> None:
        raise NotImplementedError("AdamOptimizer does not support checkpointing.")

    @classmethod
    def load_state(cls, checkpoint_dir: Path | str) -> Self:
        raise NotImplementedError("AdamOptimizer does not support checkpointing.")

    def reset(self) -> None:
        """Nothing to reset: the moments live only inside :meth:`optimize`."""
