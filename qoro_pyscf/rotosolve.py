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
Rotosolve: analytical single-parameter optimiser for variational circuits.

For any parameterised gate whose generator has eigenvalues ±½ (e.g. Ry, Rz,
or a Pauli rotation exp(-iθP/2)), the energy landscape as a function of
that single parameter is exactly sinusoidal:

    E(θ) = a + b·cos(θ) + c·sin(θ)

This allows finding the exact minimum with only **3 circuit evaluations**
per parameter. Generators with more eigenvalues give higher harmonics, which
``order`` covers with ``2 order + 1`` evaluations.

Reference: Ostaszewski et al., Quantum 5, 391 (2021)
           "Structure optimization for parameterized quantum circuits"
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np
from scipy.optimize import minimize_scalar


def rotosolve_step(
    cost_fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    index: int,
    freq: int = 1,
    order: int = 1,
) -> tuple[np.ndarray, float]:
    """
    Optimise a single parameter of a variational circuit.

    The energy along the parameter is a trigonometric polynomial
    ``a + sum_{k <= order} b_k cos(k freq θ) + c_k sin(k freq θ)``. It is
    sampled at ``2 order + 1`` equally spaced points over one period, fitted
    exactly, and minimised: in closed form for ``order=1``, numerically
    otherwise.

    Parameters
    ----------
    cost_fn : callable
        Cost function mapping parameter vector → scalar energy.
    params : np.ndarray
        Current parameter vector; a copy is returned.
    index : int
        Index of the parameter to optimise.
    freq : int
        Base frequency of the energy landscape.  Use ``freq=1``
        when the gate uses ``ry(θ)`` (period 2π), ``freq=2`` when the
        gate uses ``ry(2θ)`` (period π).
    order : int
        Highest harmonic of ``freq`` in the landscape, e.g. 2 for a Givens
        rotation, whose generator has eigenvalues -1, 0 and 1.

    Returns
    -------
    params : np.ndarray
        Updated parameter vector with ``params[index]`` set to the optimum.
    energy : float
        Energy at the optimal parameter value.
    """
    params = params.copy()
    theta_0 = params[index]
    n_points = 2 * order + 1
    energies = []
    for point in range(n_points):
        params[index] = theta_0 + 2.0 * np.pi * point / (n_points * freq)
        energies.append(cost_fn(params))
    # E(θ₀ + u/freq) = Re(sum_k weights[k] e^{iku}), u = freq·s
    weights = np.fft.rfft(energies) / n_points
    weights[1:] *= 2

    def model(u: float) -> float:
        return float(np.real(weights @ np.exp(1j * np.arange(order + 1) * u)))

    if order == 1:
        u_opt = np.pi - np.angle(weights[1])
    else:
        grid = np.linspace(0.0, 2.0 * np.pi, 64 * order, endpoint=False)
        u_grid = grid[np.argmin([model(u) for u in grid])]
        step = grid[1]
        u_opt = minimize_scalar(
            model,
            bounds=(u_grid - step, u_grid + step),
            method="bounded",
            options={"xatol": 1e-10},
        ).x

    period = 2.0 * np.pi / freq
    params[index] = (theta_0 + u_opt / freq + period / 2) % period - period / 2
    return params, model(u_opt)


def rotosolve_sweep(
    cost_fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    max_sweeps: int = 50,
    tol: float = 1e-8,
    verbose: bool = False,
    callback: Callable[[int, float, np.ndarray], None] | None = None,
    freqs: Sequence[int] | None = None,
    orders: Sequence[int] | None = None,
) -> tuple[np.ndarray, float, list[float], bool]:
    """
    Optimise all parameters by sweeping Rotosolve across them.

    Each sweep visits every parameter once and analytically sets it to
    its optimal value (3 cost evaluations per parameter). Sweeps repeat
    until the energy improvement falls below ``tol`` or ``max_sweeps``
    is reached.

    Cost per sweep: ``3 × n_params`` circuit evaluations.

    Parameters
    ----------
    cost_fn : callable
        Cost function mapping parameter vector → scalar energy.
    params : np.ndarray
        Initial parameter vector.
    max_sweeps : int
        Maximum number of full sweeps over all parameters.
    tol : float
        Convergence tolerance on energy change between sweeps.
    verbose : bool
        Print progress after each sweep.
    callback : callable or None
        Called after each sweep with ``(sweep, energy, params)``.
    freqs, orders : sequence of int or None
        Per-parameter frequency and order, as in :func:`rotosolve_step`.
        Default: all 1.

    Returns
    -------
    params : np.ndarray
        Optimised parameter vector.
    energy : float
        Final energy.
    energy_history : list[float]
        Energy after each sweep.
    converged : bool
        Whether convergence tolerance was reached.
    """
    params = params.copy()
    n_params = len(params)
    if freqs is None:
        freqs = [1] * n_params
    if orders is None:
        orders = [1] * n_params
    energy_history: list[float] = []
    converged = False

    # Initial energy
    prev_energy = cost_fn(params)
    energy_history.append(prev_energy)
    t_start = time.perf_counter()

    for sweep in range(1, max_sweeps + 1):
        for j in range(n_params):
            params, energy = rotosolve_step(
                cost_fn, params, j, freq=freqs[j], order=orders[j]
            )

        energy_history.append(energy)

        if verbose:
            elapsed = time.perf_counter() - t_start
            delta = energy - prev_energy
            print(
                f"    sweep {sweep:4d}  E = {energy:+.10f}  Ha"
                f"  ΔE = {delta:+.2e}  [{elapsed:.1f}s]",
                flush=True,
            )

        if callback is not None:
            callback(sweep, energy, params)

        if abs(energy - prev_energy) < tol:
            converged = True
            if verbose:
                print(
                    f"    Rotosolve converged: |ΔE| = {abs(energy - prev_energy):.2e}"
                    f" < {tol:.2e}",
                    flush=True,
                )
            break

        prev_energy = energy

    return params, energy, energy_history, converged
