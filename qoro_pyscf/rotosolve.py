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
per parameter, rather than hundreds of iterative optimizer steps.

Reference: Ostaszewski et al., Quantum 5, 391 (2021)
           "Structure optimization for parameterized quantum circuits"
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def rotosolve_step(
    cost_fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    index: int,
    freq: int = 1,
) -> tuple[np.ndarray, float]:
    """
    Analytically optimise a single parameter of a variational circuit.

    Evaluates the cost function at three points and fits the sinusoidal
    model to find the exact minimum.

    Parameters
    ----------
    cost_fn : callable
        Cost function mapping parameter vector → scalar energy.
    params : np.ndarray
        Current parameter vector (will be modified in-place).
    index : int
        Index of the parameter to optimise.
    freq : int
        Frequency multiplier for the energy landscape.  Use ``freq=1``
        when the gate uses ``ry(θ)`` (period 2π), ``freq=2`` when the
        gate uses ``ry(2θ)`` (period π).

    Returns
    -------
    params : np.ndarray
        Updated parameter vector with ``params[index]`` set to the optimum.
    energy : float
        Energy at the optimal parameter value.
    """
    params = params.copy()
    theta_0 = params[index]

    # The energy landscape is E(θ) = a + b·cos(freq·θ) + c·sin(freq·θ)
    # with period 2π/freq.  We sample at three points separated by
    # 2π/(3·freq) in θ-space so that the argument of cos/sin is
    # separated by 2π/3 — the standard 3-point DFT spacing.
    ds = 2.0 * np.pi / (3.0 * freq)  # spacing in θ-space
    shifts = [0.0, ds, 2.0 * ds]

    energies = []
    for s in shifts:
        params[index] = theta_0 + s
        energies.append(cost_fn(params))

    e0, e1, e2 = energies

    # Fit E(θ₀ + s) = a + b·cos(freq·s) + c·sin(freq·s)
    # With u = freq·s, the three evaluation points are u = 0, 2π/3, 4π/3,
    # giving the standard system:
    a = (e0 + e1 + e2) / 3.0
    b = (2.0 * e0 - e1 - e2) / 3.0
    c = (e1 - e2) / np.sqrt(3.0)

    # Minimum of a + R·cos(freq·s - φ) is at freq·s* = φ + π
    phi = np.arctan2(c, b)
    s_opt = (phi + np.pi) / freq

    # Set the optimal parameter value
    period = 2.0 * np.pi / freq
    params[index] = theta_0 + s_opt
    # Wrap to [-period/2, period/2]
    params[index] = (params[index] + period / 2) % period - period / 2

    # Compute the minimum energy analytically
    R = np.sqrt(b**2 + c**2)
    energy = a - R

    return params, energy


def rotosolve_sweep(
    cost_fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    max_sweeps: int = 50,
    tol: float = 1e-8,
    verbose: bool = False,
    callback: Callable[[int, float, np.ndarray], None] | None = None,
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
    import time

    params = params.copy()
    n_params = len(params)
    energy_history: list[float] = []
    converged = False

    # Initial energy
    prev_energy = cost_fn(params)
    energy_history.append(prev_energy)
    t_start = time.perf_counter()

    for sweep in range(1, max_sweeps + 1):
        for j in range(n_params):
            params, energy = rotosolve_step(cost_fn, params, j)

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
