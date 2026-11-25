# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from qoro_pyscf import rotosolve_step, rotosolve_sweep


def test_step_finds_sinusoidal_minimum():
    params, energy = rotosolve_step(
        lambda p: 2.0 + 3.0 * np.cos(p[0] - 0.7), np.array([0.0]), 0
    )
    assert energy == pytest.approx(-1.0, abs=1e-10)
    assert params[0] == pytest.approx((0.7 + 2 * np.pi) % (2 * np.pi) - np.pi)


def test_step_sets_one_parameter_without_mutating_the_input():
    start = np.zeros(2)
    params, _ = rotosolve_step(
        lambda p: np.cos(p[0] - 1.0) + np.cos(p[1] - 2.0), start, 0
    )
    assert params[0] == pytest.approx(1.0 - np.pi)
    assert params[1] == 0.0
    np.testing.assert_array_equal(start, np.zeros(2))


def test_step_with_frequency_two_wraps_into_its_period():
    params, energy = rotosolve_step(
        lambda p: 2.0 + 3.0 * np.cos(2.0 * p[0] - 1.4), np.array([0.0]), 0, freq=2
    )
    assert energy == pytest.approx(-1.0, abs=1e-10)
    assert 2.0 + 3.0 * np.cos(2.0 * params[0] - 1.4) == pytest.approx(-1.0)
    assert -np.pi / 2 <= params[0] < np.pi / 2


def test_step_with_order_two_finds_the_two_harmonic_minimum():
    def cost(p):
        return np.cos(p[0] - 1.0) + 0.6 * np.cos(2.0 * p[0] + 0.3)

    params, energy = rotosolve_step(cost, np.array([0.4]), 0, order=2)
    grid = np.linspace(-np.pi, np.pi, 200001)
    reference = np.min(np.cos(grid - 1.0) + 0.6 * np.cos(2.0 * grid + 0.3))
    assert energy == pytest.approx(reference, abs=1e-9)
    assert cost(params) == pytest.approx(reference, abs=1e-9)
    assert -np.pi <= params[0] < np.pi


def test_sweep_converges():
    _, energy, history, converged = rotosolve_sweep(
        lambda p: np.cos(p[0] - 1.0) + np.cos(p[1] - 2.0),
        np.zeros(2),
        max_sweeps=10,
        tol=1e-12,
    )
    assert energy == pytest.approx(-2.0, abs=1e-10)
    assert converged
    assert len(history) <= 4


def test_sweep_uses_per_parameter_frequencies():
    def cost(p):
        return np.cos(p[0] - 1.0) + np.cos(2.0 * p[1] - 2.0)

    params, energy, _, _ = rotosolve_sweep(
        cost, np.zeros(2), max_sweeps=5, tol=1e-12, freqs=[1, 2]
    )
    assert energy == pytest.approx(-2.0, abs=1e-10)
    assert cost(params) == pytest.approx(-2.0, abs=1e-10)
