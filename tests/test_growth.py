import numpy as np
import pytest
from pytest import approx
from scipy.interpolate import UnivariateSpline

from greapy.grea import GREA
from greapy.growth import (
    solve_growth,
    gamma,
    analytical_D,
    analytical_Dprime,
    analytical_gamma,
)


@pytest.fixture(scope="module")
def growth_inputs(cosmo):
    a = np.logspace(-2, 0, 200)
    H_arr = cosmo.Hubble(a)
    H_spline = UnivariateSpline(a, H_arr, s=0)
    return a, cosmo.Omega_m, 0.8, H_spline, H_spline.derivative()


@pytest.fixture(scope="module")
def growth_solution(cosmo, growth_inputs):
    a, Omega_m, sigma8, h_fn, hp_fn = growth_inputs
    return solve_growth(a, Omega_m, sigma8, h_fn, hp_fn)


def test_output_shapes(cosmo, growth_inputs, growth_solution):
    a = growth_inputs[0]
    d, f, fs8 = growth_solution
    assert d.shape == a.shape
    assert f.shape == a.shape
    assert fs8.shape == a.shape


def test_growth_factor_positive(growth_solution):
    d, _, _ = growth_solution
    assert np.all(d > 0)


def test_growth_factor_increasing(growth_solution):
    d, _, _ = growth_solution
    assert np.all(np.diff(d) > 0)


def test_growth_rate_positive(growth_solution):
    _, f, _ = growth_solution
    assert np.all(f > 0)


def test_fs8_definition_at_a1(growth_solution):
    sigma8 = 0.8
    d, f, fs8 = growth_solution
    assert fs8[-1] == approx(sigma8 * f[-1], rel=1e-4)


def test_reversed_input_consistent(cosmo, growth_inputs):
    a, Omega_m, sigma8, h_fn, hp_fn = growth_inputs
    d_fwd, f_fwd, _ = solve_growth(a, Omega_m, sigma8, h_fn, hp_fn)
    d_rev, f_rev, _ = solve_growth(a[::-1], Omega_m, sigma8, h_fn, hp_fn)
    assert d_fwd == approx(d_rev[::-1], rel=1e-5)
    assert f_fwd == approx(f_rev[::-1], rel=1e-5)


def test_gamma_raises_on_negative_f(cosmo):
    a = np.array([0.5, 1.0])
    f_bad = np.array([-0.1, 0.5])
    with pytest.raises(ValueError, match="f must be positive"):
        gamma(a, f_bad, cosmo)


def test_gamma_raises_on_small_omz():
    from unittest.mock import MagicMock
    mock_cosmo = MagicMock()
    mock_cosmo.Omega_m = 1e-10
    mock_cosmo.H0 = 67.0
    mock_cosmo.Hubble.return_value = np.array([67.0, 67.0])
    a = np.array([0.5, 1.0])
    f_ok = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="greater than 1e-5"):
        gamma(a, f_ok, mock_cosmo)


def test_analytical_D_normalized_at_a1():
    assert analytical_D(1.0, 0.3, normalize=True) == approx(1.0, rel=1e-8)


def test_analytical_D_positive_and_increasing():
    a = np.linspace(0.01, 1.0, 50)
    D = analytical_D(a, 0.3)
    assert np.all(D > 0)
    assert np.all(np.diff(D) > 0)


def test_analytical_gamma_near_gr_prediction():
    # GR prediction: gamma ~ 0.55 for LCDM at a=1
    assert analytical_gamma(1.0, 0.3) == approx(0.55, abs=0.03)
