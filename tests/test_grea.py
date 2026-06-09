import numpy as np
import pytest
from pytest import approx

from greapy.grea import GREA
from greapy.common import C_KMS


def test_hubble_today_equals_H0(cosmo):
    # In GREA H(z=0) is a model output (not simply 100*h) — verify it equals H0
    assert cosmo.H(0) == approx(cosmo.H0, rel=1e-10)
    assert cosmo.H0 > 0


def test_hubble_positive_at_multiple_redshifts(cosmo):
    for z in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]:
        assert cosmo.H(z) > 0


def test_hubble_array_input(cosmo):
    z_arr = np.array([0.1, 0.5, 1.0])
    result = cosmo.H(z_arr)
    assert result.shape == (3,)


def test_hubble_units_conversion(cosmo):
    a = 0.5
    h_kms = cosmo.Hubble(a, units="km/s/Mpc")
    h_inv = cosmo.Hubble(a, units="1/Mpc")
    assert h_inv == approx(h_kms / C_KMS, rel=1e-10)


def test_comoving_distance_zero_at_zero(cosmo):
    assert cosmo.comoving_distance(0.0) == approx(0.0, abs=1e-5)


def test_luminosity_distance_relation(cosmo):
    for z in [0.1, 0.5, 1.0, 2.0]:
        assert cosmo.luminosity_distance(z) == approx(
            cosmo.comoving_distance(z) * (1 + z), rel=1e-6
        )


def test_angular_diameter_distance_relation(cosmo):
    for z in [0.1, 0.5, 1.0, 2.0]:
        assert cosmo.angular_diameter_distance(z) == approx(
            cosmo.comoving_distance(z) / (1 + z), rel=1e-6
        )


def test_comoving_distance_increases_with_redshift(cosmo):
    z = np.array([0.1, 0.5, 1.0, 2.0, 3.0])
    dc = np.array([cosmo.comoving_distance(zi) for zi in z])
    assert np.all(np.diff(dc) > 0)


def test_tau_strictly_increasing(cosmo, scale_factors):
    tau_vals = cosmo.tau(scale_factors)
    assert np.all(np.diff(tau_vals) > 0)


def test_tau_spline_cached(cosmo):
    _ = cosmo.tau(0.5)
    spline_id = id(cosmo.tau_spline)
    _ = cosmo.tau(0.8)
    assert id(cosmo.tau_spline) == spline_id


def test_require_update_invalidates_cache(cosmo):
    _ = cosmo.tau(0.5)
    old_id = id(cosmo.tau_spline)
    cosmo._require_update = True
    _ = cosmo.tau(0.5)
    assert id(cosmo.tau_spline) != old_id


def test_w_callable_on_fresh_instance():
    fresh = GREA()
    assert fresh.tau_spline is None
    result = fresh.w(1.0)
    assert np.isfinite(result)


def test_wa_callable_on_fresh_instance():
    fresh = GREA()
    assert fresh.tau_spline is None
    assert np.isfinite(fresh.wa)


def test_w0_physically_plausible(cosmo):
    assert -2.0 < cosmo.w0 < 0.0


def test_thetastar_close_to_planck(cosmo):
    assert cosmo.thetastar == approx(0.0104, rel=0.02)


def test_aeq_small_and_positive(cosmo):
    assert 1e-5 < cosmo.aeq < 1e-2


def test_density_parameters_sum_below_unity(cosmo):
    assert cosmo.Omega_bc + cosmo.Omega_g + cosmo.Omega_nu < 1.0


def test_rdrag_in_physical_range(cosmo):
    assert 100 < cosmo.rdrag < 200


def test_rdrag_greater_than_rs_rec(cosmo):
    # z_drag < z_rec in GREA → more acoustic travel time → rdrag > rs_rec
    assert cosmo.rdrag > cosmo.rs_rec


def test_alpha_positive(cosmo):
    assert cosmo.alpha > 0


def test_sound_speed_decreases_with_a(cosmo):
    a = np.array([0.001, 0.01, 0.1, 1.0])
    cs = cosmo.cs(a)
    assert np.all(np.diff(cs) < 0)
