import pytest
from pytest import approx

from greapy.approximations import zCMB, zdrag

OMEGA_M = 0.14237
OMEGA_B = 0.02237


def test_zcmb_in_physical_range():
    assert 900 < zCMB(OMEGA_M, OMEGA_B) < 1200


def test_zdrag_in_physical_range():
    assert 900 < zdrag(OMEGA_M, OMEGA_B) < 1100


def test_zdrag_less_than_zcmb():
    assert zdrag(OMEGA_M, OMEGA_B) < zCMB(OMEGA_M, OMEGA_B)


def test_zdrag_planck_fiducial():
    assert zdrag(OMEGA_M, OMEGA_B) == approx(1059, rel=0.01)


@pytest.mark.parametrize("omega_m", [0.12, 0.14, 0.16, 0.18])
def test_zdrag_changes_monotonically_with_omega_m(omega_m):
    z1 = zdrag(omega_m, OMEGA_B)
    z2 = zdrag(omega_m + 0.02, OMEGA_B)
    assert z1 != approx(z2)


def test_both_return_finite_floats():
    assert isinstance(zCMB(OMEGA_M, OMEGA_B), float)
    assert isinstance(zdrag(OMEGA_M, OMEGA_B), float)
