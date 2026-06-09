import numpy as np
import pytest
from greapy.grea import GREA


@pytest.fixture(scope="module")
def cosmo():
    return GREA(h=0.6736, omega_cdm=0.12, omega_b=0.02237, kappa=3.55)


@pytest.fixture
def scale_factors():
    return np.logspace(-2, 0, 100)


@pytest.fixture
def fiducial_params():
    return {"omega_m": 0.14237, "omega_b": 0.02237}
