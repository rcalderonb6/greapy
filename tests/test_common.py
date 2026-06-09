import os
import textwrap
from unittest.mock import MagicMock

import pytest
from pytest import approx

from greapy.common import (
    is_monotonic_increasing,
    get_bestfit,
    get_Rm1,
    extract_lnZ,
    get_dV_rs,
    get_F_AP,
    get_Mb_from_H0,
)


@pytest.mark.parametrize("arr,strict,expected", [
    ([1, 2, 3], False, True),
    ([1, 2, 2, 3], False, True),
    ([1, 2, 2, 3], True, False),
    ([3, 2, 1], False, False),
    ([1], False, True),
    ([], False, True),
])
def test_is_monotonic_increasing(arr, strict, expected):
    assert is_monotonic_increasing(arr, strict=strict) == expected


@pytest.fixture
def minimum_txt(tmp_path):
    content = textwrap.dedent("""\
        # chi2 h omega_cdm omega_b kappa
        10.5 0.6736 0.12 0.02237 3.55
    """)
    (tmp_path / "ds.minimum.txt").write_text(content)
    return tmp_path, "ds"


def test_get_bestfit_all_params(minimum_txt):
    path, dataset = minimum_txt
    result = get_bestfit(dataset, str(path))
    assert result == approx({"chi2": 10.5, "h": 0.6736, "omega_cdm": 0.12,
                             "omega_b": 0.02237, "kappa": 3.55})


def test_get_bestfit_subset(minimum_txt):
    path, dataset = minimum_txt
    result = get_bestfit(dataset, str(path), parameters=["h", "kappa"])
    assert set(result.keys()) == {"h", "kappa"}
    assert result["h"] == approx(0.6736)


def test_get_bestfit_no_parameters_no_unboundlocalerror(minimum_txt):
    path, dataset = minimum_txt
    result = get_bestfit(dataset, str(path))
    assert "h" in result


def test_get_bestfit_missing_file():
    with pytest.raises(FileNotFoundError):
        get_bestfit("nonexistent", "/tmp/doesnotexist123")


def test_get_Rm1_returns_dict():
    mock_chain = MagicMock()
    mock_chain.getGelmanRubin.return_value = 0.023
    result = get_Rm1({"dataset_A": mock_chain})
    assert isinstance(result, dict)
    assert result["dataset_A"] == approx(0.023)


def test_get_Rm1_multiple_chains():
    chains = {name: MagicMock() for name in ["A", "B", "C"]}
    for i, m in enumerate(chains.values()):
        m.getGelmanRubin.return_value = float(i) * 0.01
    result = get_Rm1(chains)
    assert set(result.keys()) == {"A", "B", "C"}


@pytest.fixture
def logz_file(tmp_path):
    content = "logZstd: 0.05\nlogZ: -42.137\n"
    (tmp_path / "run.logZ").write_text(content)
    return tmp_path, "run"


def test_extract_lnz_parses_value(logz_file):
    path, name = logz_file
    assert extract_lnZ(name, str(path)) == approx(-42.137)


def test_extract_lnz_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        extract_lnZ("nonexistent", "/tmp/doesnotexist123")


def test_get_Mb_from_H0_fiducial():
    assert get_Mb_from_H0(73.04) == approx(-19.253)


def test_get_Mb_from_H0_shifts_with_H0():
    mb1 = get_Mb_from_H0(70.0)
    mb2 = get_Mb_from_H0(75.0)
    assert mb1 < mb2


def test_get_dV_rs_positive(cosmo):
    result = get_dV_rs(0.5, cosmo)
    assert result > 0


def test_get_F_AP_positive(cosmo):
    result = get_F_AP(0.5, cosmo)
    assert result > 0
