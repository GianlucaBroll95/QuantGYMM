import pandas as pd
import pytest

from ..utils import accrual_factor


@pytest.mark.parametrize("dcc, start, end, expected", [
    # ACT/360
    ("ACT/360", "2023-01-01", "2023-07-01", 181 / 360),
    # ACT/365
    ("ACT/365", "2023-01-01", "2023-07-01", 181 / 365),
    # 30/360 (US bond basis) — day-31 edge case, conditional on start day
    ("30/360", "2023-01-31", "2023-03-31", 60 / 360),
    ("30/360", "2023-01-29", "2023-03-31", 62 / 360),  # start not 30/31 -> end day kept as 31
    # 30E/360 (Eurobond) — end-day-31 always forced to 30, independent of start
    ("30E/360", "2023-01-31", "2023-03-31", 60 / 360),
    ("30E/360", "2023-01-29", "2023-03-31", 61 / 360),  # diverges from 30/360 here
    # NL/365 — leap day excluded from the count
    ("NL/365", "2024-02-28", "2024-03-01", 1 / 365),  # spans Feb 29 2024, excluded
    ("NL/365", "2023-02-28", "2023-03-01", 1 / 365),  # non-leap year, same result
])
def test_accrual_factor_simple(dcc, start, end, expected):
    result = accrual_factor(dcc, pd.Timestamp(start), pd.Timestamp(end))
    assert result[0] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("start, end, expected", [
    # ACT/ACT ISDA — period entirely within a non-leap year
    ("2023-03-15", "2023-09-15", 184 / 365),
    # ACT/ACT ISDA — period spans the 2024 leap year, split at year boundary
    ("2023-09-15", "2024-09-15", None),  # checked separately below (split calc)
])
def test_act_act_isda(start, end, expected):
    result = accrual_factor("ACT/ACT", pd.Timestamp(start), pd.Timestamp(end))[0]
    if expected is not None:
        assert result == pytest.approx(expected, abs=1e-6)
    else:
        # manual split: days in 2023 (non-leap, /365) + days in 2024 (leap, /366)
        days_2023 = (pd.Timestamp("2024-01-01") - pd.Timestamp(start)).days
        days_2024 = (pd.Timestamp(end) - pd.Timestamp("2024-01-01")).days
        expected_split = days_2023 / 365 + days_2024 / 366
        assert result == pytest.approx(expected_split, abs=1e-6)


@pytest.mark.parametrize("start, end, frequency, expected", [
    ("2023-03-15", "2023-09-15", None, 0.5),    # inferred semiannual
    ("2023-03-15", "2024-03-15", None, 1.0),    # inferred annual
    ("2023-03-15", "2023-06-15", None, 0.25),   # inferred quarterly
    ("2023-03-15", "2023-09-15", 4, 0.25),      # explicit override beats inference
])
def test_act_act_icma(start, end, frequency, expected):
    kwargs = {} if frequency is None else {"frequency": frequency}
    result = accrual_factor("ACT/ACT ICMA", pd.Timestamp(start), pd.Timestamp(end), **kwargs)
    assert result[0] == pytest.approx(expected, abs=1e-6)


def test_accrual_factor_list_of_dates():
    dates = pd.DatetimeIndex(["2023-01-01", "2023-07-01", "2024-01-01"])
    result = accrual_factor("ACT/365", dates)
    assert len(result) == 2
    assert result[0] == pytest.approx(181 / 365, abs=1e-6)
    assert result[1] == pytest.approx(184 / 365, abs=1e-6)


def test_accrual_factor_invalid_dcc_raises():
    with pytest.raises(ValueError):
        accrual_factor("INVALID/DCC", pd.Timestamp("2023-01-01"), pd.Timestamp("2023-07-01"))


def test_accrual_factor_invalid_dimension_raises():
    with pytest.raises(ValueError):
        accrual_factor("ACT/365", pd.Timestamp("2023-01-01"), pd.Timestamp("2023-07-01"), pd.Timestamp("2024-01-01"))
