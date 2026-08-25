"""
Tests for day count conventions in utils.py.

Covers:
- correctness on simple cases
- scalar vs vector consistency
- edge cases (month-end, leap years)
- invalid input handling
"""

import numpy as np
import pandas as pd
import pytest

from QuantGYMM.utils import (
    act360, act365, thirty360, thirty_e_360,
    act_act, nl365, act_act_icma
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

D1 = pd.Timestamp("2023-01-01")
D2 = pd.Timestamp("2023-02-01")
D3 = pd.Timestamp("2023-03-01")

DATES_SEQ = [D1, D2, D3]


# ---------------------------------------------------------------------------
# Helper checks
# ---------------------------------------------------------------------------

def assert_close(x, y, tol=1e-10):
    assert abs(x - y) < tol


# ---------------------------------------------------------------------------
# ACT/360 & ACT/365
# ---------------------------------------------------------------------------

class TestAct:

    def test_act360_scalar(self):
        af = act360(D1, D2)[0]
        assert_close(af, 31 / 360)

    def test_act365_scalar(self):
        af = act365(D1, D2)[0]
        assert_close(af, 31 / 365)

    def test_act360_vector(self):
        af = act360(DATES_SEQ)
        assert len(af) == 2

    def test_scalar_vs_vector_consistency(self):
        scalar = act360(D1, D2)[0]
        vector = act360([D1, D2])[0]
        assert_close(scalar, vector)


# ---------------------------------------------------------------------------
# 30/360
# ---------------------------------------------------------------------------

class TestThirty360:

    def test_basic_month(self):
        af = thirty360(D1, D2)[0]
        assert_close(af, 30 / 360)

    def test_end_month_31_not_adjusted(self):
        d_start = pd.Timestamp("2023-01-20")
        d_end   = pd.Timestamp("2023-01-31")

        af = thirty360(d_start, d_end)[0]

        expected = (31 - 20) / 360
        assert_close(af, expected)

    def test_end_month_31_adjusted(self):
        d_start = pd.Timestamp("2023-01-30")
        d_end   = pd.Timestamp("2023-01-31")

        af = thirty360(d_start, d_end)[0]

        assert_close(af, 0.0)

    def test_vector_length(self):
        af = thirty360(DATES_SEQ)
        assert len(af) == 2


# ---------------------------------------------------------------------------
# 30E/360
# ---------------------------------------------------------------------------

class TestThirtyE360:

    def test_basic(self):
        af = thirty_e_360(D1, D2)[0]
        assert_close(af, 30 / 360)

    def test_31_always_adjusted(self):
        d_start = pd.Timestamp("2023-01-20")
        d_end   = pd.Timestamp("2023-01-31")

        af = thirty_e_360(d_start, d_end)[0]

        expected = (30 - 20) / 360
        assert_close(af, expected)


# ---------------------------------------------------------------------------
# ACT/ACT ISDA
# ---------------------------------------------------------------------------

class TestActAct:

    def test_same_date(self):
        af = act_act(D1, D1)[0]
        assert af == 0.0

    def test_one_year(self):
        d_start = pd.Timestamp("2022-01-01")
        d_end   = pd.Timestamp("2023-01-01")

        af = act_act(d_start, d_end)[0]
        assert_close(af, 1.0)

    def test_leap_year(self):
        d_start = pd.Timestamp("2020-02-28")
        d_end   = pd.Timestamp("2020-03-01")

        af = act_act(d_start, d_end)[0]
        assert af > 0


# ---------------------------------------------------------------------------
# NL/365
# ---------------------------------------------------------------------------

class TestNL365:

    def test_basic(self):
        af = nl365(D1, D2)[0]
        assert_close(af, 31 / 365)

    def test_excludes_feb29(self):
        d_start = pd.Timestamp("2020-02-28")
        d_end   = pd.Timestamp("2020-03-01")

        af = nl365(d_start, d_end)[0]

        # should be 2 days - 1 leap day = 1
        assert_close(af, 1 / 365)


# ---------------------------------------------------------------------------
# ACT/ACT ICMA
# ---------------------------------------------------------------------------

class TestActActICMA:

    def test_fixed_frequency(self):
        af = act_act_icma(D1, D2, frequency=12)[0]
        assert_close(af, 1 / 12)

    def test_infer_frequency(self):
        d_start = pd.Timestamp("2023-01-01")
        d_end   = pd.Timestamp("2023-07-01")

        af = act_act_icma(d_start, d_end)[0]

        # ~6 months → freq ≈ 2
        assert_close(af, 1 / 2)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_invalid_dimension(self):
        with pytest.raises(ValueError):
            act360(D1, D2, D3)

    def test_mismatched_vectors(self):
        with pytest.raises(ValueError):
            act360([D1, D2], [D2])

    def test_vector_scalar_not_allowed(self):
        with pytest.raises(ValueError):
            thirty360([D1, D2], D3)


# ---------------------------------------------------------------------------
# is_bd — TARGET calendar
# ---------------------------------------------------------------------------

from QuantGYMM.utils import is_bd, business_adjustment, modified_following, following, preceding


class TestIsBusinessDay:

    @pytest.mark.parametrize("date", ["2023-03-13", "2023-03-14", "2023-03-17"])
    def test_weekday_is_bd(self, date):
        assert is_bd(pd.Timestamp(date)) is True

    @pytest.mark.parametrize("date", ["2023-03-18", "2023-03-19"])
    def test_weekend_is_not_bd(self, date):
        assert is_bd(pd.Timestamp(date)) is False

    @pytest.mark.parametrize("date", [
        "2023-01-01",   # New Year
        "2023-04-07",   # Good Friday
        "2023-04-10",   # Easter Monday
        "2023-05-01",   # Labour Day
        "2023-12-25",   # Christmas
        "2023-12-26",   # St. Stephen
    ])
    def test_target_holiday_is_not_bd(self, date):
        assert is_bd(pd.Timestamp(date)) is False


# ---------------------------------------------------------------------------
# Convention functions — single date
# ---------------------------------------------------------------------------

class TestConventionSingle:

    def test_following_weekend_goes_to_monday(self):
        assert following(pd.Timestamp("2023-03-18")) == pd.Timestamp("2023-03-20")

    def test_following_bd_unchanged(self):
        assert following(pd.Timestamp("2023-03-13")) == pd.Timestamp("2023-03-13")

    def test_preceding_weekend_goes_to_friday(self):
        assert preceding(pd.Timestamp("2023-03-18")) == pd.Timestamp("2023-03-17")

    def test_modified_following_same_month_goes_forward(self):
        assert modified_following(pd.Timestamp("2023-03-18")) == pd.Timestamp("2023-03-20")

    def test_modified_following_crosses_month_goes_backward(self):
        # 2023-09-30 is Saturday; next BD is 2023-10-02 (October) → go back to 2023-09-29
        result = modified_following(pd.Timestamp("2023-09-30"))
        assert result.month == 9

    def test_modified_following_bd_unchanged(self):
        assert modified_following(pd.Timestamp("2023-03-13")) == pd.Timestamp("2023-03-13")


# ---------------------------------------------------------------------------
# business_adjustment — bulk dispatch
# ---------------------------------------------------------------------------

class TestBusinessAdjustment:

    def test_single_date_following(self):
        result = business_adjustment("following", pd.Timestamp("2023-03-18"))
        assert result == pd.Timestamp("2023-03-20")

    def test_list_of_dates_following(self):
        dates  = [pd.Timestamp("2023-03-18"), pd.Timestamp("2023-03-19")]
        result = business_adjustment("following", dates)
        assert len(result) == 2
        assert result[0] == pd.Timestamp("2023-03-20")

    def test_single_date_preceding(self):
        result = business_adjustment("preceding", pd.Timestamp("2023-03-19"))
        assert result == pd.Timestamp("2023-03-17")

    def test_single_date_modified_following(self):
        result = business_adjustment("modified_following", pd.Timestamp("2023-03-18"))
        assert result == pd.Timestamp("2023-03-20")

    def test_single_date_bimonthly(self):
        result = business_adjustment("modified_following_bimonthly", pd.Timestamp("2023-03-18"))
        assert isinstance(result, pd.Timestamp)

    def test_invalid_convention_raises(self):
        with pytest.raises(ValueError):
            business_adjustment("no_such_convention", pd.Timestamp("2023-03-18"))

    def test_list_preserves_length(self):
        dates  = pd.date_range("2023-01-01", "2023-01-10")
        result = business_adjustment("following", dates)
        assert len(result) == len(dates)

    def test_bd_unchanged_by_all_conventions(self):
        bd = pd.Timestamp("2023-03-13")  # Monday, non-holiday
        for conv in ("following", "preceding", "modified_following", "modified_following_bimonthly"):
            assert business_adjustment(conv, bd) == bd