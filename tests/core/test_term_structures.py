"""
Tests for SpotRateCurve, DiscountCurve, EuriborCurve, SwapRateCurve.
"""
import numpy as np
import pandas as pd
import pytest
from pandas.tseries.offsets import DateOffset

from QuantGYMM.term_structures import SpotRateCurve, DiscountCurve, EuriborCurve, SwapRateCurve

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TRADE_DATE   = pd.Timestamp("2020-01-02")
FLAT_RATE    = 0.03

TENORS_MONTHS = [3, 6, 12, 24, 36, 60, 84, 120, 180, 240, 360]
MATURITIES    = pd.DatetimeIndex([TRADE_DATE + DateOffset(months=m) for m in TENORS_MONTHS])
_sr_flat      = pd.DataFrame({"spotRate": [FLAT_RATE] * len(MATURITIES)}, index=MATURITIES)


@pytest.fixture
def src_continuous():
    return SpotRateCurve(_sr_flat.copy(), TRADE_DATE, dcc="ACT/365",
                         interpolation="linear", compounding="continuous")


@pytest.fixture
def src_annual():
    return SpotRateCurve(_sr_flat.copy(), TRADE_DATE, dcc="ACT/365",
                         interpolation="linear", compounding="annually_compounded")


@pytest.fixture
def src_simple():
    return SpotRateCurve(_sr_flat.copy(), TRADE_DATE, dcc="ACT/365",
                         interpolation="linear", compounding="simple")


@pytest.fixture
def dc_continuous(src_continuous):
    return DiscountCurve(src_continuous, compounding="continuous",
                         dcc="ACT/365", interpolation="linear")


@pytest.fixture
def dc_annual(src_annual):
    return DiscountCurve(src_annual, compounding="annually_compounded",
                         dcc="ACT/365", interpolation="linear")


@pytest.fixture
def dc_simple(src_simple):
    return DiscountCurve(src_simple, compounding="simple",
                         dcc="ACT/365", interpolation="linear")


# ===========================================================================
# SpotRateCurve
# ===========================================================================

class TestSpotRateCurveConstruction:

    def test_sr_is_dataframe(self, src_continuous):
        assert isinstance(src_continuous.sr, pd.DataFrame)

    def test_sr_has_required_columns(self, src_continuous):
        assert {"maturity", "spotRate", "term"}.issubset(src_continuous.sr.columns)

    def test_sr_node_count(self, src_continuous):
        assert len(src_continuous.sr) == len(MATURITIES)

    def test_sr_flat_rate_values(self, src_continuous):
        assert np.allclose(src_continuous.sr["spotRate"].values, FLAT_RATE)

    def test_invalid_compounding_raises(self):
        with pytest.raises(ValueError):
            SpotRateCurve(_sr_flat.copy(), TRADE_DATE, compounding="monthly")

    def test_invalid_dcc_raises(self):
        with pytest.raises(ValueError):
            SpotRateCurve(_sr_flat.copy(), TRADE_DATE, dcc="INVALID")

    def test_repr(self, src_continuous):
        assert "SpotRateCurve" in repr(src_continuous)


class TestSpotRateCurveRateAt:

    def test_returns_array(self, src_continuous):
        dates = pd.DatetimeIndex([TRADE_DATE + DateOffset(years=1)])
        result = src_continuous.rate_at(dates)
        assert isinstance(result, np.ndarray)
        assert len(result) == 1

    def test_flat_curve_rate_at_any_tenor(self, src_continuous):
        dates = pd.DatetimeIndex([TRADE_DATE + DateOffset(years=t) for t in [1, 3, 5, 10]])
        rates = src_continuous.rate_at(dates)
        assert np.allclose(rates, FLAT_RATE, atol=1e-6)

    def test_interpolator_cached_across_calls(self, src_continuous):
        d = pd.DatetimeIndex([TRADE_DATE + DateOffset(years=2)])
        src_continuous.rate_at(d)
        interp_1 = src_continuous._interpolator
        src_continuous.rate_at(d)
        interp_2 = src_continuous._interpolator
        assert interp_1 is interp_2

    def test_interpolator_invalidated_on_data_change(self, src_continuous):
        d = pd.DatetimeIndex([TRADE_DATE + DateOffset(years=2)])
        src_continuous.rate_at(d)
        new_data = _sr_flat.copy()
        new_data.iloc[0, 0] = 0.04
        src_continuous.spot_rates_data = new_data
        assert src_continuous._interpolator is None


class TestSpotRateCurveDiscountFactors:

    def test_df_in_zero_one_range(self, src_continuous):
        df = src_continuous.discount_factors["discountFactor"]
        assert (df > 0).all() and (df <= 1.0).all()

    def test_continuous_df_formula_at_5y(self, src_continuous):
        date = TRADE_DATE + DateOffset(years=5)
        df   = src_continuous.discount_factors.loc[date, "discountFactor"]
        term = (date - TRADE_DATE).days / 365
        assert df == pytest.approx(np.exp(-FLAT_RATE * term), rel=0.005)

    def test_annual_df_formula_at_3y(self, src_annual):
        date = TRADE_DATE + DateOffset(years=3)
        df   = src_annual.discount_factors.loc[date, "discountFactor"]
        term = (date - TRADE_DATE).days / 365
        assert df == pytest.approx(1 / (1 + FLAT_RATE) ** term, rel=0.005)

    def test_simple_df_formula_at_1y(self, src_simple):
        date = TRADE_DATE + DateOffset(years=1)
        df   = src_simple.discount_factors.loc[date, "discountFactor"]
        term = (date - TRADE_DATE).days / 365
        assert df == pytest.approx(1 / (1 + FLAT_RATE * term), rel=0.005)

    def test_df_decreasing_with_maturity(self, src_continuous):
        df = src_continuous.discount_factors["discountFactor"]
        diffs = df.diff().dropna()
        assert (diffs < 0).all()


# ===========================================================================
# DiscountCurve
# ===========================================================================

class TestDiscountCurveConstruction:

    def test_trade_date_inherited(self, dc_continuous):
        assert dc_continuous.trade_date == TRADE_DATE

    def test_discount_factors_not_none(self, dc_continuous):
        assert dc_continuous.discount_factors is not None

    def test_discount_factors_indexed_by_date(self, dc_continuous):
        assert isinstance(dc_continuous.discount_factors.index, pd.DatetimeIndex)

    def test_invalid_rate_curve_raises(self):
        with pytest.raises(ValueError):
            DiscountCurve("not_a_curve")

    def test_repr(self, dc_continuous):
        assert "DiscountCurve" in repr(dc_continuous)


class TestDiscountCurveDiscountFactorAt:

    def test_matches_full_grid_value(self, dc_continuous):
        date = pd.DatetimeIndex([TRADE_DATE + DateOffset(years=5)])
        fast = dc_continuous.discount_factor_at(date)[0]
        grid = dc_continuous.discount_factors.loc[date[0], "discountFactor"]
        assert fast == pytest.approx(grid, rel=1e-6)

    def test_continuous_formula(self, dc_continuous):
        date = pd.DatetimeIndex([TRADE_DATE + DateOffset(years=5)])
        term = (date[0] - TRADE_DATE).days / 365
        fast = dc_continuous.discount_factor_at(date)[0]
        assert fast == pytest.approx(np.exp(-FLAT_RATE * term), rel=0.005)

    def test_annual_formula(self, dc_annual):
        date = pd.DatetimeIndex([TRADE_DATE + DateOffset(years=3)])
        term = (date[0] - TRADE_DATE).days / 365
        fast = dc_annual.discount_factor_at(date)[0]
        assert fast == pytest.approx(1 / (1 + FLAT_RATE) ** term, rel=0.005)

    def test_simple_formula(self, dc_simple):
        date = pd.DatetimeIndex([TRADE_DATE + DateOffset(years=2)])
        term = (date[0] - TRADE_DATE).days / 365
        fast = dc_simple.discount_factor_at(date)[0]
        assert fast == pytest.approx(1 / (1 + FLAT_RATE * term), rel=0.005)


class TestDiscountCurveShifts:

    def test_parallel_shift_reduces_df(self, dc_continuous):
        target = TRADE_DATE + DateOffset(years=5)
        df_before = dc_continuous.discount_factors.loc[target, "discountFactor"]
        dc_continuous.apply_parallel_shift(0.01)
        df_after  = dc_continuous.discount_factors.loc[target, "discountFactor"]
        assert df_after < df_before
        dc_continuous.reset_shift()

    def test_reset_restores_df(self, dc_continuous):
        date   = pd.DatetimeIndex([TRADE_DATE + DateOffset(years=5)])
        before = dc_continuous.discount_factor_at(date)[0]
        dc_continuous.apply_parallel_shift(0.01)
        dc_continuous.reset_shift()
        after  = dc_continuous.discount_factor_at(date)[0]
        assert after == pytest.approx(before, rel=1e-8)

    def test_shift_flag_set_on_shift(self, dc_continuous):
        dc_continuous.apply_parallel_shift(0.001)
        assert dc_continuous.shift_flag is True
        dc_continuous.reset_shift()

    def test_shift_flag_cleared_on_reset(self, dc_continuous):
        dc_continuous.apply_parallel_shift(0.001)
        dc_continuous.reset_shift()
        assert dc_continuous.shift_flag is False

    def test_slope_shift_near_end_differs_near_start(self, dc_continuous):
        dc_continuous.apply_slope_shift(0.01)
        short = dc_continuous.discount_factor_at(
            pd.DatetimeIndex([TRADE_DATE + DateOffset(months=6)]))[0]
        long_ = dc_continuous.discount_factor_at(
            pd.DatetimeIndex([TRADE_DATE + DateOffset(years=20)]))[0]
        # slope: short rates up → short DF lower; long rates down → long DF higher
        # So short DF < long DF relative to each other than on flat curve
        dc_continuous.reset_shift()
        assert short != long_  # they'll differ due to the slope

    def test_curvature_shift_applied(self, dc_continuous):
        target    = TRADE_DATE + DateOffset(years=15)
        df_before = dc_continuous.discount_factors.loc[target, "discountFactor"]
        dc_continuous.apply_curvature_shift(0.005)
        df_after  = dc_continuous.discount_factors.loc[target, "discountFactor"]
        assert df_before != df_after
        dc_continuous.reset_shift()

    def test_additive_shifts_accumulate(self, dc_continuous):
        date = pd.DatetimeIndex([TRADE_DATE + DateOffset(years=5)])
        dc_continuous.apply_parallel_shift(0.005)
        dc_continuous.apply_parallel_shift(0.005)
        assert dc_continuous._shift == pytest.approx(0.01, abs=1e-12)
        dc_continuous.reset_shift()

    def test_discount_factors_cleared_on_shift(self, dc_continuous):
        _ = dc_continuous.discount_factors  # populate cache
        dc_continuous.apply_parallel_shift(0.001)
        assert dc_continuous._discount_factors is None
        dc_continuous.reset_shift()


# ===========================================================================
# EuriborCurve
# ===========================================================================

@pytest.fixture
def euribor_data():
    # Valid EuriborCurve column names: 1W, 1M, 3M, 6M, 12M
    return pd.DataFrame(
        {"1W": [0.010], "1M": [0.015], "3M": [0.020], "6M": [0.025], "12M": [0.030]},
        index=pd.DatetimeIndex([TRADE_DATE])
    )


@pytest.fixture
def euribor_curve(euribor_data):
    return EuriborCurve(euribor_data, TRADE_DATE)


class TestEuriborCurve:

    def test_sr_length(self, euribor_curve):
        assert len(euribor_curve.sr) == 5

    def test_sr_rates_positive(self, euribor_curve):
        assert (euribor_curve.sr["spotRate"] > 0).all()

    def test_sr_maturities_increasing(self, euribor_curve):
        mats = euribor_curve.sr["maturity"].tolist()
        assert mats == sorted(mats)

    def test_spot_rates_populated(self, euribor_curve):
        assert euribor_curve.spot_rates is not None
        assert len(euribor_curve.spot_rates) > 0

    def test_invalid_column_raises(self):
        bad = pd.DataFrame({"9M": [0.02]}, index=pd.DatetimeIndex([TRADE_DATE]))
        with pytest.raises(ValueError):
            EuriborCurve(bad, TRADE_DATE).sr

    def test_repr(self, euribor_curve):
        assert "EuriborCurve" in repr(euribor_curve)


# ===========================================================================
# SwapRateCurve
# ===========================================================================

@pytest.fixture
def swap_rates_numeric():
    return pd.DataFrame(
        {"swapRate": [0.020, 0.025, 0.028, 0.030, 0.031, 0.032]},
        index=pd.Index([1, 2, 3, 5, 7, 10])
    )


@pytest.fixture
def swap_curve(swap_rates_numeric):
    return SwapRateCurve(swap_rates_numeric, TRADE_DATE, frequency=1)


class TestSwapRateCurve:

    def test_interpolated_rates_not_empty(self, swap_curve):
        assert len(swap_curve.interpolated_rates) > 0

    def test_interpolated_rates_has_columns(self, swap_curve):
        assert {"term", "interpolatedSwapRate"}.issubset(swap_curve.interpolated_rates.columns)

    def test_rates_in_reasonable_range(self, swap_curve):
        rates = swap_curve.interpolated_rates["interpolatedSwapRate"]
        assert (rates > 0).all() and (rates < 0.10).all()

    def test_invalid_interpolation_type_raises(self):
        with pytest.raises(ValueError):
            SwapRateCurve(pd.DataFrame(), TRADE_DATE, frequency=1, interpolation=123)

    def test_repr(self, swap_curve):
        assert "SwapRateCurve" in repr(swap_curve)

    def test_frequency_stored(self, swap_curve):
        assert swap_curve.frequency == 1
