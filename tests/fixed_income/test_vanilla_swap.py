"""
Tests for VanillaSwap.
"""
import numpy as np
import pandas as pd
import pytest
from pandas.tseries.offsets import DateOffset

from QuantGYMM.term_structures import SpotRateCurve, DiscountCurve
from QuantGYMM.fixed_income.instruments import VanillaSwap

TRADE_DATE = pd.Timestamp("2020-01-02")
FLAT_RATE  = 0.03

MATURITIES = pd.DatetimeIndex([TRADE_DATE + DateOffset(months=m)
                                for m in [3, 6, 12, 24, 36, 60, 84, 120, 180, 240, 360]])
_sr_flat = pd.DataFrame({"spotRate": [FLAT_RATE] * len(MATURITIES)}, index=MATURITIES)


@pytest.fixture
def dc():
    src = SpotRateCurve(_sr_flat.copy(), TRADE_DATE, dcc="ACT/365",
                        interpolation="linear", compounding="continuous")
    return DiscountCurve(src, compounding="continuous", dcc="ACT/365", interpolation="linear")


@pytest.fixture
def swap_5y(dc):
    """5Y pay-fixed / receive-floating at-market swap, annual fixed / semiannual floating."""
    return VanillaSwap(dc, fixed_leg_frequency=1, floating_leg_frequency=2,
                       maturity=5, start="today")


@pytest.fixture
def swap_10y(dc):
    return VanillaSwap(dc, fixed_leg_frequency=1, floating_leg_frequency=2,
                       maturity=10, start="today")


# ---------------------------------------------------------------------------
# Construction and setters
# ---------------------------------------------------------------------------

class TestVanillaSwapConstruction:

    def test_trade_date_propagated(self, swap_5y):
        assert swap_5y.start == TRADE_DATE

    def test_maturity_later_than_start(self, swap_5y):
        assert swap_5y.maturity > swap_5y.start

    def test_value_date_two_days_after_start(self, swap_5y):
        from pandas.tseries.offsets import BDay
        assert swap_5y.value_date == TRADE_DATE + BDay(2)

    def test_invalid_discount_curve_raises(self):
        with pytest.raises(ValueError):
            VanillaSwap("not_a_curve", fixed_leg_frequency=1,
                        floating_leg_frequency=2, maturity=5)

    def test_maturity_as_string(self, dc):
        s = VanillaSwap(dc, fixed_leg_frequency=1, floating_leg_frequency=2,
                        maturity="2025-01-02")
        assert isinstance(s.maturity, pd.Timestamp)

    def test_maturity_as_timestamp(self, dc):
        s = VanillaSwap(dc, fixed_leg_frequency=1, floating_leg_frequency=2,
                        maturity=pd.Timestamp("2025-01-02"))
        assert s.maturity == pd.Timestamp("2025-01-02")

    def test_invalid_maturity_string_raises(self, dc):
        with pytest.raises(Exception):
            VanillaSwap(dc, fixed_leg_frequency=1, floating_leg_frequency=2,
                        maturity="not-a-date")


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------

class TestVanillaSwapCalendar:

    def test_calendar_has_required_keys(self, swap_5y):
        cal = swap_5y.calendar
        assert "fixedLeg" in cal
        assert "floatingLeg" in cal
        assert "resetDate" in cal

    def test_fixed_leg_count_annual_5y(self, swap_5y):
        assert len(swap_5y.calendar["fixedLeg"]) == 5

    def test_floating_leg_count_semiannual_5y(self, swap_5y):
        assert len(swap_5y.calendar["floatingLeg"]) == 10

    def test_reset_dates_count(self, swap_5y):
        assert len(swap_5y.calendar["resetDate"]) == 10

    def test_calendar_cached(self, swap_5y):
        c1 = swap_5y.calendar
        c2 = swap_5y.calendar
        assert c1 is c2

    def test_calendar_invalidated_on_maturity_change(self, swap_5y, dc):
        _ = swap_5y.calendar
        swap_5y.maturity = 7
        assert swap_5y._calendar is None


# ---------------------------------------------------------------------------
# Swap rate
# ---------------------------------------------------------------------------

class TestVanillaSwapRate:

    def test_swap_rate_positive(self, swap_5y):
        assert swap_5y.swap_rate > 0

    def test_swap_rate_close_to_market_rate(self, swap_5y):
        """On a flat 3% curve, fair swap rate should be near 3%."""
        assert swap_5y.swap_rate == pytest.approx(FLAT_RATE, rel=0.05)

    def test_swap_rate_cached(self, swap_5y):
        r1 = swap_5y.swap_rate
        r2 = swap_5y.swap_rate
        assert r1 == r2

    def test_longer_maturity_rate_close_to_shorter(self, swap_5y, swap_10y):
        """On a flat curve, swap rates at different maturities are nearly identical."""
        assert swap_5y.swap_rate == pytest.approx(swap_10y.swap_rate, rel=0.01)


# ---------------------------------------------------------------------------
# Market price (NPV)
# ---------------------------------------------------------------------------

class TestVanillaSwapPrice:

    def test_market_price_zero_at_inception(self, swap_5y):
        """By construction the fair swap rate makes NPV = 0."""
        assert swap_5y.market_price() == pytest.approx(0.0, abs=1e-6)

    def test_market_price_zero_10y(self, swap_10y):
        assert swap_10y.market_price() == pytest.approx(0.0, abs=1e-6)

    def test_above_market_fixed_rate_positive_npv(self, dc):
        """If the fixed leg is above the fair rate, the fixed-receiver NPV > 0."""
        swap = VanillaSwap(dc, fixed_leg_frequency=1, floating_leg_frequency=2,
                           maturity=5)
        fair = swap.swap_rate
        # Manually set a higher fixed rate
        swap._swap_rate = fair + 0.01
        assert swap.market_price() > 0

    def test_below_market_fixed_rate_negative_npv(self, dc):
        swap = VanillaSwap(dc, fixed_leg_frequency=1, floating_leg_frequency=2,
                           maturity=5)
        fair = swap.swap_rate
        swap._swap_rate = fair - 0.01
        assert swap.market_price() < 0


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------

class TestVanillaSwapSensitivity:

    @pytest.mark.parametrize("shift_type", ["parallel", "slope", "curvature"])
    def test_sensitivity_returns_float(self, swap_5y, shift_type):
        result = swap_5y.sensitivity(shift_type=shift_type)
        assert isinstance(result, float)

    def test_parallel_sensitivity_negative(self, swap_5y):
        """Receive-fixed NPV falls when rates rise."""
        dv01 = swap_5y.sensitivity(shift_type="parallel")
        assert dv01 < 0

    @pytest.mark.parametrize("kind", ["symmetric", "oneside"])
    def test_both_kinds_work(self, swap_5y, kind):
        result = swap_5y.sensitivity(kind=kind)
        assert isinstance(result, float)

    def test_symmetric_and_oneside_same_sign(self, swap_5y):
        sym = swap_5y.sensitivity(kind="symmetric")
        one = swap_5y.sensitivity(kind="oneside")
        assert np.sign(sym) == np.sign(one)

    def test_invalid_shift_type_raises(self, swap_5y):
        with pytest.raises(ValueError):
            swap_5y.sensitivity(shift_type="invalid")

    def test_invalid_kind_raises(self, swap_5y):
        with pytest.raises(ValueError):
            swap_5y.sensitivity(kind="invalid")

    def test_curve_restored_after_sensitivity(self, swap_5y):
        price_before = swap_5y.market_price()
        swap_5y.sensitivity(shift_type="parallel")
        # reset_shift clears discount_factors; re-price should use restored rates
        price_after = swap_5y.market_price()
        assert abs(price_before - price_after) < 1e-8

    def test_longer_swap_larger_dv01(self, swap_5y, swap_10y):
        """10Y swap should have larger absolute DV01 than 5Y swap."""
        dv01_5  = abs(swap_5y.sensitivity())
        dv01_10 = abs(swap_10y.sensitivity())
        assert dv01_10 > dv01_5
