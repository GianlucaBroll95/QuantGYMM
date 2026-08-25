"""
Tests for CallableBond (price-to-worst and Hull-White valuation).
"""
import numpy as np
import pandas as pd
import pytest

from QuantGYMM.calendar import Schedule
from QuantGYMM.fixed_income.instruments import FixedRateBond, FloatingRateBond, CallableBond
from QuantGYMM.fixed_income.pricers import Pricer
from QuantGYMM.term_structures import SpotRateCurve, DiscountCurve
from QuantGYMM.fixed_income.models import hw_b, hw_zcb_option

TRADE_DATE = pd.Timestamp("2020-01-02")
EVAL_DATE = pd.Timestamp("2022-06-15")
FLAT_RATE = 0.02

MATURITIES = pd.DatetimeIndex([TRADE_DATE + pd.DateOffset(months=m)
                                for m in [3, 6, 12, 24, 36, 60, 84, 120]])
_sr_flat = pd.DataFrame({"spotRate": [FLAT_RATE] * len(MATURITIES)}, index=MATURITIES)


@pytest.fixture
def dc():
    src = SpotRateCurve(_sr_flat.copy(), TRADE_DATE, dcc="ACT/365",
                        interpolation="linear", compounding="continuous")
    return DiscountCurve(src, compounding="continuous", dcc="ACT/365",
                         interpolation="linear")


@pytest.fixture
def schedule_annual():
    return Schedule(start_date=TRADE_DATE,
                    end_date=pd.Timestamp("2024-01-02"),
                    frequency=1)


@pytest.fixture
def call_date(schedule_annual):
    payments = schedule_annual.schedule["paymentDate"]
    return [d for d in payments if d.year == 2023][0]


@pytest.fixture
def call_sched(call_date):
    return pd.Series([100.0], index=pd.DatetimeIndex([call_date]))


@pytest.fixture
def fixed_bond(schedule_annual, dc):
    b = FixedRateBond(schedule_annual, "30/360", 1_000_000.0, coupon_rate=0.05)
    b.set_evaluation_date(EVAL_DATE)
    b.set_discount_curve(dc)
    return b


@pytest.fixture
def floating_bond(schedule_annual, dc):
    b = FloatingRateBond(
        schedule=schedule_annual,
        dcc="ACT/360",
        face_amount=1_000_000.0,
        fixing_days=0,
        spread=0.001,
    )
    b.set_evaluation_date(EVAL_DATE)
    reset_dates = b.schedule.schedule["resetDate"]
    known_mask = reset_dates <= EVAL_DATE
    hist_resets = reset_dates[known_mask]
    rates = [0.015, 0.018, 0.020][:len(hist_resets)]
    euribor = pd.DataFrame({"rate": rates}, index=hist_resets)
    b.set_historical_euribor(euribor)
    b.set_pricer(Pricer(dc))
    return b


@pytest.fixture
def callable_fixed(fixed_bond, call_sched):
    return CallableBond(fixed_bond, call_sched)


@pytest.fixture
def callable_floating(floating_bond, call_sched):
    return CallableBond(floating_bond, call_sched)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_wrong_bond_type_raises(self, call_sched):
        with pytest.raises(ValueError):
            CallableBond("not_a_bond", call_sched)

    def test_call_schedule_not_series_raises(self, fixed_bond):
        with pytest.raises(ValueError):
            CallableBond(fixed_bond, [100.0])

    def test_non_datetime_index_raises(self, fixed_bond):
        bad = pd.Series([100.0], index=[0])
        with pytest.raises(ValueError):
            CallableBond(fixed_bond, bad)

    def test_call_date_not_on_payment_date_raises(self, fixed_bond):
        bad = pd.Series([100.0], index=pd.DatetimeIndex([pd.Timestamp("2023-05-01")]))
        with pytest.raises(ValueError, match="payment date"):
            CallableBond(fixed_bond, bad)

    def test_all_past_call_dates_raises_on_access(self, fixed_bond, schedule_annual):
        past_date = schedule_annual.schedule["paymentDate"][0]  # 2021 payment, before EVAL_DATE
        sched = pd.Series([100.0], index=pd.DatetimeIndex([past_date]))
        cb = CallableBond(fixed_bond, sched)
        with pytest.raises(ValueError, match="past"):
            _ = cb.call_schedule

    def test_past_filtered_when_some_future(self, fixed_bond, schedule_annual, call_date):
        past_date = schedule_annual.schedule["paymentDate"][0]
        sched = pd.Series([100.0, 100.0], index=pd.DatetimeIndex([past_date, call_date]))
        cb = CallableBond(fixed_bond, sched)
        assert len(cb.call_schedule) == 1
        assert cb.call_schedule.index[0] == call_date

    def test_face_amount_delegation(self, callable_fixed, fixed_bond):
        assert callable_fixed.face_amount == fixed_bond.face_amount

    def test_currency_delegation(self, callable_fixed, fixed_bond):
        assert callable_fixed.currency == fixed_bond.currency == "EUR"

    def test_currency_delegation_non_default(self, schedule_annual, dc, call_sched):
        bond = FixedRateBond(schedule_annual, "30/360", 1_000_000.0, coupon_rate=0.05, currency="USD")
        bond.set_evaluation_date(EVAL_DATE)
        bond.set_discount_curve(dc)
        cb = CallableBond(bond, call_sched)
        assert cb.currency == "USD"

    def test_evaluation_date_delegation(self, callable_fixed, fixed_bond):
        assert callable_fixed.evaluation_date == fixed_bond.evaluation_date

    def test_invalid_mean_reversion_raises(self, fixed_bond, call_sched):
        with pytest.raises(ValueError):
            CallableBond(fixed_bond, call_sched, mean_reversion=-0.01)

    def test_invalid_volatility_raises(self, fixed_bond, call_sched):
        with pytest.raises(ValueError):
            CallableBond(fixed_bond, call_sched, volatility=0.0)


# ---------------------------------------------------------------------------
# Price to worst — fixed
# ---------------------------------------------------------------------------

class TestPriceToWorstFixed:

    def test_returns_dict_with_keys(self, callable_fixed):
        p = callable_fixed.prices()
        assert set(p.keys()) == {"straightValue", "optionValue", "callableValue"}

    def test_callable_leq_straight(self, callable_fixed):
        p = callable_fixed.prices()
        assert p["callableValue"]["dirtyPrice"] <= p["straightValue"]["dirtyPrice"]

    def test_option_value_nonnegative(self, callable_fixed):
        p = callable_fixed.prices()
        assert p["optionValue"] >= 0

    def test_absurd_call_price_gives_zero_option(self, fixed_bond, call_date):
        sched = pd.Series([999.0], index=pd.DatetimeIndex([call_date]))
        cb = CallableBond(fixed_bond, sched)
        p = cb.prices()
        assert p["optionValue"] == pytest.approx(0.0, abs=1e-9)
        assert p["callableValue"]["dirtyPrice"] == pytest.approx(p["straightValue"]["dirtyPrice"])

    def test_straight_value_matches_bond(self, callable_fixed, fixed_bond):
        p = callable_fixed.prices()
        assert p["straightValue"] == fixed_bond.prices()["riskFreeValue"]

    def test_dirty_equals_clean_plus_accrued(self, callable_fixed):
        p = callable_fixed.prices()["callableValue"]
        assert abs(p["dirtyPrice"] - p["cleanPrice"] - p["accruedInterest"]) < 1e-6


# ---------------------------------------------------------------------------
# Price to worst — floating
# ---------------------------------------------------------------------------

class TestPriceToWorstFloating:

    def test_callable_leq_straight(self, callable_floating):
        p = callable_floating.prices()
        assert p["callableValue"]["dirtyPrice"] <= p["straightValue"]["dirtyPrice"]

    def test_option_value_nonnegative(self, callable_floating):
        p = callable_floating.prices()
        assert p["optionValue"] >= 0

    def test_absurd_call_price_gives_zero_option(self, floating_bond, call_date):
        sched = pd.Series([999.0], index=pd.DatetimeIndex([call_date]))
        cb = CallableBond(floating_bond, sched)
        p = cb.prices()
        assert p["optionValue"] == pytest.approx(0.0, abs=1e-9)

    def test_works_without_historical_euribor(self, schedule_annual, dc, call_date):
        b = FloatingRateBond(schedule_annual, "ACT/360", 1_000_000.0, fixing_days=0, spread=0.001)
        b.set_evaluation_date(pd.Timestamp("2020-03-15"))
        b.set_current_coupon_rate(0.02)
        b.set_pricer(Pricer(dc))
        sched = pd.Series([100.0], index=pd.DatetimeIndex([call_date]))
        cb = CallableBond(b, sched)
        p = cb.prices()
        assert p["optionValue"] >= 0


# ---------------------------------------------------------------------------
# Hull-White
# ---------------------------------------------------------------------------

class TestHullWhite:

    def test_option_increasing_in_sigma(self, fixed_bond, call_sched):
        cb_low = CallableBond(fixed_bond, call_sched, volatility=0.005)
        cb_mid = CallableBond(fixed_bond, call_sched, volatility=0.01)
        cb_high = CallableBond(fixed_bond, call_sched, volatility=0.02)
        v_low = cb_low.prices(method="hw")["optionValue"]
        v_mid = cb_mid.prices(method="hw")["optionValue"]
        v_high = cb_high.prices(method="hw")["optionValue"]
        assert v_low < v_mid < v_high

    def test_low_sigma_matches_intrinsic(self, fixed_bond, call_sched, dc):
        cb = CallableBond(fixed_bond, call_sched, volatility=1e-9)
        p = cb.prices(method="hw")
        straight = p["straightValue"]["dirtyPrice"]
        callable_ = p["callableValue"]["dirtyPrice"]
        call_date = call_sched.index[0]

        dates, coupons = cb._future_cash_flows()
        mask = dates <= call_date
        called_dates = dates[mask].append(pd.DatetimeIndex([call_date]))
        called_cfs = np.append(coupons[mask], call_sched.iloc[0] / 100.0 * cb.face_amount)
        df = dc.discount_factor_at(called_dates)
        called_pv = called_cfs.dot(df)

        intrinsic = max(0.0, straight - called_pv)
        assert abs((straight - callable_) - intrinsic) < 1e-6 * cb.face_amount

    def test_callable_leq_straight(self, fixed_bond, call_sched):
        cb = CallableBond(fixed_bond, call_sched)
        p = cb.prices(method="hw")
        assert p["callableValue"]["dirtyPrice"] <= p["straightValue"]["dirtyPrice"]

    def test_option_nonnegative(self, fixed_bond, call_sched):
        cb = CallableBond(fixed_bond, call_sched)
        assert cb.prices(method="hw")["optionValue"] >= 0

    def test_floater_raises(self, callable_floating):
        with pytest.raises(NotImplementedError):
            callable_floating.prices(method="hw")

    def test_two_call_dates_raises(self, fixed_bond, schedule_annual):
        payments = schedule_annual.schedule["paymentDate"]
        future = [d for d in payments if d > EVAL_DATE][:2]
        sched = pd.Series([100.0, 100.0], index=pd.DatetimeIndex(future))
        cb = CallableBond(fixed_bond, sched)
        with pytest.raises(NotImplementedError):
            cb.prices(method="hw")

    def test_cds_underlying_raises(self, fixed_bond, call_sched):
        fixed_bond.set_cds_spread(0.02)
        fixed_bond.set_recovery_rate(0.4)
        cb = CallableBond(fixed_bond, call_sched)
        with pytest.raises(NotImplementedError):
            cb.prices(method="hw")
        fixed_bond.set_cds_spread(None)  # cleanup


class TestBulletToCall:

    def test_returns_dict_with_risk_free_value(self, callable_fixed):
        p = callable_fixed.bullet_price()
        assert "riskFreeValue" in p

    def test_dirty_price_positive(self, callable_fixed):
        p = callable_fixed.bullet_price()
        assert p["riskFreeValue"]["dirtyPrice"] > 0

    def test_matches_manual_truncated_bond(self, fixed_bond, call_sched, call_date, dc):
        cb = CallableBond(fixed_bond, call_sched)
        p = cb.bullet_price()

        manual = FixedRateBond(
            Schedule(TRADE_DATE, call_date, frequency=1),
            "30/360", fixed_bond.face_amount, fixed_bond.coupon_rate,
            redemption=call_sched.iloc[0]
        )
        manual.set_evaluation_date(EVAL_DATE)
        manual.set_discount_curve(dc)
        expected = manual.prices()

        assert p["riskFreeValue"]["dirtyPrice"] == pytest.approx(
            expected["riskFreeValue"]["dirtyPrice"], rel=1e-8)

    def test_absurd_call_price_gives_higher_bullet_price(self, fixed_bond, call_date):
        low_sched = pd.Series([100.0], index=pd.DatetimeIndex([call_date]))
        high_sched = pd.Series([120.0], index=pd.DatetimeIndex([call_date]))
        cb_low = CallableBond(fixed_bond, low_sched)
        cb_high = CallableBond(fixed_bond, high_sched)
        assert (cb_high.bullet_price()["riskFreeValue"]["dirtyPrice"]
                > cb_low.bullet_price()["riskFreeValue"]["dirtyPrice"])

    def test_original_bond_schedule_untouched(self, callable_fixed, fixed_bond):
        n_before = len(fixed_bond.schedule.schedule["paymentDate"])
        callable_fixed.bullet_price()
        n_after = len(fixed_bond.schedule.schedule["paymentDate"])
        assert n_before == n_after

    def test_floater_raises(self, callable_floating):
        with pytest.raises(NotImplementedError):
            callable_floating.bullet_price()

    def test_second_call_index(self, fixed_bond, schedule_annual):
        payments = schedule_annual.schedule["paymentDate"]
        future = [d for d in payments if d > EVAL_DATE][:2]
        sched = pd.Series([100.0, 101.0], index=pd.DatetimeIndex(future))
        cb = CallableBond(fixed_bond, sched)
        p0 = cb.bullet_price(call_index=0)
        p1 = cb.bullet_price(call_index=1)
        assert p0["riskFreeValue"]["dirtyPrice"] != p1["riskFreeValue"]["dirtyPrice"]

# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

class TestGreeks:

    def test_sensitivity_negative(self, callable_fixed):
        assert callable_fixed.sensitivity() < 0

    def test_callable_dv01_leq_straight_dv01(self, fixed_bond, call_sched):
        # in-the-money call (5% coupon, 2% curve, at-par call): shorter effective duration
        cb = CallableBond(fixed_bond, call_sched)
        cb_dv01 = abs(cb.sensitivity())
        straight_dv01 = abs(fixed_bond.sensitivity())
        assert cb_dv01 <= straight_dv01

    def test_curve_restored_after_sensitivity(self, callable_fixed, dc):
        original = dc.rate_curve.spot_rates_data.copy()
        callable_fixed.sensitivity()
        pd.testing.assert_frame_equal(dc.rate_curve.spot_rates_data, original)

    def test_curve_restored_after_krd(self, callable_fixed, dc):
        original = dc.rate_curve.spot_rates_data.copy()
        callable_fixed.key_rate_dv01()
        pd.testing.assert_frame_equal(dc.rate_curve.spot_rates_data, original)

    def test_krd_keys_match_curve_nodes(self, callable_fixed, dc):
        krd = callable_fixed.key_rate_dv01()
        assert list(krd.keys()) == list(dc.rate_curve.spot_rates_data.index)

    def test_symmetric_oneside_same_sign(self, callable_fixed):
        sym = callable_fixed.sensitivity(kind="symmetric")
        one = callable_fixed.sensitivity(kind="oneside")
        assert np.sign(sym) == np.sign(one)

    def test_effective_duration_positive_and_less_than_straight(self, callable_fixed, fixed_bond):
        eff = callable_fixed.effective_duration()
        straight_mod_dur = fixed_bond.modified_duration()
        assert eff > 0
        assert eff < straight_mod_dur