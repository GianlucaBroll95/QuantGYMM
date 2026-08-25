"""
Tests for FloatingRateBond and Pricer (base, Black, Bachelier, DisplacedBlack).
"""
import numpy as np
import pandas as pd
import pytest
from pandas.tseries.offsets import DateOffset, BDay

from QuantGYMM.calendar import Schedule
from QuantGYMM.fixed_income.instruments import FloatingRateBond
from QuantGYMM.fixed_income.pricers import Pricer, BlackPricer, BachelierPricer, DisplacedBlackPricer
from QuantGYMM.term_structures import SpotRateCurve, DiscountCurve

# ---------------------------------------------------------------------------
# Common market data
# ---------------------------------------------------------------------------

TRADE_DATE = pd.Timestamp("2020-01-02")
EVAL_DATE  = pd.Timestamp("2022-06-15")
FLAT_RATE  = 0.02

MATURITIES = pd.DatetimeIndex([TRADE_DATE + DateOffset(months=m)
                                for m in [3, 6, 12, 24, 36, 60, 84, 120]])
_sr_flat   = pd.DataFrame({"spotRate": [FLAT_RATE] * len(MATURITIES)}, index=MATURITIES)


@pytest.fixture
def dc():
    src = SpotRateCurve(_sr_flat.copy(), TRADE_DATE, dcc="ACT/365",
                        interpolation="linear", compounding="continuous")
    return DiscountCurve(src, compounding="continuous", dcc="ACT/365",
                         interpolation="linear")


@pytest.fixture
def schedule_annual():
    """Annual schedule 2020-01-02 → 2024-01-02 (4 coupon periods)."""
    return Schedule(start_date=TRADE_DATE,
                    end_date=pd.Timestamp("2024-01-02"),
                    frequency=1)


@pytest.fixture
def bond_annual(schedule_annual, dc):
    """Annual FloatingRateBond with fixing_days=0 (reset = starting date)."""
    b = FloatingRateBond(
        schedule=schedule_annual,
        dcc="ACT/360",
        face_amount=1_000_000.0,
        fixing_days=0,
        spread=0.001,
    )
    b.set_evaluation_date(EVAL_DATE)

    # Include every reset date that has already occurred, which covers:
    #   - fully settled periods (paymentDate <= EVAL_DATE)
    #   - the current in-flight period (reset <= EVAL_DATE, payment still future)
    # _get_current_coupon looks up the current period's reset in historical_euribor.
    reset_dates = b.schedule.schedule["resetDate"]
    known_mask  = reset_dates <= EVAL_DATE
    hist_resets = reset_dates[known_mask]
    rates       = [0.015, 0.018, 0.020][:len(hist_resets)]
    euribor     = pd.DataFrame({"rate": rates}, index=hist_resets)
    b.set_historical_euribor(euribor)
    return b


@pytest.fixture
def pricer(dc):
    return Pricer(dc)


# ---------------------------------------------------------------------------
# Construction and descriptor validation
# ---------------------------------------------------------------------------

class TestFloatingRateBondConstruction:

    def test_face_amount_stored(self, bond_annual):
        assert bond_annual.face_amount == 1_000_000.0

    def test_spread_stored(self, bond_annual):
        assert bond_annual.spread == 0.001

    def test_fixing_days_stored(self, bond_annual):
        assert bond_annual.fixing_days == 0

    def test_default_currency_is_eur(self, schedule_annual):
        b = FloatingRateBond(schedule_annual, "ACT/360", 100.0, 0)
        assert b.currency == "EUR"

    def test_custom_currency(self, schedule_annual):
        b = FloatingRateBond(schedule_annual, "ACT/360", 100.0, 0, currency="GBP")
        assert b.currency == "GBP"

    def test_default_cap_is_nan(self, schedule_annual):
        b = FloatingRateBond(schedule_annual, "ACT/360", 100.0, 0)
        assert np.isnan(b.cap)

    def test_default_floor_is_nan(self, schedule_annual):
        b = FloatingRateBond(schedule_annual, "ACT/360", 100.0, 0)
        assert np.isnan(b.floor)

    def test_invalid_face_amount_raises(self, schedule_annual):
        with pytest.raises(TypeError):
            FloatingRateBond(schedule_annual, "ACT/360", -100.0, 0)

    def test_invalid_schedule_raises(self):
        with pytest.raises(ValueError):
            FloatingRateBond("not_a_schedule", "ACT/360", 100.0, 0)

    def test_invalid_dcc_raises(self, schedule_annual):
        with pytest.raises(ValueError):
            FloatingRateBond(schedule_annual, "INVALID", 100.0, 0)

    def test_repr(self, bond_annual):
        r = repr(bond_annual)
        assert "Bond" in r


# ---------------------------------------------------------------------------
# Schedule — reset dates
# ---------------------------------------------------------------------------

class TestFloatingRateBondSchedule:

    def test_reset_dates_in_schedule(self, bond_annual):
        assert "resetDate" in bond_annual.schedule.schedule

    def test_reset_dates_count(self, bond_annual):
        # fixing_days=0 → resetDate = startingDate, same count as coupons
        assert len(bond_annual.schedule.schedule["resetDate"]) == len(
            bond_annual.schedule.schedule["paymentDate"])

    def test_reset_dates_before_starting(self, schedule_annual):
        b = FloatingRateBond(schedule_annual, "ACT/360", 100.0, fixing_days=2)
        resets   = b.schedule.schedule["resetDate"]
        starting = b.schedule.schedule["startingDate"]
        for r, s in zip(resets, starting):
            assert r <= s

    def test_invalid_schedule_type_raises(self, bond_annual):
        with pytest.raises(ValueError):
            bond_annual.schedule = "not_a_schedule"


# ---------------------------------------------------------------------------
# Evaluation date and setters
# ---------------------------------------------------------------------------

class TestFloatingRateBondSetters:

    def test_evaluation_date_stored(self, bond_annual):
        assert bond_annual.evaluation_date == EVAL_DATE

    def test_evaluation_date_not_set_raises(self, schedule_annual):
        b = FloatingRateBond(schedule_annual, "ACT/360", 100.0, 0)
        with pytest.raises(ValueError, match="set_evaluation_date"):
            _ = b.evaluation_date

    def test_invalid_evaluation_date_raises(self, schedule_annual):
        b = FloatingRateBond(schedule_annual, "ACT/360", 100.0, 0)
        with pytest.raises(ValueError):
            b.set_evaluation_date("not-a-date")

    def test_historical_euribor_stored(self, bond_annual):
        assert bond_annual.historical_euribor is not None

    def test_historical_euribor_not_set_raises(self, schedule_annual):
        b = FloatingRateBond(schedule_annual, "ACT/360", 100.0, 0)
        with pytest.raises(ValueError):
            _ = b.historical_euribor


# ---------------------------------------------------------------------------
# Coupon history (past periods — no Pricer needed)
# ---------------------------------------------------------------------------

class TestFloatingRateBondCouponHistory:

    def test_history_count(self, bond_annual):
        """At EVAL_DATE 2022-06-15, 2 annual payments have occurred."""
        hist = bond_annual.coupons_history
        assert len(hist) == 2

    def test_history_rates_positive(self, bond_annual):
        hist = bond_annual.coupons_history
        assert (hist["couponRate"] > 0).all()

    def test_spread_included_in_rate(self, bond_annual):
        hist = bond_annual.coupons_history
        # couponRate = resetRate + spread; spread is 0.001
        diff = (hist["couponRate"] - hist["resetRate"]).values
        assert np.allclose(diff, 0.001, atol=1e-10)

    def test_coupons_positive(self, bond_annual):
        hist = bond_annual.coupons_history
        assert (hist["coupon"] > 0).all()

    def test_history_cached(self, bond_annual):
        h1 = bond_annual.coupons_history
        h2 = bond_annual.coupons_history
        assert h1 is h2

    def test_history_invalidated_on_eval_date_change(self, bond_annual):
        _ = bond_annual.coupons_history
        bond_annual.set_evaluation_date(pd.Timestamp("2023-06-01"))
        assert bond_annual._coupons_history is None


# ---------------------------------------------------------------------------
# CDS / survival probabilities
# ---------------------------------------------------------------------------

class TestFloatingRateBondCDS:

    def test_invalid_cds_spread_raises(self, bond_annual):
        with pytest.raises(ValueError):
            bond_annual.set_cds_spread("not_a_float")

    def test_invalid_recovery_rate_raises(self, bond_annual):
        with pytest.raises(ValueError):
            bond_annual.set_recovery_rate("not_a_float")

    def test_cds_spread_stored(self, bond_annual):
        bond_annual.set_cds_spread(0.02)
        assert bond_annual.cds_spread == 0.02
        bond_annual.set_cds_spread(None)  # cleanup

    def test_cds_not_set_raises(self, bond_annual):
        with pytest.raises(ValueError):
            _ = bond_annual.cds_spread


# ---------------------------------------------------------------------------
# Hedging
# ---------------------------------------------------------------------------

class TestFloatingRateBondHedging:

    def test_set_hedging_instruments_requires_iterable(self, bond_annual):
        with pytest.raises(ValueError):
            bond_annual.set_hedging_instruments("not_a_list")

    def test_hedging_instruments_stored(self, bond_annual, schedule_annual, dc):
        """Hedging instruments must expose a 'sensitivity' method."""
        other = FloatingRateBond(schedule_annual, "ACT/360", 500_000.0, 0)
        other.set_evaluation_date(EVAL_DATE)
        hist_resets = bond_annual.schedule.schedule["resetDate"]
        past_mask   = bond_annual.schedule.schedule["paymentDate"] <= EVAL_DATE
        euribor     = pd.DataFrame(
            {"rate": [0.015, 0.018]},
            index=hist_resets[past_mask]
        )
        other.set_historical_euribor(euribor)
        bond_annual.set_hedging_instruments([other])
        assert len(bond_annual.hedging_instruments) == 1

    def test_hedging_not_set_raises(self, bond_annual):
        with pytest.raises(ValueError):
            _ = bond_annual.hedging_instruments


# ---------------------------------------------------------------------------
# Pricer — attachment
# ---------------------------------------------------------------------------

class TestPricerAttachment:

    def test_pricer_attached(self, bond_annual, pricer):
        bond_annual.set_pricer(pricer)
        assert bond_annual.pricer is pricer

    def test_invalid_pricer_raises(self, bond_annual):
        with pytest.raises(ValueError):
            bond_annual.set_pricer("not_a_pricer")

    def test_pricer_not_set_raises(self, schedule_annual):
        b = FloatingRateBond(schedule_annual, "ACT/360", 100.0, 0)
        with pytest.raises(ValueError, match="set_pricer"):
            _ = b.pricer


# ---------------------------------------------------------------------------
# Pricer.present_value — full pricing path
#
# These tests exercise _get_expected_coupons, which references
# self.bond.coupon_history (no 's'). FloatingRateBond exposes
# self.bond.coupons_history (with 's'). Tests are marked xfail
# until the naming inconsistency in pricers.py is resolved.
# ---------------------------------------------------------------------------

class TestPricerPresentValue:

    def test_present_value_returns_dict(self, bond_annual, pricer):
        bond_annual.set_pricer(pricer)
        pv = bond_annual.prices()
        assert isinstance(pv, dict)
        assert "riskFreeValue" in pv

    def test_dirty_price_positive(self, bond_annual, pricer):
        bond_annual.set_pricer(pricer)
        pv = bond_annual.prices()["riskFreeValue"]["dirtyPrice"]
        assert pv > 0

    def test_dirty_equals_clean_plus_accrued(self, bond_annual, pricer):
        bond_annual.set_pricer(pricer)
        p = bond_annual.prices()["riskFreeValue"]
        assert abs(p["dirtyPrice"] - p["cleanPrice"] - p["accruedInterest"]) < 1e-6


# ---------------------------------------------------------------------------
# BlackPricer / BachelierPricer / DisplacedBlackPricer — unit construction
# ---------------------------------------------------------------------------

@pytest.fixture
def vol_surface():
    """Simple flat volatility surface: maturity × strike."""
    maturities = [0.5, 1.0, 2.0, 5.0, 10.0]
    strikes    = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]
    return pd.DataFrame(
        [[0.25] * len(strikes)] * len(maturities),
        index=maturities, columns=strikes
    )


class TestOptionPricers:

    def test_black_pricer_construction(self, dc, vol_surface):
        p = BlackPricer(dc, vol_surface)
        assert p is not None

    def test_bachelier_pricer_construction(self, dc, vol_surface):
        p = BachelierPricer(dc, vol_surface)
        assert p is not None

    def test_displaced_black_pricer_construction(self, dc, vol_surface):
        p = DisplacedBlackPricer(dc, vol_surface, shift=0.03)
        assert p is not None

    def test_invalid_discount_curve_raises(self, vol_surface):
        with pytest.raises(ValueError):
            BlackPricer("not_a_curve", vol_surface)

    def test_invalid_vol_surface_raises(self, dc):
        with pytest.raises(ValueError):
            BlackPricer(dc, "not_a_dataframe")

    def test_black_discount_curve_stored(self, dc, vol_surface):
        p = BlackPricer(dc, vol_surface)
        assert p.discount_curve is dc

    def test_black_pricer_present_value(self, bond_annual, dc, vol_surface):
        """Cap/floor bond with BlackPricer."""
        bond_cap = FloatingRateBond(
            bond_annual.schedule, "ACT/360", 1_000_000.0,
            fixing_days=0, cap=0.04, floor=0.0
        )
        bond_cap.set_evaluation_date(EVAL_DATE)
        bond_cap.set_historical_euribor(bond_annual.historical_euribor)
        p = BlackPricer(dc, vol_surface)
        bond_cap.set_pricer(p)
        pv = bond_cap.prices()
        assert "riskFreeValue" in pv


# ---------------------------------------------------------------------------
# set_current_coupon_rate — bypass historical Euribor
# ---------------------------------------------------------------------------

EVAL_DATE_EARLY = pd.Timestamp("2020-03-15")   # inside first period, no past payments
CURRENT_RATE    = 0.020                         # annualised coupon rate for the current period
SPREAD          = 0.001


@pytest.fixture
def bond_at_inception(schedule_annual, dc):
    """Annual FloatingRateBond at inception: no past payments, no historical Euribor needed."""
    b = FloatingRateBond(
        schedule=schedule_annual,
        dcc="ACT/360",
        face_amount=1_000_000.0,
        fixing_days=0,
        spread=SPREAD,
    )
    b.set_evaluation_date(EVAL_DATE_EARLY)
    b.set_current_coupon_rate(CURRENT_RATE)
    return b


class TestCurrentCouponRatePath:

    def test_current_coupon_rate_stored(self, schedule_annual):
        b = FloatingRateBond(schedule_annual, "ACT/360", 1_000_000.0, 0)
        b.set_current_coupon_rate(0.025)
        assert b.current_coupon_rate == 0.025

    def test_current_coupon_rate_none_resets(self, schedule_annual):
        b = FloatingRateBond(schedule_annual, "ACT/360", 1_000_000.0, 0)
        b.set_current_coupon_rate(0.025)
        b.set_current_coupon_rate(None)
        assert b.current_coupon_rate is None

    def test_invalid_type_raises(self, schedule_annual):
        b = FloatingRateBond(schedule_annual, "ACT/360", 1_000_000.0, 0)
        with pytest.raises(ValueError):
            b.set_current_coupon_rate("not_a_float")

    def test_prices_without_historical_euribor(self, bond_at_inception, dc):
        bond_at_inception.set_pricer(Pricer(dc))
        pv = bond_at_inception.prices()
        assert isinstance(pv, dict)
        assert "riskFreeValue" in pv

    def test_dirty_price_positive_no_history(self, bond_at_inception, dc):
        bond_at_inception.set_pricer(Pricer(dc))
        assert bond_at_inception.prices()["riskFreeValue"]["dirtyPrice"] > 0

    def test_dirty_equals_clean_plus_accrued_no_history(self, bond_at_inception, dc):
        bond_at_inception.set_pricer(Pricer(dc))
        p = bond_at_inception.prices()["riskFreeValue"]
        assert abs(p["dirtyPrice"] - p["cleanPrice"] - p["accruedInterest"]) < 1e-6

    def test_pricer_cache_cleared_on_rate_change(self, bond_annual, pricer):
        bond_annual.set_pricer(pricer)
        _ = bond_annual.prices()                        # populate cache
        bond_annual.set_current_coupon_rate(0.03)
        assert bond_annual.pricer._current_coupon is None
        assert bond_annual.pricer._expected_coupons is None

    def test_higher_rate_gives_higher_price(self, bond_at_inception, dc):
        bond_at_inception.set_pricer(Pricer(dc))
        price_low = bond_at_inception.prices()["riskFreeValue"]["dirtyPrice"]

        bond_at_inception.set_current_coupon_rate(0.05)
        price_high = bond_at_inception.prices()["riskFreeValue"]["dirtyPrice"]

        assert price_high > price_low

    def test_prices_matches_euribor_equivalent(self, schedule_annual, dc):
        """set_current_coupon_rate(r) must give the same price as historical_euribor reset_rate = r - spread."""
        b_direct = FloatingRateBond(schedule_annual, "ACT/360", 1_000_000.0,
                                    fixing_days=0, spread=SPREAD)
        b_direct.set_evaluation_date(EVAL_DATE_EARLY)
        b_direct.set_current_coupon_rate(CURRENT_RATE)
        b_direct.set_pricer(Pricer(dc))
        price_direct = b_direct.prices()["riskFreeValue"]["dirtyPrice"]

        # equivalent bond via historical Euribor: reset_rate = CURRENT_RATE - SPREAD
        b_euribor = FloatingRateBond(schedule_annual, "ACT/360", 1_000_000.0,
                                     fixing_days=0, spread=SPREAD)
        b_euribor.set_evaluation_date(EVAL_DATE_EARLY)
        reset = b_euribor.schedule.schedule["resetDate"][0]
        euribor = pd.DataFrame({"rate": [CURRENT_RATE - SPREAD]},
                               index=pd.DatetimeIndex([reset]))
        b_euribor.set_historical_euribor(euribor)
        b_euribor.set_pricer(Pricer(dc))
        price_euribor = b_euribor.prices()["riskFreeValue"]["dirtyPrice"]

        assert price_direct == pytest.approx(price_euribor, rel=1e-8)


# ---------------------------------------------------------------------------
# key_rate_dv01
# ---------------------------------------------------------------------------

class TestKeyRateDV01:

    def test_returns_dict(self, bond_annual, pricer):
        bond_annual.set_pricer(pricer)
        krd = bond_annual.key_rate_dv01()
        assert isinstance(krd, dict)

    def test_keys_are_timestamps(self, bond_annual, pricer):
        bond_annual.set_pricer(pricer)
        krd = bond_annual.key_rate_dv01()
        assert all(isinstance(k, pd.Timestamp) for k in krd.keys())

    def test_node_count_matches_curve(self, bond_annual, pricer):
        bond_annual.set_pricer(pricer)
        krd = bond_annual.key_rate_dv01()
        n_nodes = len(pricer.discount_curve.rate_curve.spot_rates_data)
        assert len(krd) == n_nodes

    def test_nonzero_nodes_negative(self, bond_annual, pricer):
        """Long bond: rates up → price down → DV01 negative."""
        bond_annual.set_pricer(pricer)
        krd = bond_annual.key_rate_dv01()
        nonzero = [v for v in krd.values() if abs(v) > 1e-12]
        assert len(nonzero) > 0
        assert all(v < 0 for v in nonzero)

    def test_sum_matches_parallel_dv01(self, bond_annual, pricer):
        """Sum of key-rate DV01s ≈ parallel DV01 (within 5%)."""
        bond_annual.set_pricer(pricer)
        krd_sum = sum(bond_annual.key_rate_dv01().values())
        parallel = bond_annual.sensitivity(shift_type="parallel")
        assert abs(krd_sum - parallel) < abs(parallel) * 0.05

    def test_curve_restored_after_krd(self, bond_annual, pricer):
        bond_annual.set_pricer(pricer)
        price_before = bond_annual.prices()["riskFreeValue"]["dirtyPrice"]
        bond_annual.key_rate_dv01()
        price_after = bond_annual.prices()["riskFreeValue"]["dirtyPrice"]
        assert abs(price_before - price_after) < 1e-10

    @pytest.mark.parametrize("kind", ["symmetric", "oneside"])
    def test_both_kinds_work(self, bond_annual, pricer, kind):
        bond_annual.set_pricer(pricer)
        krd = bond_annual.key_rate_dv01(kind=kind)
        assert isinstance(krd, dict)
        assert len(krd) > 0

    def test_symmetric_and_oneside_same_sign(self, bond_annual, pricer):
        bond_annual.set_pricer(pricer)
        sym = bond_annual.key_rate_dv01(kind="symmetric")
        one = bond_annual.key_rate_dv01(kind="oneside")
        for k in sym:
            if abs(sym[k]) > 1e-12:
                assert np.sign(sym[k]) == np.sign(one[k])

    def test_invalid_kind_raises(self, bond_annual, pricer):
        bond_annual.set_pricer(pricer)
        with pytest.raises(ValueError, match="kind"):
            bond_annual.key_rate_dv01(kind="invalid")

    def test_no_history_path(self, bond_at_inception, dc):
        """key_rate_dv01 works with set_current_coupon_rate (no historical Euribor)."""
        bond_at_inception.set_pricer(Pricer(dc))
        krd = bond_at_inception.key_rate_dv01()
        assert isinstance(krd, dict)
        nonzero = [v for v in krd.values() if abs(v) > 1e-12]
        assert len(nonzero) > 0
