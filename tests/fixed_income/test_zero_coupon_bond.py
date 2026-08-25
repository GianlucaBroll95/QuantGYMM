"""
Tests for ZeroCouponBond class.

Covers: initialization, descriptor validation, pricing (risk-free and
risk-adjusted), duration, sensitivity, key_rate_dv01, and error handling.
"""
import numpy as np
import pandas as pd
import pytest
from pandas.tseries.offsets import DateOffset

from QuantGYMM.fixed_income.instruments import ZeroCouponBond
from QuantGYMM.term_structures import SpotRateCurve, DiscountCurve
from QuantGYMM.utils import accrual_factor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TRADE_DATE = pd.Timestamp("2022-01-03")
EVAL_DATE  = pd.Timestamp("2022-01-03")
MATURITY   = pd.Timestamp("2027-01-04")   # ~5Y zero coupon
FLAT_RATE  = 0.03                          # 3% flat curve, continuous compounding

TENORS_YEARS = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
MATURITIES   = [TRADE_DATE + DateOffset(months=round(t * 12)) for t in TENORS_YEARS]

_spot_df = pd.DataFrame({"spotRate": [FLAT_RATE] * len(MATURITIES)},
                        index=pd.DatetimeIndex(MATURITIES))


@pytest.fixture
def spot_curve():
    return SpotRateCurve(_spot_df.copy(), TRADE_DATE, dcc="ACT/365",
                         interpolation="linear", compounding="continuous")


@pytest.fixture
def discount_curve(spot_curve):
    return DiscountCurve(spot_curve, compounding="continuous", dcc="ACT/365",
                         interpolation="linear")


@pytest.fixture
def zcb(discount_curve):
    b = ZeroCouponBond(maturity_date=MATURITY, face_amount=100.0)
    b.set_evaluation_date(EVAL_DATE)
    b.set_discount_curve(discount_curve)
    return b


# ---------------------------------------------------------------------------
# Initialization and descriptor validation
# ---------------------------------------------------------------------------

class TestInitialization:

    def test_valid_construction(self):
        b = ZeroCouponBond(maturity_date="2027-01-04", face_amount=100.0)
        assert b.face_amount == 100.0
        assert b.maturity_date == pd.Timestamp("2027-01-04")

    def test_default_currency_is_eur(self):
        b = ZeroCouponBond(maturity_date="2027-01-04", face_amount=100.0)
        assert b.currency == "EUR"

    def test_custom_currency(self):
        b = ZeroCouponBond(maturity_date="2027-01-04", face_amount=100.0, currency="JPY")
        assert b.currency == "JPY"

    def test_repr(self, zcb):
        assert "ZeroCouponBond" in repr(zcb)
        assert "2027-01-04" in repr(zcb)

    def test_negative_face_amount_rejected(self):
        with pytest.raises(TypeError):
            ZeroCouponBond(maturity_date=MATURITY, face_amount=-100.0)

    def test_invalid_maturity_rejected(self):
        with pytest.raises(TypeError):
            ZeroCouponBond(maturity_date="not a date", face_amount=100.0)

    def test_errors_before_setup(self):
        b = ZeroCouponBond(maturity_date=MATURITY, face_amount=100.0)
        with pytest.raises(ValueError):
            _ = b.evaluation_date
        with pytest.raises(ValueError):
            _ = b.discount_curve
        with pytest.raises(ValueError):
            b.set_discount_curve("not a curve")


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

class TestPricing:

    def test_price_equals_discounted_face(self, zcb, discount_curve):
        p = zcb.prices()["riskFreeValue"]
        df = discount_curve.discount_factor_at(pd.DatetimeIndex([MATURITY])).item()
        assert abs(p["dirtyPrice"] - 100.0 * df) < 1e-10

    def test_no_accrued_interest(self, zcb):
        p = zcb.prices()["riskFreeValue"]
        assert p["accruedInterest"] == 0.0
        assert p["cleanPrice"] == p["dirtyPrice"]

    def test_price_below_face_with_positive_rates(self, zcb):
        assert zcb.prices()["riskFreeValue"]["dirtyPrice"] < 100.0

    def test_risk_adjusted_below_risk_free(self, zcb):
        zcb.set_cds_spread(0.02)
        zcb.set_recovery_rate(0.4)
        p = zcb.prices()
        assert "riskAdjustedValue" in p
        assert p["riskAdjustedValue"]["dirtyPrice"] < p["riskFreeValue"]["dirtyPrice"]

    def test_cds_without_recovery_raises(self, zcb):
        zcb.set_cds_spread(0.02)
        with pytest.raises(ValueError):
            zcb.prices()


# ---------------------------------------------------------------------------
# Duration and sensitivity
# ---------------------------------------------------------------------------

class TestRiskMeasures:

    def test_duration_equals_time_to_maturity(self, zcb, discount_curve):
        expected = accrual_factor(discount_curve.dcc, EVAL_DATE,
                                  pd.DatetimeIndex([MATURITY])).item()
        assert abs(zcb.duration() - expected) < 1e-10

    def test_parallel_dv01_negative(self, zcb):
        # Long a zero coupon: higher rates → lower price.
        assert zcb.sensitivity(shift_type="parallel", shift_size=0.0001) < 0

    def test_modified_duration_close_to_maturity(self, zcb):
        # Continuous compounding, flat curve: ModDur ≈ time to maturity.
        mod = zcb.modified_duration()
        assert abs(mod - zcb.duration()) < 0.1

    def test_key_rate_dv01_sums_to_parallel(self, zcb):
        krd = zcb.key_rate_dv01(kind="symmetric")
        total = sum(krd.values())
        parallel = zcb.sensitivity(shift_type="parallel", shift_size=0.0001)
        assert abs(total - parallel) < abs(parallel) * 0.05

    def test_key_rate_dv01_concentrated_around_maturity(self, zcb):
        # With linear interpolation only the two nodes bracketing the maturity matter.
        krd = zcb.key_rate_dv01(kind="symmetric")
        sorted_nodes = sorted(krd, key=lambda d: abs(krd[d]), reverse=True)
        assert all(abs((d - MATURITY).days) < 800 for d in sorted_nodes[:2])

    def test_invalid_shift_type_raises(self, zcb):
        with pytest.raises(ValueError):
            zcb.sensitivity(shift_type="twist")
        with pytest.raises(ValueError):
            zcb.sensitivity(kind="threeside")
        with pytest.raises(ValueError):
            zcb.key_rate_dv01(kind="threeside")

    def test_curve_restored_after_sensitivity(self, zcb, discount_curve):
        original = discount_curve.rate_curve.spot_rates_data.copy()
        zcb.sensitivity(shift_type="parallel")
        zcb.key_rate_dv01(kind="oneside")
        pd.testing.assert_frame_equal(discount_curve.rate_curve.spot_rates_data, original)
