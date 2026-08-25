"""
Tests for FixedRateBond class.

Covers: initialization, descriptor validation, cash flow generation, pricing (risk-free
and risk-adjusted), coupon history, sensitivity, key_rate_dv01, and error handling.
"""
import numpy as np
import pandas as pd
import pytest
from pandas.tseries.offsets import DateOffset

from QuantGYMM.calendar import Schedule
from QuantGYMM.fixed_income.instruments import FixedRateBond
from QuantGYMM.term_structures import SpotRateCurve, DiscountCurve


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Curve trade date and bond issuance aligned — eval date mid-life
TRADE_DATE = pd.Timestamp("2020-01-02")
EVAL_DATE  = pd.Timestamp("2022-06-15")   # 2.5Y into a 5Y bond → 3 future cash flows

TENORS_YEARS = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
MATURITIES   = [TRADE_DATE + DateOffset(months=round(t * 12)) for t in TENORS_YEARS]
FLAT_RATE    = 0.03   # 3% flat curve, continuously compounded

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
def schedule_5y():
    """5Y annual coupon schedule, no stub, issued 2020-01-02."""
    return Schedule(start_date=pd.Timestamp("2020-01-02"),
                    end_date=pd.Timestamp("2025-01-02"),
                    frequency=1)


@pytest.fixture
def bond_5y(schedule_5y, discount_curve):
    """Plain vanilla 5Y fixed rate bond, 3% annual coupon, face 100.
    Evaluated at 2022-06-15: 2 past coupons (2021, 2022), 3 future (2023, 2024, 2025)."""
    b = FixedRateBond(schedule=schedule_5y, dcc="ACT/365",
                      face_amount=100.0, coupon_rate=0.03)
    b.set_evaluation_date(EVAL_DATE)
    b.set_discount_curve(discount_curve)
    return b


# ---------------------------------------------------------------------------
# Initialization and descriptor validation
# ---------------------------------------------------------------------------

class TestInitialization:

    def test_valid_construction(self, schedule_5y):
        b = FixedRateBond(schedule=schedule_5y, dcc="ACT/365",
                          face_amount=100.0, coupon_rate=0.03)
        assert b.face_amount == 100.0
        assert b.coupon_rate == 0.03
        assert b.redemption == 100.0   # default

    def test_custom_redemption(self, schedule_5y):
        b = FixedRateBond(schedule=schedule_5y, dcc="ACT/365",
                          face_amount=100.0, coupon_rate=0.03, redemption=101.0)
        assert b.redemption == 101.0

    def test_default_currency_is_eur(self, schedule_5y):
        b = FixedRateBond(schedule=schedule_5y, dcc="ACT/365",
                          face_amount=100.0, coupon_rate=0.03)
        assert b.currency == "EUR"

    def test_custom_currency(self, schedule_5y):
        b = FixedRateBond(schedule=schedule_5y, dcc="ACT/365",
                          face_amount=100.0, coupon_rate=0.03, currency="USD")
        assert b.currency == "USD"

    @pytest.mark.parametrize("currency", [1, False, ["USD"]])
    def test_invalid_currency_raises(self, schedule_5y, currency):
        with pytest.raises(TypeError):
            FixedRateBond(schedule=schedule_5y, dcc="ACT/365",
                          face_amount=100.0, coupon_rate=0.03, currency=currency)

    def test_invalid_face_amount_raises(self, schedule_5y):
        with pytest.raises(TypeError):
            FixedRateBond(schedule=schedule_5y, dcc="ACT/365",
                          face_amount=-100.0, coupon_rate=0.03)

    def test_invalid_dcc_raises(self, schedule_5y):
        with pytest.raises(ValueError):
            FixedRateBond(schedule=schedule_5y, dcc="INVALID/DCC",
                          face_amount=100.0, coupon_rate=0.03)

    def test_invalid_schedule_raises(self):
        with pytest.raises(ValueError):
            FixedRateBond(schedule="not_a_schedule", dcc="ACT/365",
                          face_amount=100.0, coupon_rate=0.03)

    def test_repr(self, bond_5y):
        r = repr(bond_5y)
        assert "FixedRateBond" in r
        assert "couponRate=0.03" in r

    @pytest.mark.parametrize("dcc", ["ACT/360", "ACT/365", "ACT/ACT", "ACT/ACT ISDA",
                                     "ACT/ACT ICMA", "30/360", "30E/360", "NL/365"])
    def test_all_valid_dcc(self, schedule_5y, dcc):
        b = FixedRateBond(schedule=schedule_5y, dcc=dcc,
                          face_amount=100.0, coupon_rate=0.03)
        assert b.dcc == dcc


# ---------------------------------------------------------------------------
# Evaluation date and discount curve setters
# ---------------------------------------------------------------------------

class TestSetters:

    def test_set_evaluation_date(self, schedule_5y):
        b = FixedRateBond(schedule=schedule_5y, dcc="ACT/365",
                          face_amount=100.0, coupon_rate=0.03)
        b.set_evaluation_date("2022-06-15")
        assert b.evaluation_date == pd.Timestamp("2022-06-15")

    def test_evaluation_date_invalidates_cash_flows(self, bond_5y):
        _ = bond_5y.cash_flows   # trigger cache
        assert bond_5y._cash_flows is not None
        bond_5y.set_evaluation_date("2023-06-01")
        assert bond_5y._cash_flows is None

    def test_evaluation_date_not_set_raises(self, schedule_5y):
        b = FixedRateBond(schedule=schedule_5y, dcc="ACT/365",
                          face_amount=100.0, coupon_rate=0.03)
        with pytest.raises(ValueError, match="set_evaluation_date"):
            _ = b.evaluation_date

    def test_invalid_evaluation_date_raises(self, schedule_5y):
        b = FixedRateBond(schedule=schedule_5y, dcc="ACT/365",
                          face_amount=100.0, coupon_rate=0.03)
        with pytest.raises(ValueError):
            b.set_evaluation_date("not_a_date")

    def test_discount_curve_not_set_raises(self, schedule_5y):
        b = FixedRateBond(schedule=schedule_5y, dcc="ACT/365",
                          face_amount=100.0, coupon_rate=0.03)
        with pytest.raises(ValueError, match="set_discount_curve"):
            _ = b.discount_curve

    def test_invalid_discount_curve_raises(self, schedule_5y):
        b = FixedRateBond(schedule=schedule_5y, dcc="ACT/365",
                          face_amount=100.0, coupon_rate=0.03)
        b.set_evaluation_date(EVAL_DATE)
        with pytest.raises(ValueError):
            b.set_discount_curve("not_a_curve")


# ---------------------------------------------------------------------------
# Cash flows
# ---------------------------------------------------------------------------

class TestCashFlows:

    def test_cash_flows_count(self, bond_5y):
        """At EVAL_DATE = 2022-06-15, future payments are 2023, 2024, 2025 → 3 cash flows."""
        cf = bond_5y.cash_flows
        assert len(cf) == 3

    def test_redemption_in_last_cash_flow(self, bond_5y):
        cf = bond_5y.cash_flows
        # last cash flow must include redemption (100) on top of coupon (~3)
        assert cf["cashFlow"].iloc[-1] > 100.0

    def test_cash_flows_cached(self, bond_5y):
        cf1 = bond_5y.cash_flows
        cf2 = bond_5y.cash_flows
        assert cf1 is cf2   # same object, not recomputed

    def test_cash_flows_invalidated_on_schedule_change(self, bond_5y):
        _ = bond_5y.cash_flows
        new_sched = Schedule(start_date=pd.Timestamp("2020-01-02"),
                             end_date=pd.Timestamp("2026-01-02"), frequency=1)
        bond_5y.schedule = new_sched
        assert bond_5y._cash_flows is None


# ---------------------------------------------------------------------------
# Coupon history
# ---------------------------------------------------------------------------

class TestCouponHistory:

    def test_coupons_history_count(self, bond_5y):
        """At EVAL_DATE = 2022-06-15, past payments are 2021-01-02 and 2022-01-02 → 2 coupons."""
        hist = bond_5y.coupons_history
        assert len(hist) == 2

    def test_coupons_history_deterministic(self, bond_5y):
        """Fixed rate bond: cedole devono essere positive e proporzionali al coupon rate."""
        hist = bond_5y.coupons_history
        coupons = hist["cashFlow"].values
        assert all(c > 0 for c in coupons)
        assert all(abs(c - bond_5y.coupon_rate * bond_5y.face_amount) < 0.5 for c in coupons)

    def test_coupons_history_cached(self, bond_5y):
        h1 = bond_5y.coupons_history
        h2 = bond_5y.coupons_history
        assert h1 is h2

    def test_coupon_history_invalidated_on_eval_date_change(self, bond_5y):
        _ = bond_5y.coupons_history
        bond_5y.set_evaluation_date(pd.Timestamp("2023-06-01"))
        assert bond_5y._coupons_history is None

    def test_coupons_history_empty_before_first_payment(self, schedule_5y, discount_curve):
        b = FixedRateBond(schedule=schedule_5y, dcc="ACT/365",
                          face_amount=100.0, coupon_rate=0.03)
        b.set_evaluation_date(pd.Timestamp("2020-06-01"))   # before first payment
        b.set_discount_curve(discount_curve)
        assert len(b.coupons_history) == 0


# ---------------------------------------------------------------------------
# Pricing — risk-free
# ---------------------------------------------------------------------------

class TestPricingRiskFree:

    def test_premium_bond_when_coupon_above_market(self, schedule_5y, spot_curve):
        """High coupon bond should price above par."""
        dc = DiscountCurve(spot_curve, compounding="continuous", dcc="ACT/365",
                           interpolation="linear")
        b = FixedRateBond(schedule=schedule_5y, dcc="ACT/365",
                          face_amount=100.0, coupon_rate=0.08)   # 8% vs 3% market
        b.set_evaluation_date(EVAL_DATE)
        b.set_discount_curve(dc)
        price = b.prices()["riskFreeValue"]["dirtyPrice"]
        assert price > 100.0

    def test_discount_bond_when_coupon_below_market(self, schedule_5y, spot_curve):
        """Low coupon bond should price below par."""
        dc = DiscountCurve(spot_curve, compounding="continuous", dcc="ACT/365",
                           interpolation="linear")
        b = FixedRateBond(schedule=schedule_5y, dcc="ACT/365",
                          face_amount=100.0, coupon_rate=0.01)   # 1% vs 3% market
        b.set_evaluation_date(EVAL_DATE)
        b.set_discount_curve(dc)
        price = b.prices()["riskFreeValue"]["dirtyPrice"]
        assert price < 100.0

    def test_prices_returns_expected_keys(self, bond_5y):
        p = bond_5y.prices()
        assert "riskFreeValue" in p
        assert "dirtyPrice" in p["riskFreeValue"]
        assert "cleanPrice" in p["riskFreeValue"]
        assert "accruedInterest" in p["riskFreeValue"]

    def test_dirty_equals_clean_plus_accrued(self, bond_5y):
        p = bond_5y.prices()["riskFreeValue"]
        assert abs(p["dirtyPrice"] - p["cleanPrice"] - p["accruedInterest"]) < 1e-10


# ---------------------------------------------------------------------------
# Pricing — risk-adjusted (CDS)
# ---------------------------------------------------------------------------

class TestPricingRiskAdjusted:

    def test_risk_adjusted_price_lower_than_risk_free(self, bond_5y):
        """CDS spread > 0 should push risk-adjusted price below risk-free price."""
        bond_5y.set_cds_spread(0.02)
        bond_5y.set_recovery_rate(0.4)
        p = bond_5y.prices()
        assert p["riskAdjustedValue"]["dirtyPrice"] < p["riskFreeValue"]["dirtyPrice"]

    def test_risk_adjusted_not_in_prices_without_cds(self, bond_5y):
        assert "riskAdjustedValue" not in bond_5y.prices()

    def test_cds_without_recovery_raises(self, bond_5y):
        bond_5y.set_cds_spread(0.02)
        with pytest.raises(ValueError, match="recovery"):
            bond_5y.prices()

    def test_invalid_cds_spread_raises(self, bond_5y):
        with pytest.raises(ValueError):
            bond_5y.set_cds_spread("not_a_float")

    def test_invalid_recovery_rate_raises(self, bond_5y):
        with pytest.raises(ValueError):
            bond_5y.set_recovery_rate("not_a_float")


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------

class TestSensitivity:

    def test_parallel_sensitivity_negative(self, bond_5y):
        """DV01 for a long bond position should be negative (price falls when rates rise)."""
        dv01 = bond_5y.sensitivity(shift_type="parallel")
        assert dv01 < 0

    @pytest.mark.parametrize("shift_type", ["parallel", "slope", "curvature"])
    def test_sensitivity_returns_float(self, bond_5y, shift_type):
        result = bond_5y.sensitivity(shift_type=shift_type)
        assert isinstance(result, float)

    @pytest.mark.parametrize("kind", ["symmetric", "oneside"])
    def test_sensitivity_both_kinds(self, bond_5y, kind):
        result = bond_5y.sensitivity(kind=kind)
        assert isinstance(result, float)

    def test_symmetric_same_sign_as_oneside(self, bond_5y):
        sym = bond_5y.sensitivity(kind="symmetric")
        one = bond_5y.sensitivity(kind="oneside")
        assert np.sign(sym) == np.sign(one)

    def test_invalid_shift_type_raises(self, bond_5y):
        with pytest.raises(ValueError, match="shift type"):
            bond_5y.sensitivity(shift_type="invalid")

    def test_invalid_kind_raises(self, bond_5y):
        with pytest.raises(ValueError, match="kind"):
            bond_5y.sensitivity(kind="invalid")

    def test_curve_restored_after_sensitivity(self, bond_5y):
        """Curve should not be left in shifted state after sensitivity call."""
        price_before = bond_5y.prices()["riskFreeValue"]["dirtyPrice"]
        bond_5y.sensitivity(shift_type="parallel")
        price_after = bond_5y.prices()["riskFreeValue"]["dirtyPrice"]
        assert abs(price_before - price_after) < 1e-10


# ---------------------------------------------------------------------------
# Key Rate DV01
# ---------------------------------------------------------------------------

class TestKeyRateDV01:

    def test_returns_dict(self, bond_5y):
        krd = bond_5y.key_rate_dv01()
        assert isinstance(krd, dict)

    def test_keys_are_timestamps(self, bond_5y):
        krd = bond_5y.key_rate_dv01()
        assert all(isinstance(k, pd.Timestamp) for k in krd.keys())

    def test_zero_beyond_maturity(self, bond_5y):
        """Nodes beyond the bond maturity (2025-01-02) should have zero DV01
        with linear interpolation."""
        krd = bond_5y.key_rate_dv01()
        maturity = pd.Timestamp("2025-01-02")
        for date, val in krd.items():
            if date > maturity:
                assert val == 0.0, f"Expected 0 at {date}, got {val}"

    def test_nonzero_nodes_negative(self, bond_5y):
        """All non-zero DV01 values should be negative (long bond, rates up → price down)."""
        krd = bond_5y.key_rate_dv01()
        nonzero = [v for v in krd.values() if v != 0.0]
        assert len(nonzero) > 0
        assert all(v < 0 for v in nonzero)

    def test_sum_matches_parallel_dv01(self, bond_5y):
        """Sum of KRD DV01 should match the parallel DV01 within tolerance."""
        krd_sum = sum(bond_5y.key_rate_dv01().values())
        parallel = bond_5y.sensitivity(shift_type="parallel")
        assert abs(krd_sum - parallel) < abs(parallel) * 0.02   # within 2%

    def test_curve_restored_after_krd(self, bond_5y):
        """Curve must be fully restored after key_rate_dv01."""
        price_before = bond_5y.prices()["riskFreeValue"]["dirtyPrice"]
        bond_5y.key_rate_dv01()
        price_after = bond_5y.prices()["riskFreeValue"]["dirtyPrice"]
        assert abs(price_before - price_after) < 1e-10

    @pytest.mark.parametrize("kind", ["symmetric", "oneside"])
    def test_both_kinds(self, bond_5y, kind):
        krd = bond_5y.key_rate_dv01(kind=kind)
        assert isinstance(krd, dict)
        assert len(krd) > 0

    def test_invalid_kind_raises(self, bond_5y):
        with pytest.raises(ValueError, match="kind"):
            bond_5y.key_rate_dv01(kind="invalid")