"""
Tests for the Hull-White trinomial tree and Bermudan callable valuation.

The tree earns its keep by handling what the Jamshidian decomposition cannot:
several exercise dates, and a call price that changes date by date. No closed
form exists for those, so the tree is pinned down two ways - it must reproduce
the closed form where one exists (a single call), and it must satisfy the
structural properties that hold for any correct lattice.
"""
import numpy as np
import pandas as pd
import pytest

from QuantGYMM.calendar import Schedule
from QuantGYMM.fixed_income.instruments import CallableBond, FixedRateBond, FloatingRateBond
from QuantGYMM.fixed_income.models import hw_bermudan_bond_value, hw_trinomial_tree
from QuantGYMM.fixed_income.pricers import Pricer
from QuantGYMM.term_structures import DiscountCurve, SpotRateCurve

TRADE_DATE = pd.Timestamp("2026-06-30")
MATURITY = pd.Timestamp("2034-06-30")
_NODES = pd.DatetimeIndex([TRADE_DATE + pd.DateOffset(years=y) for y in (1, 2, 5, 10, 20, 30, 50)])
_RATES = [0.024, 0.025, 0.027, 0.030, 0.033, 0.035, 0.035]


def _curve(t, level=0.03, slope=0.004):
    """Upward sloping test curve, so nothing hides behind a flat term structure."""
    return np.exp(-(level + slope * np.sqrt(t)) * t)


@pytest.fixture
def dc():
    src = SpotRateCurve(pd.DataFrame({"spotRate": _RATES}, index=_NODES),
                        TRADE_DATE, interpolation="linear")
    return DiscountCurve(src, compounding="annually_compounded", dcc="ACT/365",
                         interpolation="linear")


@pytest.fixture
def bond(dc):
    def build():
        b = FixedRateBond(
            schedule=Schedule(start_date=TRADE_DATE - pd.DateOffset(months=12),
                              end_date=MATURITY, frequency=2, eom=False),
            dcc="ACT/ACT ICMA", face_amount=100.0, coupon_rate=0.045, redemption=100.0)
        b.set_evaluation_date(TRADE_DATE)
        b.set_discount_curve(dc)
        return b
    return build


@pytest.fixture
def call_dates(bond):
    payments = pd.DatetimeIndex(bond().schedule.schedule["paymentDate"])
    return payments[(payments > pd.Timestamp("2030-01-01")) & (payments < MATURITY)]


# --- the lattice itself -----------------------------------------------------

@pytest.mark.parametrize("a, sigma, dt, n", [
    (0.03, 0.010, 0.25, 40), (0.10, 0.020, 0.50, 60),
    (0.05, 0.015, 0.10, 200), (0.02, 0.030, 0.25, 120),
])
def test_probabilities_are_positive_and_sum_to_one(a, sigma, dt, n):
    """The branching switch at the edges exists precisely to keep them positive."""
    tree = hw_trinomial_tree(dt, n, a, sigma, _curve(np.arange(n + 1) * dt))
    for probs in tree["probs"]:
        assert probs.min() > 0
        np.testing.assert_allclose(probs.sum(axis=0), 1.0, atol=1e-12)


@pytest.mark.parametrize("a, sigma, dt, n", [
    (0.03, 0.010, 0.25, 40), (0.10, 0.020, 0.50, 60), (0.05, 0.015, 0.10, 200),
])
def test_the_tree_reprices_the_input_curve(a, sigma, dt, n):
    """Second stage of the construction: alpha is solved for exactly this."""
    discount_factors = _curve(np.arange(n + 1) * dt)
    tree = hw_trinomial_tree(dt, n, a, sigma, discount_factors)
    zero_coupon = np.zeros(n + 1)
    zero_coupon[-1] = 1.0
    assert hw_bermudan_bond_value(tree, dt, zero_coupon) == pytest.approx(
        discount_factors[-1], abs=1e-12)


def test_a_bond_without_calls_is_just_its_discounted_cash_flows():
    dt, n, a, sigma = 0.5, 20, 0.05, 0.015
    discount_factors = _curve(np.arange(n + 1) * dt)
    tree = hw_trinomial_tree(dt, n, a, sigma, discount_factors)
    cash_flows = np.zeros(n + 1)
    cash_flows[1:] = 2.0
    cash_flows[-1] += 100.0
    exact = (cash_flows[1:] * discount_factors[1:]).sum()
    assert hw_bermudan_bond_value(tree, dt, cash_flows) == pytest.approx(exact, abs=1e-10)


def test_the_tree_widens_as_the_step_shrinks():
    """j_max = ceil(0.184 / (a dt)): halving the step doubles the width, and the
    cost with it. It is why a daily grid is expensive, not merely slow."""
    wide = hw_trinomial_tree(0.05, 20, 0.03, 0.01, _curve(np.arange(21) * 0.05))
    narrow = hw_trinomial_tree(0.50, 20, 0.03, 0.01, _curve(np.arange(21) * 0.50))
    assert wide["j_max"] > narrow["j_max"]


@pytest.mark.parametrize("a, sigma", [(0.0, 0.01), (0.03, 0.0), (-0.03, 0.01)])
def test_a_non_positive_parameter_is_refused(a, sigma):
    with pytest.raises(ValueError, match="strictly positive"):
        hw_trinomial_tree(0.5, 4, a, sigma, _curve(np.arange(5) * 0.5))


def test_a_mismatched_curve_length_is_refused():
    with pytest.raises(ValueError, match="discount_factors"):
        hw_trinomial_tree(0.5, 4, 0.03, 0.01, _curve(np.arange(3) * 0.5))


# --- against the closed form ------------------------------------------------

def test_a_single_call_converges_to_the_jamshidian_value(bond, call_dates):
    """The only case where an exact answer exists. Matching it here is what
    licenses trusting the tree where no closed form is available."""
    callable_bond = CallableBond(bond(), pd.Series([100.0], index=call_dates[:1]),
                                 volatility=0.01)
    closed_form = callable_bond.prices(method="hw")["optionValue"]
    assert callable_bond._option_value_tree(step_days=7) == pytest.approx(closed_form, rel=2e-3)


def test_refining_the_grid_moves_towards_the_closed_form(bond, call_dates):
    callable_bond = CallableBond(bond(), pd.Series([100.0], index=call_dates[:1]),
                                 volatility=0.01)
    closed_form = callable_bond.prices(method="hw")["optionValue"]
    coarse = abs(callable_bond._option_value_tree(step_days=30) / closed_form - 1)
    fine = abs(callable_bond._option_value_tree(step_days=7) / closed_form - 1)
    assert fine < coarse


# --- properties that must hold with no closed form to lean on ---------------

def test_more_exercise_dates_can_only_be_worth_more(bond, call_dates):
    """A Bermudan contains every European inside it, so its value cannot fall
    when a date is added. It is the property the single-call model violates."""
    values = [CallableBond(bond(), pd.Series([100.0] * n, index=call_dates[:n]),
                           volatility=0.01).prices(method="tree")["optionValue"]
              for n in (1, 2, 4, len(call_dates))]
    assert values == sorted(values)
    assert values[-1] > values[0], "the Bermudan premium cannot be zero"


def test_the_callable_is_worth_less_than_the_straight_bond(bond, call_dates):
    """The holder has sold an option: the value stays capped."""
    underlying = bond()
    straight = underlying.prices()["riskFreeValue"]["dirtyPrice"]
    priced = CallableBond(underlying, pd.Series([100.0] * len(call_dates), index=call_dates),
                          volatility=0.01).prices(method="tree")
    assert priced["callableValue"]["dirtyPrice"] < straight
    assert priced["optionValue"] > 0
    assert priced["callableValue"]["dirtyPrice"] == pytest.approx(
        straight - priced["optionValue"])


def test_a_higher_strike_makes_the_option_cheaper(bond, call_dates):
    """The declining premium schedule of high yield paper, which the closed form
    cannot express at all."""
    at_par = CallableBond(bond(), pd.Series([100.0] * len(call_dates), index=call_dates),
                          volatility=0.01).prices(method="tree")["optionValue"]
    premium = pd.Series(np.maximum(100.0, 102.5 - 0.5 * np.arange(len(call_dates))),
                        index=call_dates)
    above_par = CallableBond(bond(), premium, volatility=0.01).prices(method="tree")["optionValue"]
    assert above_par < at_par


def test_more_volatility_makes_the_option_dearer(bond, call_dates):
    schedule = pd.Series([100.0] * len(call_dates), index=call_dates)
    quiet = CallableBond(bond(), schedule, volatility=0.005).prices(method="tree")["optionValue"]
    loud = CallableBond(bond(), schedule, volatility=0.02).prices(method="tree")["optionValue"]
    assert loud > quiet


# --- scope ------------------------------------------------------------------

def test_a_floating_underlying_is_refused(dc, call_dates):
    floater = FloatingRateBond(
        schedule=Schedule(start_date=TRADE_DATE - pd.DateOffset(months=12),
                          end_date=MATURITY, frequency=2, eom=False),
        dcc="ACT/360", face_amount=100.0, fixing_days=0, spread=0.005)
    floater.set_pricer(Pricer(dc))
    floater.set_evaluation_date(TRADE_DATE)
    floater.set_current_coupon_rate(0.029)
    callable_bond = CallableBond(floater, pd.Series([100.0], index=call_dates[:1]),
                                 volatility=0.01)
    with pytest.raises(NotImplementedError, match="fixed rate"):
        callable_bond.prices(method="tree")


def test_an_unknown_method_is_refused(bond, call_dates):
    callable_bond = CallableBond(bond(), pd.Series([100.0], index=call_dates[:1]),
                                 volatility=0.01)
    with pytest.raises(ValueError, match="worst"):
        callable_bond.prices(method="binomial")
