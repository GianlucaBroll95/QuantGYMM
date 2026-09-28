import pandas as pd
import numpy as np
import scipy
from scipy.stats import norm
from .curves import YieldCurve
from ..utils import accrual_factor

__all__ = ["Pricer", "BlackCouponPricer", "BachelierCouponPricer", "DisplacedBlackCouponPricer"]


class Pricer:
    """
    Base Pricer object.
    """

    def __init__(self, discount_curve):
        """
        Args:
            discount_curve (YieldCurve): curve the cash flows are discounted on.
        """

        if not isinstance(discount_curve, YieldCurve):
            raise ValueError("Pricer needs a YieldCurve object.")
        self.discount_curve = discount_curve

    def discount_factor_at(self, bond, dates):
        curve = self.discount_curve
        t = accrual_factor(curve.dcc, curve.trade_date, dates)
        return curve.discount_factor_at(dates) * np.exp(-bond.z_spread * t)

    def present_value(self, bond, date=None) -> float:
        """
        Value of the cash flows paid after 'date', at 'date'.
        Args:
            bond (Bond): bond to price, evaluated on the curve's trade date.
            date (pandas.Timestamp): [optional] date to value at, defaults to the curve's trade date.
        Returns:
            float
        """
        if bond.evaluation_date != self.discount_curve.trade_date:
            raise ValueError(f"The bond is evaluated on {bond.evaluation_date.date()}, the discount curve is built "
                             f"on {self.discount_curve.trade_date.date()}.")
        cash_flows = bond.cash_flows
        df = self.discount_factor_at(bond, cash_flows.paymentDate)
        survival = None if bond._cds_spread is None else bond.survival_probabilities
        recovery = None if survival is None else np.broadcast_to(np.asarray(bond.recovery_rate, dtype=float),
                                                                 survival.shape)
        if date is not None:
            after = (cash_flows.paymentDate > date).to_numpy()
            cash_flows, df = cash_flows[after], df[after] / self.discount_factor_at(bond, [date])[0]
            if survival is not None:
                survival, recovery = (survival / bond._survival_at(date))[after], recovery[after]
        if survival is None:
            return float(cash_flows.cashFlow.to_numpy() @ df)
        delta_prob = np.diff(-survival, prepend=-1)
        coupon_leg = (cash_flows.coupon.to_numpy() * survival) @ df
        redemption_leg = (cash_flows.redemption.to_numpy() * survival) @ df
        default_leg = (recovery * delta_prob) @ df * bond.face_amount
        return float(coupon_leg + redemption_leg + default_leg)


class BlackCouponPricer:
    """
    Caplet and floorlet premiums under the lognormal Black model.
    """

    _DCC = "ACT/365"

    def __init__(self, volatility_surface):
        """
        Args:
            volatility_surface (pandas.DataFrame): volatility surface, maturities on the index
                                                   and strikes on the columns.
        """
        self.volatility_surface = volatility_surface

    @property
    def volatility_surface(self):
        return self._volatility_surface

    @volatility_surface.setter
    def volatility_surface(self, volatility_surface):
        if isinstance(volatility_surface, pd.DataFrame):
            self._volatility_surface = volatility_surface
        else:
            raise ValueError("Volatility surface should be a DataFrame.")

    def _unfixed_periods(self, bond, resets, af, rates):
        """
        Time to fixing, accrual factor and underlying rate of the periods not yet fixed.
        Args:
            bond (FloatingRateBond): bond whose coupons are being priced.
            resets (numpy.ndarray): fixing dates.
            af (numpy.ndarray): accrual factors of the coupons.
            rates (numpy.ndarray): projected index rates.
        Returns:
            tuple of numpy.ndarray.
        """
        return accrual_factor(self._DCC, bond.evaluation_date, resets), af, rates + bond.spread

    def _volatilities(self, bond, cap_strike, floor_strike):
        maturity = (bond.schedule.schedule["paymentDate"][-1] - bond.evaluation_date).days / 365
        interpolator = scipy.interpolate.RegularGridInterpolator(
            (self.volatility_surface.index, self.volatility_surface.columns), self.volatility_surface.values,
            bounds_error=False, fill_value=None)  # extrapolate values outside bounds
        return interpolator([(maturity, cap_strike), (maturity, floor_strike)])

    def forward_premiums(self, bond, resets, af, rates):
        """
        Caplet and floorlet forward premiums, one per period still to be fixed.
        Args:
            bond (FloatingRateBond): bond whose coupons are being priced.
            resets (numpy.ndarray): fixing dates.
            af (numpy.ndarray): accrual factors of the coupons.
            rates (numpy.ndarray): projected index rates.
        Returns:
            tuple of numpy.ndarray, (caplet, floorlet).
        """
        ttm, af, underlying_rate = self._unfixed_periods(bond, resets, af, rates)
        cap_vol, floor_vol = self._volatilities(bond, bond.cap, bond.floor)

        # The lognormal Black model cannot price negative strikes; a strike of
        # exactly 0 is handled below (log→±inf gives the correct degenerate
        # premium: worthless floor, always-in-the-money cap).
        for name, strike in (("cap", bond.cap), ("floor", bond.floor)):
            if not np.isnan(strike) and strike < 0:
                raise ValueError(
                    f"Black model requires a non-negative {name} strike; got {strike}. "
                    "Use BachelierCouponPricer (normal model) for negative strikes."
                )

        # d1 and d2 (errstate: log(rate/0) → inf is the intended limit, not an error):
        with np.errstate(divide="ignore"):
            d1_cap = (np.log(underlying_rate / bond.cap) + 0.5 * ttm * cap_vol ** 2) / (cap_vol * ttm ** 0.5)
            d2_cap = (np.log(underlying_rate / bond.cap) - 0.5 * ttm * cap_vol ** 2) / (cap_vol * ttm ** 0.5)
            d1_floor = (np.log(underlying_rate / bond.floor) + 0.5 * ttm * floor_vol ** 2) / (floor_vol * ttm ** 0.5)
            d2_floor = (np.log(underlying_rate / bond.floor) - 0.5 * ttm * floor_vol ** 2) / (floor_vol * ttm ** 0.5)

        # N(d1) and N(d2)
        nd1_cap, nd2_cap = norm.cdf(d1_cap), norm.cdf(d2_cap)
        nd1_floor, nd2_floor = norm.cdf(-d1_floor), norm.cdf(-d2_floor)

        caplet = (underlying_rate * nd1_cap - bond.cap * nd2_cap) * af * bond.face_amount
        floorlet = (bond.floor * nd2_floor - underlying_rate * nd1_floor) * af * bond.face_amount
        return caplet, floorlet


class BachelierCouponPricer(BlackCouponPricer):
    """
    Caplet and floorlet premiums under the normal model.
    """

    def forward_premiums(self, bond, resets, af, rates):
        ttm, af, underlying_rate = self._unfixed_periods(bond, resets, af, rates)
        cap_vol, floor_vol = self._volatilities(bond, bond.cap, bond.floor)

        # d1 and d2:
        d1_cap = (underlying_rate - bond.cap) / (cap_vol * ttm ** 0.5)
        d1_floor = (underlying_rate - bond.floor) / (floor_vol * ttm ** 0.5)

        # N(d1) and N(d2)
        nd1_cap, small_nd1_cap = norm.cdf(d1_cap), norm.pdf(d1_cap)
        nd1_floor, small_nd1_floor = norm.cdf(-d1_floor), norm.pdf(d1_floor)

        caplet = ((underlying_rate - bond.cap) * nd1_cap +
                  cap_vol * small_nd1_cap * ttm ** 0.5) * af * bond.face_amount
        floorlet = ((bond.floor - underlying_rate) * nd1_floor +
                    floor_vol * small_nd1_floor * ttm ** 0.5) * af * bond.face_amount
        return caplet, floorlet


class DisplacedBlackCouponPricer(BlackCouponPricer):
    """
    Caplet and floorlet premiums under the shifted lognormal model.
    """

    def __init__(self, volatility_surface, shift=0.03):
        """
        Args:
            volatility_surface (pandas.DataFrame): volatility surface for the displaced-Black model
            shift (float): displacement size (default 3%)
        """
        super().__init__(volatility_surface)
        self.shift = shift

    def forward_premiums(self, bond, resets, af, rates):
        """
         Caplet and floorlet forward premiums, one per period still to be fixed.
         Args:
             bond (FloatingRateBond): bond whose coupons are being priced.
             resets (numpy.ndarray): fixing dates.
             af (numpy.ndarray): accrual factors of the coupons.
             rates (numpy.ndarray): projected index rates.
         Returns:
             tuple of numpy.ndarray, (caplet, floorlet).
         """
        cap_strike = self.shift + bond.cap
        floor_strike = self.shift + bond.floor
        ttm, af, underlying_rate = self._unfixed_periods(bond, resets, af, rates)
        underlying_rate = underlying_rate + self.shift
        cap_vol, floor_vol = self._volatilities(bond, cap_strike, floor_strike)

        for name, strike in (("shifted cap", cap_strike), ("shifted floor", floor_strike)):
            if not np.isnan(strike) and strike < 0:
                raise ValueError(
                    f"Displaced-Black requires a non-negative {name} strike; got {strike}. "
                    "Increase the displacement or use BachelierCouponPricer."
                )

        with np.errstate(divide="ignore"):
            d1_cap = (np.log(underlying_rate / cap_strike) + 0.5 * ttm * cap_vol ** 2) / (cap_vol * ttm ** 0.5)
            d2_cap = (np.log(underlying_rate / cap_strike) - 0.5 * ttm * cap_vol ** 2) / (cap_vol * ttm ** 0.5)
            d1_floor = (np.log(underlying_rate / floor_strike) + 0.5 * ttm * floor_vol ** 2) / (floor_vol * ttm ** 0.5)
            d2_floor = (np.log(underlying_rate / floor_strike) - 0.5 * ttm * floor_vol ** 2) / (floor_vol * ttm ** 0.5)

        # N(d1) and N(d2)
        nd1_cap, nd2_cap = norm.cdf(d1_cap), norm.cdf(d2_cap)
        nd1_floor, nd2_floor = norm.cdf(-d1_floor), norm.cdf(-d2_floor)

        caplet = (underlying_rate * nd1_cap - cap_strike * nd2_cap) * af * bond.face_amount
        floorlet = (floor_strike * nd2_floor - underlying_rate * nd1_floor) * af * bond.face_amount
        return caplet, floorlet
