from .term_structures import DiscountCurve
import pandas as pd
import numpy as np
import scipy
from scipy.stats import norm
from .utils import *
import warnings
from pandas.tseries.offsets import DateOffset, BDay

__all__ = ["Pricer", "BlackPricer", "BachelierPricer", "DisplacedBlackPricer"]


class Pricer:
    """
    Base Pricer object.
    """

    def __init__(self, discount_curve):
        """
        Args:
            discount_curve (DiscountCurve): DiscountCurve instance.
        """

        self.discount_curve = discount_curve
        self._bond = None
        self._forward_rates = None
        self._current_coupon = None
        self._expected_coupons = None

    @property
    def discount_curve(self):
        return self._discount_curve

    @discount_curve.setter
    def discount_curve(self, discount_curve):
        if isinstance(discount_curve, DiscountCurve):
            self._discount_curve = discount_curve
            self._forward_rates = None
            self._current_coupon_pv = None
            self._expected_coupons = None
        else:
            raise ValueError(
                f"'discount_curve' must be a DiscountCurve object. Got {discount_curve.__class__.__name__}.")

    @property
    def forward_rates(self):
        if self._forward_rates is None or self.discount_curve.shift_flag:
            self._get_forward_rates()
        return self._forward_rates

    @property
    def current_coupon(self):
        if self._current_coupon is None:
            self._get_current_coupon()
        return self._current_coupon

    @property
    def bond(self):
        if self._bond is None:
            raise ValueError("Bond missing.")
        return self._bond

    @property
    def expected_coupons(self):
        if self._expected_coupons is None or self.discount_curve.shift_flag:
            self._get_expected_coupons()
        return self._expected_coupons

    def transfer_bond_features(self, bond) -> None:
        """
        Passes to the pricer the bond characteristics.
        Args:
            bond (Bond): bond on which the pricer needs to be bounded.
        """
        self._bond = bond

    def _get_forward_rates(self):
        reset_dates = self.bond.schedule.schedule["resetDate"]
        future_resets = reset_dates[reset_dates > self.bond.evaluation_date]
        df1 = self.discount_curve.discount_factors.loc[future_resets]
        df2 = self.discount_curve.discount_factors.loc[
            business_adjustment("modified_following",
                                future_resets + DateOffset(months=12 / self.bond.schedule.frequency))]
        af = accrual_factor(self.discount_curve.dcc, df1.index.to_list(), df2.index.to_list())
        match self.discount_curve.compounding:
            case "simple":
                self._forward_rates = ((df1.to_numpy() / df2.to_numpy() - 1) / af.reshape(-1, 1)).squeeze()
                # self._forward_rates = (df1.divide(df2.to_numpy()) - 1).divide(af, axis=0).to_numpy().squeeze()
            case "continuous":
                self._forward_rates = (np.log(df1.to_numpy() / df2.to_numpy()) / af.reshape(-1, 1)).squeeze()
                # self._forward_rates = np.log(df1.divide(df2.to_numpy())).divide(af, axis=0).to_numpy().squeeze()
            case "annually_compounded":
                self._forward_rates = ((df1.to_numpy() / df2.to_numpy()) ** (1 / af.reshape(-1, 1)) - 1).squeeze()
                # self._forward_rates = ((df1.divide(df2.to_numpy())).pow(1 / af, axis=0) - 1).to_numpy().squeeze()

    def _get_current_coupon(self):
        reset_dates = self.bond.schedule.schedule["resetDate"]
        reset = reset_dates[reset_dates <= self.bond.evaluation_date][-1]
        reset_rate = self.bond.historical_euribor.loc[reset]
        start = self.bond.schedule.schedule["startingDate"][reset_dates <= self.bond.evaluation_date][-1]
        end = self.bond.schedule.schedule["paymentDate"][reset_dates <= self.bond.evaluation_date][-1]
        af = accrual_factor(self.bond.dcc, start, end)
        coupon_rate = reset_rate + self.bond.spread
        self._current_coupon = pd.DataFrame({"resetDate": reset, "couponStart": start, "couponEnd": end,
                                             "accrualFactor": af, "resetRate": reset_rate, "spread": self.bond.spread,
                                             "couponRate": coupon_rate,
                                             "coupon": coupon_rate * af * self.bond.face_amount})

    def _get_expected_coupons(self):
        if self.bond.cap is not np.nan and self.bond.floor is not np.nan:
            raise ValueError("'Pricer' can't deal with caps and floor. Set a proper pricer.")
        reset_dates = self.bond.schedule.schedule["resetDate"]
        resets = reset_dates[reset_dates > self.bond.evaluation_date]
        starts = self.bond.schedule.schedule["startingDate"][reset_dates > self.bond.evaluation_date]
        payments = self.bond.schedule.schedule["paymentDate"][reset_dates > self.bond.evaluation_date]
        af = accrual_factor(self.bond.dcc, starts, payments)
        index = pd.RangeIndex(self.bond.coupon_history.index[-1] + 1,
                              self.bond.coupon_history.shape[0] + len(payments) + 2, name="couponNumber")
        self._expected_coupons = pd.concat([self.current_coupon, pd.DataFrame(
            {"resetDate": resets, "couponStart": starts, "couponEnd": payments, "accrualFactor": af,
             "resetRate": self.forward_rates, "spread": self.bond.spread,
             "couponRate": self.forward_rates + self.bond.spread,
             "coupon": (self.forward_rates + self.bond.spread) * af * self.bond.face_amount})],
                                           ignore_index=True).set_index(index).replace(np.nan, "-")

    def present_value(self) -> dict:
        """
        Calculate present value of the sum of the expected cash flows.
        """
        df = self.discount_curve.discount_factors.loc[self.expected_coupons.couponEnd].to_numpy()
        start, end = self.expected_coupons.couponStart.iloc[0], self.expected_coupons.couponEnd.iloc[0]
        accrued_interest = self.expected_coupons.coupon.iloc[0] * (self.bond.evaluation_date + BDay(2)
                                                                   - start).days / (end - start).days
        expected_coupon_pv = self.expected_coupons.coupon.to_numpy().dot(df)
        face_value_pv = df[-1] * self.bond.face_amount

        prices = {"riskFreeValue": {"dirtyPrice": (expected_coupon_pv + face_value_pv).item(),
                                    "accruedInterest": accrued_interest,
                                    "cleanPrice": (expected_coupon_pv + face_value_pv - accrued_interest).item()}}

        if self.bond._cds_spread:
            if self.bond._recovery_rate is None:
                warnings.warn(
                    "CDS spread detected but could not find recovery rate. Continue with risk free valuation.")
                return prices

            expected_coupon_pv_on_survival = (self.expected_coupons.coupon.to_numpy() *
                                              self.bond.survival_probabilities).dot(df)
            delta_prob = np.diff(-self.bond.survival_probabilities, prepend=-1)
            expected_coupon_pv_on_default = (self.bond.recovery_rate * delta_prob).dot(df) * self.bond.face_amount
            face_value_pv_on_survival = self.bond.face_amount * df[-1] * self.bond.survival_probabilities[-1]
            prices = {**prices,
                      "riskAdjustedValue":
                          {"dirtyPrice": (expected_coupon_pv_on_default +
                                          expected_coupon_pv_on_survival + face_value_pv_on_survival).item(),
                           "accruedInterest": accrued_interest,
                           "cleanPrice": (expected_coupon_pv_on_default +
                                          expected_coupon_pv_on_survival +
                                          face_value_pv_on_survival - accrued_interest).item()}}

        return prices


class BlackPricer(Pricer):
    """
    Class to implement the Black model.
    """

    def __init__(self, discount_curve, volatility_surface):
        """
        Args:
            discount_curve (DiscountCurve): DiscountCurve instance
            volatility_surface (pandas.DataFrame): volatility surface data for the Black model
        """
        super().__init__(discount_curve)
        self.volatility_surface = volatility_surface
        self._cap_fwd_premiums = None
        self._floor_fwd_premiums = None

    @property
    def volatility_surface(self):
        return self._volatility_surface

    @volatility_surface.setter
    def volatility_surface(self, volatility_surface):
        if isinstance(volatility_surface, pd.DataFrame):
            self._volatility_surface = volatility_surface
        else:
            raise ValueError("Volatility surface should be a DataFrame.")

    @property
    def cap_forward_premiums(self):
        if self._cap_fwd_premiums is None or self.discount_curve.shift_flag:
            self._get_cap_floor_forward_premiums()
        return self._cap_fwd_premiums

    @property
    def floor_forward_premiums(self):
        if self._floor_fwd_premiums is None or self.discount_curve.shift_flag:
            self._get_cap_floor_forward_premiums()
        return self._floor_fwd_premiums

    def _get_cap_floor_forward_premiums(self):
        # TODO: check whether to use the same AF for coupon and caplet/floorlet. Rebus sic stantibus it uses the same
        # volatility for cap and floor:
        maturity = (self.bond.schedule.schedule["paymentDate"][-1] - self.bond.evaluation_date).days / 365
        interpolator = scipy.interpolate.RegularGridInterpolator(
            (self.volatility_surface.index, self.volatility_surface.columns), self.volatility_surface.values,
            bounds_error=False, fill_value=None)  # extrapolate values outside bounds
        cap_vol, floor_vol = interpolator([(maturity, self.bond.cap), (maturity, self.bond.floor)])

        # time to maturity and af for each caplet and floorlet:
        reset_dates = self.bond.schedule.schedule["resetDate"]
        future_reset_dates = reset_dates[reset_dates > self.bond.evaluation_date]
        ttm = accrual_factor("ACT/365", self.bond.evaluation_date, future_reset_dates)
        af = accrual_factor("ACT/365",
                            self.bond.schedule.schedule["startingDate"][reset_dates > self.bond.evaluation_date],
                            self.bond.schedule.schedule["paymentDate"][reset_dates > self.bond.evaluation_date])

        # underlying:
        underlying_rate = self.forward_rates + self.bond.spread
        # d1 and d2:
        d1_cap = (np.log(underlying_rate / self.bond.cap) + 0.5 * ttm * cap_vol ** 2) / (cap_vol * ttm ** 0.5)
        d2_cap = (np.log(underlying_rate / self.bond.cap) - 0.5 * ttm * cap_vol ** 2) / (cap_vol * ttm ** 0.5)
        d1_floor = (np.log(underlying_rate / self.bond.floor) + 0.5 * ttm * floor_vol ** 2) / (floor_vol * ttm ** 0.5)
        d2_floor = (np.log(underlying_rate / self.bond.floor) - 0.5 * ttm * floor_vol ** 2) / (floor_vol * ttm ** 0.5)

        # N(d1) and N(d2)
        nd1_cap, nd2_cap = norm.cdf(d1_cap), norm.cdf(d2_cap)
        nd1_floor, nd2_floor = norm.cdf(-d1_floor), norm.cdf(-d2_floor)

        self._cap_fwd_premiums = (underlying_rate * nd1_cap - self.bond.cap * nd2_cap) * af * self.bond.face_amount
        self._floor_fwd_premiums = (self.bond.floor * nd2_floor -
                                    underlying_rate * nd1_floor) * af * self.bond.face_amount

    def _get_current_coupon(self):
        reset_dates = self.bond.schedule.schedule["resetDate"]
        reset = reset_dates[reset_dates <= self.bond.evaluation_date][-1]
        reset_rate = self.bond.historical_euribor.loc[reset].item()
        start = self.bond.schedule.schedule["startingDate"][reset_dates <= self.bond.evaluation_date][-1]
        end = self.bond.schedule.schedule["paymentDate"][reset_dates <= self.bond.evaluation_date][-1]
        af = accrual_factor(self.bond.dcc, start, end)
        floorlet = np.maximum(self.bond.floor - (reset_rate + self.bond.spread), 0) * af * self.bond.face_amount
        caplet = np.maximum((reset_rate + self.bond.spread) - self.bond.cap, 0) * af * self.bond.face_amount
        coupon_rate = reset_rate + self.bond.spread
        self._current_coupon = pd.DataFrame(
            {"resetDate": reset, "couponStart": start, "couponEnd": end, "accrualFactor": af, "resetRate": reset_rate,
             "spread": self.bond.spread, "couponRate": coupon_rate, "floorlet": floorlet, "caplet": -caplet,
             "coupon": np.nansum([coupon_rate * af * self.bond.face_amount, floorlet, - caplet])})

    def _get_expected_coupons(self):
        reset_dates = self.bond.schedule.schedule["resetDate"]
        resets = reset_dates[reset_dates > self.bond.evaluation_date]
        starts = self.bond.schedule.schedule["startingDate"][reset_dates > self.bond.evaluation_date]
        payments = self.bond.schedule.schedule["paymentDate"][reset_dates > self.bond.evaluation_date]
        af = accrual_factor(self.bond.dcc, starts, payments)
        coupon_rate = self.forward_rates + self.bond.spread
        index = pd.RangeIndex(self.bond.coupon_history.index[-1] + 1,
                              self.bond.coupon_history.shape[0] + len(payments) + 2, name="couponNumber")
        self._expected_coupons = pd.concat(
            [self.current_coupon, pd.DataFrame(
                {"resetDate": resets, "couponStart": starts, "couponEnd": payments, "accrualFactor": af,
                 "resetRate": self.forward_rates, "spread": self.bond.spread, "couponRate": coupon_rate,
                 "floorlet": self.floor_forward_premiums, "caplet": -self.cap_forward_premiums,
                 "coupon": np.nansum([coupon_rate * af * self.bond.face_amount, self.floor_forward_premiums,
                                      -self.cap_forward_premiums], axis=0)}
            )], ignore_index=True).set_index(index).replace(np.nan, "-")


class BachelierPricer(BlackPricer):
    """
    Class to implement the Bachelier model.
    """

    def __init__(self, discount_curve, volatility_surface):
        """
        Args:
            discount_curve (DiscountCurve): DiscountCurve instance
            volatility_surface (pandas.DataFrame): volatility surface data for the Bachelier model
        """
        super().__init__(discount_curve, volatility_surface)
        self.volatility_surface = volatility_surface

    def _get_cap_floor_forward_premiums(self):
        # TODO: check whether to use the same AF for coupon and caplet/floorlet. Rebus sic stantibus it uses the same
        # volatility for cap and floor:
        maturity = (self.bond.schedule.schedule["paymentDate"][-1] - self.bond.evaluation_date).days / 365
        interpolator = scipy.interpolate.RegularGridInterpolator(
            (self.volatility_surface.index, self.volatility_surface.columns), self.volatility_surface.values,
            bounds_error=False, fill_value=None)  # extrapolate values outside bounds
        cap_vol, floor_vol = interpolator([(maturity, self.bond.cap), (maturity, self.bond.floor)])

        # time to maturity and accrual factor for each caplet and floorlet:
        reset_dates = self.bond.schedule.schedule["resetDate"]
        future_reset_dates = reset_dates[reset_dates > self.bond.evaluation_date]
        ttm = accrual_factor("ACT/365", self.bond.evaluation_date, future_reset_dates)
        af = accrual_factor("ACT/365",
                            self.bond.schedule.schedule["startingDate"][reset_dates > self.bond.evaluation_date],
                            self.bond.schedule.schedule["paymentDate"][reset_dates > self.bond.evaluation_date])
        # underlying:
        underlying_rate = self.forward_rates + self.bond.spread

        # d1 and d2:
        d1_cap = (underlying_rate - self.bond.cap) / (cap_vol * ttm ** 0.5)
        d1_floor = (underlying_rate - self.bond.floor) / (floor_vol * ttm ** 0.5)

        # N(d1) and N(d2)
        nd1_cap, small_nd1_cap = norm.cdf(d1_cap), norm.pdf(d1_cap)
        nd1_floor, small_nd1_floor = norm.cdf(-d1_floor), norm.pdf(d1_floor)

        self._cap_fwd_premiums = ((underlying_rate - self.bond.cap) * nd1_cap +
                                  cap_vol * small_nd1_cap * ttm ** 0.5) * af * self.bond.face_amount
        self._floor_fwd_premiums = ((self.bond.floor - underlying_rate) * nd1_floor +
                                    floor_vol * small_nd1_floor * ttm ** 0.5) * af * self.bond.face_amount


class DisplacedBlackPricer(BlackPricer):
    """
    Class to implement the shifted Black model.
    """

    def __init__(self, discount_curve, volatility_surface, shift=0.03):
        """
        Args:
            discount_curve (DiscountCurve): DiscountCurve instance
            volatility_surface (pandas.DataFrame): volatility surface data for the displaced-Black model
            shift (float): displacement size (default 3%)
        """
        super().__init__(discount_curve, volatility_surface)
        self.shift = shift

    def _get_cap_floor_forward_premiums(self):
        # TODO: check whether to use the same AF for coupon and caplet/floorlet. Rebus sic stantibus it uses the same

        cap_strike = self.shift + self.bond.cap
        floor_strike = self.shift + self.bond.floor
        # volatility for cap and floor:
        maturity = (self.bond.schedule.schedule["paymentDate"][-1] - self.bond.evaluation_date).days / 365
        interpolator = scipy.interpolate.RegularGridInterpolator(
            (self.volatility_surface.index, self.volatility_surface.columns), self.volatility_surface.values,
            bounds_error=False, fill_value=None)  # extrapolate values outside bounds
        cap_vol, floor_vol = interpolator([(maturity, cap_strike), (maturity, floor_strike)])

        # time to maturity and accrual factor for each caplet and floorlet:
        reset_dates = self.bond.schedule.schedule["resetDate"]
        future_reset_dates = reset_dates[reset_dates > self.bond.evaluation_date]
        ttm = accrual_factor("ACT/365", self.bond.evaluation_date, future_reset_dates)
        af = accrual_factor("ACT/365",
                            self.bond.schedule.schedule["startingDate"][reset_dates > self.bond.evaluation_date],
                            self.bond.schedule.schedule["paymentDate"][reset_dates > self.bond.evaluation_date])
        # underlying:
        underlying_rate = self.forward_rates + self.bond.spread + self.shift

        # d1 and d2:
        d1_cap = (np.log(underlying_rate / cap_strike) + 0.5 * ttm * cap_vol ** 2) / (cap_vol * ttm ** 0.5)
        d2_cap = (np.log(underlying_rate / cap_strike) - 0.5 * ttm * cap_vol ** 2) / (cap_vol * ttm ** 0.5)
        d1_floor = (np.log(underlying_rate / floor_strike) + 0.5 * ttm * floor_vol ** 2) / (floor_vol * ttm ** 0.5)
        d2_floor = (np.log(underlying_rate / floor_strike) - 0.5 * ttm * floor_vol ** 2) / (floor_vol * ttm ** 0.5)

        # N(d1) and N(d2)
        nd1_cap, nd2_cap = norm.cdf(d1_cap), norm.cdf(d2_cap)
        nd1_floor, nd2_floor = norm.cdf(-d1_floor), norm.cdf(-d2_floor)

        self._cap_fwd_premiums = (underlying_rate * nd1_cap - cap_strike * nd2_cap) * af * self.bond.face_amount
        self._floor_fwd_premiums = (floor_strike * nd2_floor - underlying_rate * nd1_floor) * af * self.bond.face_amount
