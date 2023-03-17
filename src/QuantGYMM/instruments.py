from __future__ import annotations
import copy
import numpy
import numpy as np
import pandas
import pandas as pd
from pandas.tseries.offsets import DateOffset, BDay
from .utils import business_adjustment, accrual_factor, number_of_month
from scipy.optimize import minimize
from .descriptors import *
from .pricers import Pricer
from .term_structures import DiscountCurve
from .calendar import Schedule

__all__ = ["FloatingRateBond", "VanillaSwap"]




class FloatingRateBond:
    """
    Bond class for floating rate bond.
    """
    dcc = DayCountConvention(sterilize_attr=["_coupon_history"])
    face_amount = PositiveNumber(sterilize_attr=["_coupon_history"])
    fixing_days = NonNegativeInteger(sterilize_attr=["_coupon_history"])
    spread = FloatNumber(sterilize_attr=["_coupon_history"], none_accepted=True, return_if_none=0.0)
    cap = FloatNumber(sterilize_attr=["_coupon_history"], none_accepted=True, return_if_none=np.nan)
    floor = FloatNumber(sterilize_attr=["_coupon_history"], none_accepted=True, return_if_none=np.nan)

    def __init__(self, schedule, dcc, face_amount, fixing_days, spread=0.0, floor=None, cap=None):
        """
        Args:
            schedule (Schedule): schedule object for the coupons
            dcc (str): day count convention
            face_amount (int | float): bond face amount
            fixing_days (int): number of days previous to reset date on the fixing of coupon rate occurs
            spread (float): [optional] spread over the floating rate
            floor (float): [optional] floor rate for the coupon
            cap (float): [optional] cap rate for the coupon
        """
        self.dcc = dcc
        self.face_amount = face_amount
        self.fixing_days = fixing_days
        self.spread = spread
        self.floor = floor
        self.cap = cap
        self.schedule = schedule
        self._evaluation_date = None
        self._pricer = None
        self._historical_euribor = None
        self._coupon_history = None
        self._hedging_instruments = None
        self._cds_spread = None
        self._recovery_rate = None
        self._survival_probabilities = None

    @property
    def schedule(self):
        return self._schedule

    @schedule.setter
    def schedule(self, schedule):
        if isinstance(schedule, Schedule):
            bond_schedule = copy.deepcopy(schedule)
            bond_schedule._schedule = {"resetDate": schedule.schedule["startingDate"] - BDay(self.fixing_days),
                                       **schedule.schedule}
            self._schedule = bond_schedule
            self._coupon_history = None
        else:
            raise ValueError(f"'{schedule}' is not a Schedule object.")

    @property
    def historical_euribor(self):
        if self._historical_euribor is None:
            raise ValueError("Historical euribor has not been set. Call 'set_historical_euribor' method to set it.")
        return self._historical_euribor

    @property
    def evaluation_date(self):
        if self._evaluation_date is None:
            raise ValueError("Evaluation date has not been set. Call 'set_evaluation_date' method to set it.")
        return self._evaluation_date

    @property
    def pricer(self):
        if self._pricer is None:
            raise ValueError("No pricer set. Call 'set_pricer' method to set it.")
        return self._pricer

    @property
    def coupon_history(self):
        if self._coupon_history is None:
            self._coupon_history = self.get_coupons_history()
        return self._coupon_history

    @property
    def hedging_instruments(self):
        if self._hedging_instruments is None:
            raise ValueError("No hedging instrument set yet. Call 'set_hedging_instruments' method to set it.")
        return self._hedging_instruments

    @property
    def cds_spread(self):
        if self._cds_spread is None:
            raise ValueError("CDS spread has not been set. Call 'set_cds_spread' method to set it.")
        return self._cds_spread

    @property
    def recovery_rate(self):
        if self._recovery_rate is None:
            raise ValueError("Recovery rate has not been set. Call 'set_recovery_rate' method to set it.")
        return self._recovery_rate

    @property
    def survival_probabilities(self):
        if self._survival_probabilities is None:
            self._get_survival_prob()
        return self._survival_probabilities

    def __repr__(self):
        return f"Bond(faceAmount={self.face_amount}, spread={self.spread}, " \
               f"maturity={self.schedule.schedule['paymentDate'][-1].strftime(format('%Y-%m-%d'))}," \
               f" floor={self.floor}, cap={self.cap})"

    def set_evaluation_date(self, date) -> None:
        """
        Set evaluation date for market price calculation.
        Args:
            date (str | pandas.Timestamp): trade date
        """
        try:
            self._evaluation_date = pd.to_datetime(date)
            self._coupon_history = None
        except Exception:
            raise ValueError(f"Can't convert {date} to datetime.")

    def set_pricer(self, pricer) -> None:
        """
        Set the pricer to be used in the market value calculation.
        Args:
            pricer (Pricer): instance of pricer class
        """
        if isinstance(pricer, Pricer):
            self._pricer = pricer
            self.pricer.transfer_bond_features(self)
        else:
            raise ValueError("Pricer must be a Pricer object.")

    def expected_coupons(self):
        return self.pricer.expected_coupons

    def prices(self) -> dict:
        """
        Compute fair market price as the sum of discounted expected cash flows.
        Returns:
            market price
        """
        return self.pricer.present_value()

    def set_historical_euribor(self, historical_euribor) -> None:
        """
        Set the historical libor necessary to compute the historical coupons and the current coupon.
        Args:
            historical_euribor (pandas.DataFrame): past euribor data
        """
        self._historical_euribor = historical_euribor
        self._coupon_history = None

    def get_coupons_history(self) -> pandas.DataFrame:
        """
        Calculate the past history of coupons.
        Returns:
            pandas.DataFrame of coupons reset date, coupon staring date, coupon payment date, coupon accrual factor,
            coupon rate.
        """
        af = accrual_factor(self.dcc, self.schedule.schedule["startingDate"], self.schedule.schedule["paymentDate"])
        past_date_mask = self.schedule.schedule["paymentDate"] <= self.evaluation_date
        hist_reset = self.schedule.schedule["resetDate"][past_date_mask]
        hist_starting = self.schedule.schedule["startingDate"][past_date_mask]
        hist_payment = self.schedule.schedule["paymentDate"][past_date_mask]
        hist_rate = self.historical_euribor.loc[hist_reset].to_numpy().squeeze() + self.spread
        hist_accrual = af[past_date_mask]
        floorlet = np.maximum(self.floor - hist_rate, 0) * hist_accrual * self.face_amount
        caplet = np.maximum(hist_rate - self.cap, 0) * hist_accrual * self.face_amount
        self._coupon_history = pd.DataFrame(
            {"resetDate": hist_reset, "couponStart": hist_starting, "couponEnd": hist_payment,
             "accrualFactor": hist_accrual, "resetRate": hist_rate - self.spread, "spread": self.spread,
             "couponRate": hist_rate, "floorlet": floorlet, "caplet": -caplet,
             "coupon": np.nansum([hist_rate * hist_accrual * self.face_amount, floorlet, -caplet],
                                 axis=0)}, index=pd.RangeIndex(1, len(hist_rate) + 1, name="couponNumber")
        ).replace(np.nan, "-")
        return self.coupon_history

    def sensitivity(self, shift_type="parallel", shift_size=0.01, kind="symmetric") -> float:
        """
        Calculate the DV01 of the bond to different type of shift by means of finite difference approximation.
        Args:
            shift_type (str): type of term structure shift (valid inputs are 'parallel', 'slope', 'curvature').
            shift_size (float): the term structure shift size to be applied to estimate the bond first derivatives.
            kind (str): finite difference approximation type (valid inputs are 'symmetric', 'oneside').
        Returns:
            The estimated bond DV01.
        """
        self.pricer.discount_curve.reset_shift()
        if self._cds_spread:
            match kind:
                case "symmetric":
                    match shift_type:
                        case "parallel":
                            self.pricer.discount_curve.apply_parallel_shift(shift_size)
                            price_up_shift = self.prices()["riskAdjustedValue"]["dirtyPrice"]
                            self.pricer.discount_curve.reset_shift()
                            self.pricer.discount_curve.apply_parallel_shift(-shift_size)
                            price_down_shift = self.prices()["riskAdjustedValue"]["dirtyPrice"]
                        case "slope":
                            self.pricer.discount_curve.apply_slope_shift(shift_size)
                            price_up_shift = self.prices()["riskAdjustedValue"]["dirtyPrice"]
                            self.pricer.discount_curve.reset_shift()
                            self.pricer.discount_curve.apply_slope_shift(-shift_size)
                            price_down_shift = self.prices()["riskAdjustedValue"]["dirtyPrice"]
                        case "curvature":
                            self.pricer.discount_curve.apply_curvature_shift(shift_size)
                            price_up_shift = self.prices()["riskFreeValue"]["dirtyPrice"]
                            self.pricer.discount_curve.reset_shift()
                            self.pricer.discount_curve.apply_curvature_shift(-shift_size)
                            price_down_shift = self.prices()["riskFreeValue"]["dirtyPrice"]
                        case _:
                            raise ValueError("Admitted shift type are: 'parallel', 'slope' or 'curvature'.")
                    self.pricer.discount_curve.reset_shift()
                    return (price_up_shift - price_down_shift) / (2 * shift_size) * 0.0001
                case "oneside":
                    match shift_type:
                        case "parallel":
                            price = self.prices()["riskFreeValue"]["dirtyPrice"]
                            self.pricer.discount_curve.apply_parallel_shift(shift_size)
                            price_shift = self.prices()["riskFreeValue"]["dirtyPrice"]
                        case "slope":
                            price = self.prices()["riskFreeValue"]["dirtyPrice"]
                            self.pricer.discount_curve.apply_slope_shift(shift_size)
                            price_shift = self.prices()["riskFreeValue"]["dirtyPrice"]
                        case "curvature":
                            price = self.prices()["riskFreeValue"]["dirtyPrice"]
                            self.pricer.discount_curve.reset_shift()
                            self.pricer.discount_curve.apply_curvature_shift(shift_size)
                            price_shift = self.prices()["riskFreeValue"]["dirtyPrice"]
                        case _:
                            raise ValueError("Admitted shift types are: 'parallel', 'slope' or 'curvature'.")
                    self.pricer.discount_curve.reset_shift()
                    return (price_shift - price) / shift_size * 0.0001
                case _:
                    raise ValueError("Admitted kind types are: 'symmetric', 'oneside'")
        else:
            match kind:
                case "symmetric":
                    match shift_type:
                        case "parallel":
                            self.pricer.discount_curve.apply_parallel_shift(shift_size)
                            price_up_shift = self.prices()["riskFreeValue"]["dirtyPrice"]
                            self.pricer.discount_curve.reset_shift()
                            self.pricer.discount_curve.apply_parallel_shift(-shift_size)
                            price_down_shift = self.prices()["riskFreeValue"]["dirtyPrice"]
                        case "slope":
                            self.pricer.discount_curve.apply_slope_shift(shift_size)
                            price_up_shift = self.prices()["riskFreeValue"]["dirtyPrice"]
                            self.pricer.discount_curve.reset_shift()
                            self.pricer.discount_curve.apply_slope_shift(-shift_size)
                            price_down_shift = self.prices()["riskFreeValue"]["dirtyPrice"]
                        case "curvature":
                            self.pricer.discount_curve.apply_curvature_shift(shift_size)
                            price_up_shift = self.prices()["riskFreeValue"]["dirtyPrice"]
                            self.pricer.discount_curve.reset_shift()
                            self.pricer.discount_curve.apply_curvature_shift(-shift_size)
                            price_down_shift = self.prices()["riskFreeValue"]["dirtyPrice"]
                        case _:
                            raise ValueError("Admitted shift type are: 'parallel', 'slope' or 'curvature'.")
                    self.pricer.discount_curve.reset_shift()
                    return (price_up_shift - price_down_shift) / (2 * shift_size) * 0.0001
                case "oneside":
                    match shift_type:
                        case "parallel":
                            price = self.prices()["riskFreeValue"]["dirtyPrice"]
                            self.pricer.discount_curve.apply_parallel_shift(shift_size)
                            price_shift = self.prices()["riskFreeValue"]["dirtyPrice"]
                        case "slope":
                            price = self.prices()["riskFreeValue"]["dirtyPrice"]
                            self.pricer.discount_curve.apply_slope_shift(shift_size)
                            price_shift = self.prices()["riskFreeValue"]["dirtyPrice"]
                        case "curvature":
                            price = self.prices()["riskFreeValue"]["dirtyPrice"]
                            self.pricer.discount_curve.reset_shift()
                            self.pricer.discount_curve.apply_curvature_shift(shift_size)
                            price_shift = self.prices()["riskFreeValue"]["dirtyPrice"]
                        case _:
                            raise ValueError("Admitted shift types are: 'parallel', 'slope' or 'curvature'.")
                    self.pricer.discount_curve.reset_shift()
                    return (price_shift - price) / shift_size * 0.0001
                case _:
                    raise ValueError("Admitted kind types are: 'symmetric', 'oneside'")

    def set_hedging_instruments(self, instruments) -> None:
        """
        Set the hedging instruments.
        Args:
            instruments (list | tuple): list, tuple of suitable hedging instruments. Suitable
                                                    hedging instruments implements a 'sensitivity' method.
        """
        if not isinstance(instruments, (list, tuple)):
            raise ValueError("'instruments' must be a iterable.")

        for instrument in instruments:
            if not hasattr(instrument, "sensitivity"):
                raise ValueError(f"{instrument} is not a valid hedging instruments.")

        self._hedging_instruments = instruments

    def hedging_ratio(self, hedge) -> list:
        """
        Calculate the hedging ratio given some hedging instruments. If the number of instruments and the number of
        hedge is the same, it searches for an exact solution. If the number of instruments is greater than the number
        of hedge, it will minimize the 'cost' of the hedge, finding the minimum number of contracts in which enter to
        carry out the hedge. If the number of hedge is greater that the number of instruments it performs a minimization
        on the system.
        Args:
            hedge (list | tuple): list, tuple of shifts to hedge against, for example ["parallel", "slope"].
        """
        dv01hi = np.array([
            [instrument.sensitivity(shift_type=shift_type) for instrument in self.hedging_instruments]
            for shift_type in hedge])
        dv01bond = np.array([self.sensitivity(shift_type=shift_type) for shift_type in hedge])

        try:
            if len(self.hedging_instruments) > len(hedge):
                solver = minimize(fun=lambda x: np.sum(x ** 2),
                                  x0=np.random.rand(len(self.hedging_instruments)),
                                  constraints={"type": "eq", "fun": lambda x: dv01hi.dot(x) + dv01bond})
                n = solver.x

            elif len(self.hedging_instruments) < len(hedge):
                solver = minimize(fun=lambda x: np.sum((dv01hi.dot(x) + dv01bond) ** 2),
                                  x0=np.random.rand(len(self.hedging_instruments)))
                n = solver.x

            else:
                n = -np.linalg.inv(dv01hi).dot(dv01bond)
        except Exception as error:
            raise ValueError(error, "\nCould not find the hedging ratio.")

        return n

    def set_cds_spread(self, spread) -> None:
        """
        Args:
            spread (float): CDS spread for a period equal to the bond time to maturity.
        """
        if not isinstance(spread, float) and spread is not None:
            raise ValueError("Wrong type for parameter 'spread', valid type is float.")
        self._survival_probabilities = None
        self._cds_spread = spread

    def set_recovery_rate(self, recovery_rate) -> None:
        """
        Args:
            recovery_rate (float | list | numpy.ndarray): either a recovery rate or an array of recovery rates
                                                            (if the RR is assumed to be time-varying).
        """
        if not isinstance(recovery_rate, (numpy.ndarray, list, float)) and recovery_rate is not None:
            raise ValueError("Wrong type for 'recovery_rate': it must be a float or an arrays.")
        self._survival_probabilities = None
        self._recovery_rate = recovery_rate

    def _get_survival_prob(self):
        ttm = accrual_factor("ACT/365", self.evaluation_date, self.expected_coupons()["couponEnd"])
        self._survival_probabilities = (np.exp(-self.cds_spread * ttm) - self.recovery_rate) / (1 - self.recovery_rate)


class VanillaSwap:
    """
    Base class to implement vanilla swap pricing and sensitivity.
    """
    _SPOT_LEG = 2
    _BUSINESS_CONVENTION = "modified_following"
    _DCC_FIXED = "30/360"
    _DCC_FLOATING = "ACT/360"

    fixed_leg_frequency = PositiveNumber(sterilize_attr=["_swap_rate", "_calendar", "_accrual_start_dates"])
    floating_leg_frequency = PositiveNumber(sterilize_attr=["_swap_rate", "_calendar", "_accrual_start_dates"])

    def __init__(self, discount_curve, fixed_leg_frequency, floating_leg_frequency, maturity, start="today"):
        """
        Args:
            discount_curve (DiscountCurve): discount curve object
            maturity (str | int | float): swap contract maturity in years, or 'YYYY-MM-DD' string indicating
                                            the maturity date
            start (str): if "today" the contract is spot starting, otherwise specify the start date
            fixed_leg_frequency (int | float): swap fixed leg payment frequency
            floating_leg_frequency (int | float): swap floating leg payment frequency
        """
        self.discount_curve = discount_curve
        self.start = start
        self.maturity = maturity
        self.fixed_leg_frequency = fixed_leg_frequency
        self.floating_leg_frequency = floating_leg_frequency
        self._swap_rate = None
        self._calendar = None
        self._accrual_start_dates = None

    @property
    def discount_curve(self):
        return self._discount_curve

    @discount_curve.setter
    def discount_curve(self, discount_curve):
        if isinstance(discount_curve, DiscountCurve):
            self._discount_curve = discount_curve
            self._swap_rate = None
        else:
            raise ValueError(
                f"'discount_curve' must be a DiscountCurve object. Got {discount_curve.__class__.__name__}.")

    @property
    def maturity(self):
        return self._maturity

    @maturity.setter
    def maturity(self, maturity):
        self._swap_rate = None
        self._calendar = None
        self._accrual_start_dates = None
        if isinstance(maturity, pandas.Timestamp):
            self._maturity = maturity
        elif isinstance(maturity, (int, float)):
            self._maturity = self.start + BDay(self._SPOT_LEG) + DateOffset(years=maturity // 1,
                                                                            months=(maturity % 1) * 12 // 1,
                                                                            days=round((maturity % 1) * 12 % 1 * 30))
        elif isinstance(maturity, str):
            try:
                self._maturity = pd.to_datetime(maturity)
            except Exception as error:
                raise Exception(error, f"\nCould not convert {maturity} to datetime.")
        else:
            raise ValueError(f"Wrong type for input 'maturity'.")

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, start):
        self._swap_rate = None
        self._calendar = None
        self._accrual_start_dates = None
        if start == "today":
            self._start = self.discount_curve.trade_date
            self._value_date = self.discount_curve.trade_date + BDay(self._SPOT_LEG)
        else:
            try:
                self._start = pd.to_datetime(start)
                self._value_date = pd.to_datetime(start) + BDay(self._SPOT_LEG)
            except Exception as error:
                raise Exception(error, f"\nCould not convert {start} to datetime.")

    @property
    def value_date(self):
        return self._value_date

    @property
    def calendar(self):
        if self._calendar is None:
            self._get_calendar()
        return self._calendar

    @property
    def swap_rate(self):
        if self._swap_rate is None:
            self._calculate_swap_rate()
        return self._swap_rate

    @property
    def accrual_start_dates(self):
        if self._accrual_start_dates is None:
            self._get_calendar()
        return self._accrual_start_dates

    def _calculate_swap_rate(self):
        df_fixed = self.discount_curve.discount_factors.loc[self.calendar["fixedLeg"]]
        af_fixed = accrual_factor(self._DCC_FIXED, self.accrual_start_dates["fixedLeg"])
        annuity = af_fixed.dot(df_fixed).item()
        floating_leg = self._floating_leg_market_value()
        self._swap_rate = floating_leg / annuity

    def _floating_leg_market_value(self):
        # floating leg market value is calculated considering the spot lag. The first reset date is the
        # inception/trade_date, the start date is the trade_date + self.SPOT_LAG. The following reset dates occur
        # self.SPOT_LAG days before the starting date of each period. The spot lag results in interests starting to
        # accrue from reset_date + self.SPOT_LAG business days until payment day.
        # estimating L(reset_date, reset_date + tenor) by forward rate L(reset_date_0, reset_date, reset_date + tenor)
        af = accrual_factor(self._DCC_FLOATING, self.accrual_start_dates["floatingLeg"])
        df1 = self.discount_curve.discount_factors.loc[self.calendar["resetDate"]]
        df2_date = business_adjustment(self._BUSINESS_CONVENTION, self.calendar["resetDate"] + DateOffset(
            years=1 / self.floating_leg_frequency // 1,
            months=(1 / self.floating_leg_frequency % 1) * 12 // 1,
            days=round((1 / self.floating_leg_frequency % 1) * 12 % 1 * 30)))
        df2 = self.discount_curve.discount_factors.loc[df2_date]
        match self.discount_curve.compounding:
            case "simple":
                forward_rates = (df1.to_numpy() / df2.to_numpy() - 1) / af.reshape(-1, 1)
                # forward_rates = (df1.divide(df2.to_numpy()) - 1).divide(af, axis=0).to_numpy().squeeze()
            case "continuous":
                forward_rates = np.log(df1.to_numpy() / df2.to_numpy()) / af.reshape(-1, 1)
                # forward_rates = np.log(df1.divide(df2.to_numpy())).divide(af, axis=0).to_numpy().squeeze()
            case "annually_compounded":
                forward_rates = (df1.to_numpy() / df2.to_numpy()) ** (1 / af.reshape(-1, 1)) - 1
                # forward_rates = ((df1.divide(df2.to_numpy())).pow(1 / af, axis=0) - 1).to_numpy().squeeze()
            case _:
                raise ValueError("Invalid compounding convention.")
        # calculating present value of the floating leg
        df2 = self.discount_curve.discount_factors.loc[self.calendar["floatingLeg"]]
        floating_leg = (forward_rates.squeeze() * af).dot(df2)
        return floating_leg.item()

    def _fixed_leg_market_value(self):
        # fixed leg market value is calculated considering the spot lag. The spot lag results in interests starting to
        # accrue from start_date + self.SPOT_LAG business days until payment day.

        df = self.discount_curve.discount_factors.loc[self.calendar["fixedLeg"]]
        af = accrual_factor(self._DCC_FIXED, self.accrual_start_dates["fixedLeg"])
        fixed_leg = af.dot(df) * self.swap_rate
        return fixed_leg.item()

    def _get_calendar(self):

        fixed_cash_flow_num = np.ceil(number_of_month(self.value_date, self.maturity) * (self.fixed_leg_frequency / 12))
        floating_cash_flow_num = np.ceil(
            number_of_month(self.value_date, self.maturity) * (self.floating_leg_frequency / 12))

        fixed_cash_flow_date = pd.date_range(end=self.maturity,
                                             freq=DateOffset(
                                                 years=1 / self.fixed_leg_frequency // 1,
                                                 months=(1 / self.fixed_leg_frequency % 1) * 12 // 1,
                                                 days=round((1 / self.fixed_leg_frequency % 1) * 12 % 1 * 30)),
                                             periods=fixed_cash_flow_num)

        floating_cash_flow_date = pd.date_range(end=self.maturity,
                                                freq=DateOffset(
                                                    years=1 / self.floating_leg_frequency // 1,
                                                    months=(1 / self.floating_leg_frequency % 1) * 12 // 1,
                                                    days=round((1 / self.floating_leg_frequency % 1) * 12 % 1 * 30)),
                                                periods=floating_cash_flow_num)

        fixed_cash_flow_date = pd.DatetimeIndex(business_adjustment(self._BUSINESS_CONVENTION,
                                                                    fixed_cash_flow_date))
        floating_cash_flow_date = pd.DatetimeIndex(business_adjustment(self._BUSINESS_CONVENTION,
                                                                       floating_cash_flow_date))

        if self.floating_leg_frequency >= self.fixed_leg_frequency:
            reset_date = pd.DatetimeIndex([self.start]).append(floating_cash_flow_date - BDay(self._SPOT_LEG))
        else:
            reset_date = pd.DatetimeIndex([self.start]).append(fixed_cash_flow_date - BDay(self._SPOT_LEG))

        self._calendar = {"resetDate": reset_date[:-1],
                          "fixedLeg": fixed_cash_flow_date, "floatingLeg": floating_cash_flow_date}
        self._accrual_start_dates = {"floatingLeg": pd.DatetimeIndex([self.value_date]).append(floating_cash_flow_date),
                                     "fixedLeg": pd.DatetimeIndex([self.value_date]).append(fixed_cash_flow_date)}

    def market_price(self) -> float:
        """
        Returns: fair market price at the trade date (fixed leg market value - floating leg market value).
        """
        return self._fixed_leg_market_value() - self._floating_leg_market_value()

    def sensitivity(self, shift_type="parallel", shift_size=0.01, kind="symmetric") -> float:
        """
        Calculate the DV01 of the swap to different type of shift by means of finite difference approximation.
        Args:
            shift_type (str): type of term structure shift (valid inputs are 'parallel', 'slope', 'curvature').
            shift_size (float): the term structure shift size to be applied to estimate the bond first derivatives.
            kind (str): finite difference approximation type (valid inputs are 'symmetric', 'oneside').
        Returns:
            The estimated bond DV01.
        """
        self.discount_curve.reset_shift()
        self._calculate_swap_rate()
        match kind:
            case "symmetric":
                match shift_type:
                    case "parallel":
                        self.discount_curve.apply_parallel_shift(shift_size)
                        price_up_shift = self.market_price()
                        self.discount_curve.reset_shift()
                        self.discount_curve.apply_parallel_shift(-shift_size)
                        price_down_shift = self.market_price()
                    case "slope":
                        self.discount_curve.apply_slope_shift(shift_size)
                        price_up_shift = self.market_price()
                        self.discount_curve.reset_shift()
                        self.discount_curve.apply_slope_shift(-shift_size)
                        price_down_shift = self.market_price()
                    case "curvature":
                        self.discount_curve.apply_curvature_shift(shift_size)
                        price_up_shift = self.market_price()
                        self.discount_curve.reset_shift()
                        self.discount_curve.apply_curvature_shift(-shift_size)
                        price_down_shift = self.market_price()
                    case _:
                        raise ValueError("Admitted shift type are: 'parallel', 'slope' or 'curvature'.")
                self.discount_curve.reset_shift()
                return (price_up_shift - price_down_shift) / (2 * shift_size) * 0.0001
            case "oneside":
                match shift_type:
                    case "parallel":
                        price = self.market_price()
                        self.discount_curve.apply_parallel_shift(shift_size)
                        price_shift = self.market_price()
                    case "slope":
                        price = self.market_price()
                        self.discount_curve.apply_slope_shift(shift_size)
                        price_shift = self.market_price()
                    case "curvature":
                        price = self.market_price()
                        self.discount_curve.reset_shift()
                        self.discount_curve.apply_curvature_shift(shift_size)
                        price_shift = self.market_price()
                    case _:
                        raise ValueError("Admitted shift types are: 'parallel', 'slope' or 'curvature'.")
                self.discount_curve.reset_shift()
                return (price_shift - price) / shift_size * 0.0001
            case _:
                raise ValueError("Admitted kind types are: 'symmetric', 'oneside'")
