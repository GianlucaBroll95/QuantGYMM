from __future__ import annotations
import unicodedata
import numbers
import copy
import warnings
import numpy
import numpy as np
import pandas
import scipy
import re
from pandas.tseries.offsets import DateOffset, BDay, MonthEnd
from scipy.stats.distributions import norm
from QuantGYMM.utils import business_adjustment, accrual_factor, number_of_month
from scipy.optimize import minimize
from QuantGYMM.descriptors import *


# TODO: try to implement piecewise constant forward interpolation

class Schedule:
    """
    Schedule object for coupon planning.
    """
    start_date = Date(sterilize_attr=["_schedule"])
    end_date = Date(sterilize_attr=["_schedule"])
    frequency = PositiveNumber(sterilize_attr=["_schedule"])
    convention = BusinessConvention(sterilize_attr=["_schedule"])
    eom = Boolean(sterilize_attr=["_schedule"])

    def __init__(self, start_date, end_date, frequency, convention="modified_following", eom=True):
        """
        Args:
            start_date (str | pandas.Timestamp): "YYYY-MM-DD" string indicating starting date
            end_date (str | pandas.Timestamp): "YYYY-MM-DD" string indicating ending date
            frequency (float | int): frequency of payment
            convention (str): business day convention (default is "modified_following")
            eom (bool): end of month rule (default is True)
        """
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency
        self.convention = convention
        self.eom = eom
        self._schedule = None

    @property
    def schedule(self):
        if self._schedule is None:
            self._schedule = self._create_schedule()
        return self._schedule

    @schedule.setter
    def schedule(self, value):
        raise ValueError("Can't set coupon schedule directly.")

    def __repr__(self):
        return f"Schedule(start_date = {self.start_date.strftime('%Y-%m-%d')}, " \
               f"end_date = {self.end_date.strftime('%Y-%m-%d')}, frequency = {self.frequency})"

    def _create_schedule(self):

        date = pd.date_range(self.start_date,
                             self.end_date,
                             freq=DateOffset(year=(1 / self.frequency) // 1, months=((1 / self.frequency) % 1) * 12))

        if self.eom and self.start_date.is_month_end:
            date = [d + MonthEnd(0) for d in date]

        date = business_adjustment(self.convention, date)
        date = np.array([d.normalize() for d in date])
        return {"startingDate": date[:-1], "paymentDate": date[1:]}


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
            spread (float): spread over the floating rate
            floor (float): floor rate for the coupon
            cap (float): cap rate for the coupon
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


class SwapRateCurve:
    """
    Swap rate curve object for handling swap rate.
    """
    _SPOT_LEG = 2

    convention = BusinessConvention(sterilize_attr=["_interpolated_rates"])
    dcc = DayCountConvention(sterilize_attr=["_interpolated_rates"])
    trade_date = Date(sterilize_attr=["_interpolated_rates"])
    frequency = PositiveNumber(sterilize_attr=["_interpolated_rates"])

    def __init__(self, trade_date, swap_rate, frequency, convention="modified_following", dcc="30/360",
                 interpolation="linear"):
        """
        Args:
            swap_rate (pandas.DataFrame): swap rate dataframe, should have datetime index and numbers as column
                                        names (e.g. 1, 1.5, 2, ...) corresponding to swap maturities.
            frequency (int): swap fixed leg payment frequency
            trade_date (str | pandas.Timestamp): "YYYY-MM-DD" string representing the trade date
            dcc (str): day count convention of the fixed swap leg
            interpolation (str): method to be used to interpolate swap rate
        """
        self.interpolation = interpolation
        self.convention = convention
        self.dcc = dcc
        self.frequency = frequency
        self.trade_date = trade_date
        self.swap_rates = swap_rate
        self._interpolated_rates = None

    @property
    def spot_leg(self):
        return self._SPOT_LEG

    @property
    def swap_rates(self):
        return self._swap_rates

    @swap_rates.setter
    def swap_rates(self, swap_rates):
        try:
            self._swap_rates = self._get_normalized_rate(swap_rates)
            self._interpolated_rates = None
        except Exception as error:
            raise Exception(error)

    @property
    def interpolation(self):
        return self._interpolation

    @interpolation.setter
    def interpolation(self, interpolation):
        if isinstance(interpolation, str):
            self._interpolated_rates = None
            self._interpolation = interpolation
        else:
            raise ValueError("Interpolation must be a string.")

    @property
    def interpolated_rates(self):
        if self._interpolated_rates is None:
            self.interpolate()
        return self._interpolated_rates

    @interpolated_rates.setter
    def interpolated_rates(self, values):
        raise ValueError("Can't do this!")

    def __repr__(self):
        return f"SwapRateCurve(trade_date={self.trade_date.strftime('%Y-%m-%d')}," \
               f" tenor={int(12 / self.frequency)} months, dcc={self.dcc})"

    def interpolate(self) -> None:
        """
        Interpolates swap rate.
        """

        days_in_term = np.array([d.days for d in self.swap_rates.term - self.swap_rates.term.iloc[0]])
        try:
            interpolating_function = scipy.interpolate.interp1d(days_in_term, self.swap_rates.swapRate,
                                                                kind=self.interpolation)
        except Exception:
            raise ValueError(Exception)
        date_to_interpolate = pd.Index(business_adjustment(self.convention,
                                                           pd.date_range(self.swap_rates.term.iloc[0],
                                                                         self.swap_rates.term.iloc[-1],
                                                                         freq=DateOffset(
                                                                             months=(12 / self.frequency) // 1,
                                                                             days=round(
                                                                                 ((12 / self.frequency) % 1) * 30)))))
        days_to_interpolate = [d.days for d in date_to_interpolate - date_to_interpolate[0]]
        self._interpolated_rates = pd.DataFrame(
            {"term": date_to_interpolate, "interpolatedSwapRate": interpolating_function(days_to_interpolate)})

    def _get_normalized_rate(self, swap_rate):
        starting_date = self.trade_date + BDay(self.spot_leg)
        maturity = business_adjustment(self.convention,
                                       [starting_date + DateOffset(years=maturity // 1, months=12 * (maturity % 1)) for
                                        maturity in self._normalize(swap_rate.index)])
        return pd.DataFrame({"term": maturity, "swapRate": swap_rate.iloc[:, 0]})

    @staticmethod
    def _normalize(data):
        clean_mat = []
        for maturity in data:
            if isinstance(maturity, numbers.Number):
                clean_mat.append(maturity)
                continue
            maturity = unicodedata.normalize("NFKD", maturity).strip()
            if re.search("M", maturity):
                clean_mat.append(int(re.search("\d+", maturity).group()) / 12)
            else:
                clean_mat.append(int(re.search("\d+", maturity).group()))
        return np.array(clean_mat)


class DiscountCurve:
    """
    Discount curve object for extrapolation from swap rate curve.
    """

    dcc = DayCountConvention(sterilize_attr=["_spot_rates", "_discount_factors"])
    compounding = CompoundingConvention(sterilize_attr=["_spot_rates", "_discount_factors"])

    def __init__(self, swap_rate_curve, compounding="annually_compounded", dcc="ACT/365", interpolation="linear"):
        """
        Args:
            swap_rate_curve (SwapRateCurve): swap rate curve object
            compounding (str): compounding of the discount curve (default is 'annually_compounded')
            dcc (str): day count convention for the market rates (default is ACT/365)
            interpolation (str): interpolation method to be used on spot rates interpolation
        """
        self.dcc = dcc
        self.compounding = compounding
        self.interpolation = interpolation
        self.swap_rate_curve = swap_rate_curve
        self._discount_factors = None
        self._spot_rates = None
        self._shift = 0
        self._shift_flag = False

    @property
    def swap_rate_curve(self):
        return self._swap_rate_curve

    @swap_rate_curve.setter
    def swap_rate_curve(self, swap_rate_curve):
        if isinstance(swap_rate_curve, SwapRateCurve):
            self.trade_date = swap_rate_curve.trade_date
            self._swap_rate_curve = swap_rate_curve
            self._discount_factors = None
            self._spot_rates = None
            self._starting_date = swap_rate_curve.trade_date + BDay(swap_rate_curve.spot_leg)
            self._get_df_from_swap_rate()
            self._get_spot()
        else:
            raise ValueError("You need to pass a 'SwapRateCurve' object.")

    @property
    def discount_factors(self):
        if self._discount_factors is None:
            self._get_df_from_spot_rate()
        return self._discount_factors

    @discount_factors.setter
    def discount_factors(self, value):
        raise ValueError("Can't set discount factors directly.")

    @property
    def interpolation(self):
        return self._interpolation

    @interpolation.setter
    def interpolation(self, interpolation):
        if isinstance(interpolation, str):
            self._discount_factors = None
            self._spot_rates = None
            self._interpolation = interpolation
        else:
            raise ValueError("Interpolation must be a string.")

    @property
    def spot_rates(self):
        if self._spot_rates is None:
            self._interpolate_spot_rates()
        return self._spot_rates

    @spot_rates.setter
    def spot_rates(self, value):
        raise ValueError("Can't set spot rates directly.")

    @property
    def sr(self):
        return self._sr

    @property
    def df(self):
        return self._df

    @property
    def shift_flag(self):
        return self._shift_flag

    def __repr__(self):
        return f"DiscountCurve(trade_date={self.swap_rate_curve.trade_date.strftime('%Y-%m-%d')}, " \
               f"compounding={self.compounding}, dcc={self.dcc})"

    def reset_shift(self) -> None:
        """
        Reset the applied shift.
        """
        self._discount_factors = None
        self._spot_rates["spotRate"] = self.spot_rates.spotRate - self._shift
        self._shift = 0

    def apply_parallel_shift(self, shift=0.0001) -> None:
        """
        Applies parallel shift to the term structure.
        Args:
            shift (float): sift size, default is 1 basis point.
        """
        self._shift_flag = True
        self._discount_factors = None
        self._shift += shift
        self._spot_rates["spotRate"] = self.spot_rates.spotRate + shift

    def apply_slope_shift(self, value=0.0001) -> None:
        """
        Applies slope shift to the term structure
        Args:
            value (float): determines the intensity and the direction of the slope shift, the shock applied is
                            linear and goes from value to -value. This means that for positive 'value', further
                            maturities will face a decrease while shorter maturity will endure an increase.
        """
        self._shift_flag = True
        self._discount_factors = None
        schifter = scipy.interpolate.interp1d([0, len(self.spot_rates)], [value, -value])
        shift = schifter(np.arange(len(self.spot_rates)))
        self._shift += shift
        self._spot_rates["spotRate"] = self.spot_rates.spotRate + shift

    def apply_curvature_shift(self, value) -> None:
        """
        Args:
            value (float): the y-axis interception value of the parabola to be added to the term structure to apply a
                            curvature shift. If positive, the parabola convex, if negative the parabola is concave.
                            This means that for positive 'vertex' central values of the term structure will face a
                            negative shock, while extreme values will face positive shock. The opposite will happen
                            for negative 'value'.
        """
        self._shift_flag = True
        self._discount_factors = None
        schifter = scipy.interpolate.interp1d([0, len(self.spot_rates) / 2, len(self.spot_rates)],
                                              [value, -value, value], kind="quadratic")
        shift = schifter(np.arange(len(self.spot_rates)))
        self._shift += shift
        self._spot_rates["spotRate"] = self.spot_rates.spotRate + shift

    def set_df(self, df) -> None:
        # TODO: check this
        """
        Overrides internal calculation of discount factor from swap rate.
        Args:
            df (pandas.DataFrame | pandas.Series): series of discount factor
        """
        self._spot_rates = None
        self._discount_factors = None
        maturity = [self._starting_date] + df.index.to_list()
        afs = accrual_factor("ACT/365", maturity)
        annuity = np.array([0])
        for d, af in zip(df, afs):
            annuity = np.append(annuity, annuity[-1] + d * af)
        self._df = pd.DataFrame(
            {"maturity": maturity[1:], "annuity": annuity[1:], "discountFactor": df,
             "accrualFactor": afs}).reset_index(drop=True)
        self._get_spot()

    def _get_spot(self):
        df = self.df["discountFactor"]
        term = accrual_factor(self.dcc, self.trade_date, self.df.maturity)
        # term = accrual_factor(self.dcc, self._stating_date, self.df.maturity)

        match self.compounding:
            case "simple":
                spot_rates = pd.DataFrame({"maturity": self.df.maturity, "spotRate": (1 / df - 1) / term})
            case "continuous":
                spot_rates = pd.DataFrame({"maturity": self.df.maturity, "spotRate": -np.log(df) / term})
            case "annually_compounded":
                spot_rates = pd.DataFrame(
                    {"maturity": self.df.maturity, "spotRate": (1 / df) ** (1 / term) - 1})
            case _:
                raise ValueError("Invalid parameter for compounding.")
        self._sr = pd.DataFrame({"term": term, **spot_rates})

    def _get_df_from_swap_rate(self):
        maturity = [self._starting_date] + self.swap_rate_curve.interpolated_rates.term.to_list()
        afs = accrual_factor(self.swap_rate_curve.dcc, maturity)
        annuity, df = np.array([0]), np.array([1])
        for sr, af in zip(self.swap_rate_curve.interpolated_rates.interpolatedSwapRate, afs):
            df = np.append(df, (1 - annuity[-1] * sr) / (1 + af * sr))
            annuity = np.append(annuity, annuity[-1] + df[-1] * af)
        self._df = pd.DataFrame(
            {"maturity": maturity[1:], "annuity": annuity[1:], "discountFactor": df[1:],
             "accrualFactor": afs})

    def _interpolate_spot_rates(self):
        day_since_start = np.array([d.days for d in self._sr.maturity - self._starting_date])
        interpolating_function = scipy.interpolate.interp1d(day_since_start, self._sr.spotRate,
                                                            kind=self.interpolation)

        date_to_interpolate = pd.date_range(self._sr.maturity.iloc[0], self._sr.maturity.iloc[-1])
        days_to_interpolate = np.array([d.days for d in date_to_interpolate - self._starting_date])

        af_since_trade = accrual_factor(self.dcc, self.trade_date, date_to_interpolate)

        spot_rates = pd.DataFrame({"maturity": date_to_interpolate, "term": af_since_trade,
                                   "spotRate": interpolating_function(days_to_interpolate)})

        # for maturity shorter than 1M, simply back-propagate the 1M euribor
        dates = pd.date_range(self.trade_date, date_to_interpolate[0], inclusive="left")
        lm = pd.DataFrame({"maturity": dates, "term": accrual_factor(self.dcc, dates[0], dates)})
        self._spot_rates = pd.concat([lm, spot_rates], ignore_index=True).bfill()

    def _get_df_from_spot_rate(self):
        maturity = self.spot_rates.maturity
        spot_rate = self.spot_rates.spotRate.to_numpy()
        term = self.spot_rates.term.to_numpy()
        match self.compounding:
            case "simple":
                self._discount_factors = pd.DataFrame(
                    {"maturity": maturity, "discountFactor": 1 / (1 + spot_rate * term)}).set_index("maturity")
            case "annually_compounded":
                self._discount_factors = pd.DataFrame(
                    {"maturity": maturity, "discountFactor": 1 / (1 + spot_rate) ** term}).set_index("maturity")
            case "continuous":
                self._discount_factors = pd.DataFrame(
                    {"maturity": maturity, "discountFactor": np.exp(-spot_rate * term)}).set_index("maturity")

    def __add__(self, other):
        if isinstance(other, EuriborCurve):
            self._sr = pd.concat([other.spot_rates, self._sr],
                                 ignore_index=True).drop_duplicates("maturity", keep="first")
        else:
            raise ValueError(f"DiscountCurve can be added only to EuriborCurve. Got '{other.__class__.__name__}'.")
        return self


class EuriborCurve:
    """
    Euribor curve object for money market rates.
    """
    _DCC = "ACT/360"
    _BUSINESS_CONVENTION = "modified_following"
    _SPOT_LEG = 2
    _MAPPING = {"1M": 1, "3M": 3, "6M": 6, "12M": 12}

    trade_date = Date()

    def __init__(self, euribor, trade_date):
        """
        Args:
            euribor (pandas.DataFrame): dataframe with euribor rate 1M, 3M, 6M and 12M.
            trade_date (str): "YYYY-MM-DD" or pandas.Timestamp for the starting date
        """
        self.trade_date = trade_date
        self.euribor = euribor
        self._discount_factors = None

    @property
    def euribor(self):
        return self._euribor

    @euribor.setter
    def euribor(self, euribor):
        self._euribor = euribor
        try:
            self._process_spot()
        except Exception:
            raise ValueError(Exception)

    @property
    def spot_rates(self):
        return self._spot_rates

    @spot_rates.setter
    def spot_rates(self, value):
        raise ValueError("Can't do this!")

    @property
    def discount_factors(self):
        if self._discount_factors is None:
            self._get_discount_factors()
            return self._discount_factors

    @discount_factors.setter
    def discount_factors(self, value):
        raise ValueError("Can't do this!")

    def __repr__(self):
        return f"EuriborCurve(trade_date={self.trade_date.strftime('%Y-%m-%d')})"

    def _process_spot(self):
        starting_date = self.trade_date + BDay(self._SPOT_LEG)
        maturity = [starting_date + DateOffset(months=months) for months in self.euribor.columns.map(self._MAPPING)]
        maturity = pd.Index(business_adjustment(self._BUSINESS_CONVENTION, maturity))
        self._spot_rates = pd.DataFrame({"maturity": maturity,
                                         "term": accrual_factor(self._DCC, self.trade_date, maturity),
                                         "spotRate": self.euribor.loc[self.trade_date]})

    def _get_discount_factors(self):
        term = accrual_factor(self._DCC, self.trade_date, self.spot_rates.maturity)
        spot_rate = self.spot_rates.spotRate.to_numpy()
        self._discount_factors = pd.DataFrame({"maturity": self.spot_rates.maturity,
                                               "discountFactor": 1 / (1 + spot_rate * term)})

    def __add__(self, other):

        if isinstance(other, DiscountCurve):
            other._sr = pd.concat([self.spot_rates, other.sr],
                                  ignore_index=True).drop_duplicates("maturity", keep="first")

        else:
            raise ValueError(f"EuriborCruve can be added only to DiscountCurve. Got '{other.__class__.__name__}'.")

        return other


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
                #self._forward_rates = (df1.divide(df2.to_numpy()) - 1).divide(af, axis=0).to_numpy().squeeze()
            case "continuous":
                self._forward_rates = (np.log(df1.to_numpy() / df2.to_numpy()) / af.reshape(-1, 1)).squeeze()
                # self._forward_rates = np.log(df1.divide(df2.to_numpy())).divide(af, axis=0).to_numpy().squeeze()
            case "annually_compounded":
                self._forward_rates = ((df1.to_numpy() / df2.to_numpy()) ** (1 / af.reshape(-1, 1)) - 1).squeeze()
                #self._forward_rates = ((df1.divide(df2.to_numpy())).pow(1 / af, axis=0) - 1).to_numpy().squeeze()


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
        if isinstance(volatility_surface, pandas.DataFrame):
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
                forward_rates = (df1.to_numpy() / df2.to_numpy()) ** (1 / af.reshape(-1,1)) - 1
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


class MertonSimulator:
    """
    Class for risk neutral calibration and simulation of the Merton model.
    """
    start_date = Date(sterilize_attr=["_simulated_short_rates", "_simulated_spot_rates",
                                      "_simulated_discount_factors", "_ttm"])
    nsim = PositiveInteger(sterilize_attr=["_simulated_short_rates", "_simulated_spot_rates",
                                           "_simulated_discount_factors", "_ttm"])
    seed = NonNegativeInteger(sterilize_attr=["_simulated_short_rates", "_simulated_spot_rates",
                                              "_simulated_discount_factors", "_ttm"], none_accepted=True)
    spot_rates = DataFrame(index_type=pandas.DatetimeIndex, sterilize_attr=["_simulated_short_rates",
                                                                            "_simulated_spot_rates",
                                                                            "_simulated_discount_factors", "_ttm"])

    def __init__(self, spot_rates, start_date, nsim, seed=None):
        """
        Args:
            spot_rates (pandas.Series | pandas.DataFrame): spot rates to be used for the risk-neutral calibration
            start_date (str | pandas.Timestamp): starting date for the simulation
            nsim (int): number of simulation
            seed (int): seed for reproducibility
        """
        self.spot_rates = spot_rates
        self.start_date = start_date
        self.nsim = nsim
        self.seed = seed
        self._simulated_short_rates = None
        self._simulated_spot_rates = None
        self._simulated_discount_factors = None
        self._ttm = None
        self._s = None
        self._mu = None
        self._r = None

    @property
    def ttm(self):
        if self._ttm is None:
            self._ttm = accrual_factor("ACT/365", self.start_date, self.spot_rates.index)
        return self._ttm

    @property
    def s(self):
        if self._s is None:
            self._param()
        return self._s

    @property
    def mu(self):
        if self._mu is None:
            self._param()
        return self._mu

    @property
    def r(self):
        if self._r is None:
            self._param()
        return self._r

    @property
    def simulated_short_rates(self):
        if self._simulated_short_rates is None:
            self._simulate_short_rates()
        return self._simulated_short_rates

    @property
    def simulated_spot_rates(self):
        if self._simulated_spot_rates is None:
            self._simulate_spot_rates()
        return self._simulated_spot_rates

    @property
    def simulated_discount_factors(self):
        if self._simulated_discount_factors is None:
            self._simulate_discount_factors()
        return self._simulated_discount_factors

    def _simulate_discount_factors(self):
        A = - self.mu / 2 * self.ttm ** 2 + self.s / 6 * self.ttm ** 3
        self._simulated_discount_factors = np.exp(-self.ttm * self.simulated_short_rates.T + A).T

    def _simulate_spot_rates(self):
        self._simulated_spot_rates = (-np.log(self.simulated_discount_factors.T) / self.ttm).T

    def _simulate_short_rates(self):
        dt = np.diff(self.ttm, prepend=0)
        np.random.seed(self.seed)
        # epsilon = np.random.randn(self.nsim, len(self.ttm))
        epsilon = np.random.randn(len(self.ttm), self.nsim)
        # short_rate = np.empty((self.nsim, len(self.ttm)))
        short_rate = np.empty((len(self.ttm), self.nsim))
        short_rate[0, :] = self.r
        for i in range(1, len(self.ttm + 1)):
            short_rate[i, :] = (short_rate[i - 1, :] + self.mu * dt[i - 1]
                                + (self.s * dt[i - 1]) ** 0.5 * epsilon[i - 1, :])
        self._simulated_short_rates = short_rate

    def _param(self):
        X = np.array([np.ones(len(self.ttm)), self.ttm / 2, -(self.ttm ** 2) / 6]).T
        self._r, self._mu, self._s = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(self.spot_rates.to_numpy()).squeeze()
        if self.s < 0:
            raise ValueError("Calibration failed. Sigma is negative.")
