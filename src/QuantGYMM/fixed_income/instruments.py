from __future__ import annotations
import copy
import numpy
import numpy as np
import pandas
import pandas as pd
from pandas.tseries.offsets import DateOffset, BDay
from ..utils import business_adjustment, accrual_factor, number_of_month
from scipy.optimize import minimize, brentq
from ..descriptors import *
from .pricers import Pricer
from ..term_structures import DiscountCurve
from ..calendar import Schedule
from .models import hw_b, hw_bermudan_bond_value, hw_trinomial_tree, hw_zcb_option

__all__ = ["FloatingRateBond", "VanillaSwap", "FixedRateBond", "ZeroCouponBond", "CallableBond"]


class FloatingRateBond:
    """
    Bond class for floating rate bond.
    """
    dcc = DayCountConvention(sterilize_attr=["_coupons_history"])
    face_amount = PositiveNumber(sterilize_attr=["_coupons_history"])
    fixing_days = NonNegativeInteger(sterilize_attr=["_coupons_history"])
    spread = FloatNumber(sterilize_attr=["_coupons_history"], none_accepted=True, return_if_none=0.0)
    cap = FloatNumber(sterilize_attr=["_coupons_history"], none_accepted=True, return_if_none=np.nan)
    floor = FloatNumber(sterilize_attr=["_coupons_history"], none_accepted=True, return_if_none=np.nan)
    currency = String(none_accepted=True, return_if_none="EUR")

    def __init__(self, schedule, dcc, face_amount, fixing_days, spread=0.0, floor=None, cap=None, currency=None):
        """
        Args:
            schedule (Schedule): schedule object for the coupons
            dcc (str): day count convention
            face_amount (int | float): bond face amount
            fixing_days (int): number of days previous to reset date on the fixing of coupon rate occurs
            spread (float): [optional] spread over the floating rate
            floor (float): [optional] floor rate for the coupon
            cap (float): [optional] cap rate for the coupon
            currency (str): [optional] face amount currency, defaults to EUR
        """
        self.dcc = dcc
        self.face_amount = face_amount
        self.fixing_days = fixing_days
        self.spread = spread
        self.floor = floor
        self.cap = cap
        self.currency = currency
        self.schedule = schedule
        self._evaluation_date = None
        self._pricer = None
        self._historical_euribor = None
        self._coupons_history = None
        self._hedging_instruments = None
        self._cds_spread = None
        self._recovery_rate = None
        self._survival_probabilities = None
        self._current_coupon_rate = None

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
            self._coupons_history = None
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
    def coupons_history(self):
        if self._coupons_history is None:
            self._coupons_history = self.get_coupons_history()
        return self._coupons_history

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

    @property
    def current_coupon_rate(self):
        return self._current_coupon_rate

    def set_current_coupon_rate(self, rate: float) -> None:
        """
        Override the current coupon rate that is estimated from historical euribor curve.
        This method is useful when there is no euribor curve.

        Args:
            rate (float): annualised coupon rate already fixed for the current period
                          (e.g. 0.0375 for 3.75%).
        """
        if rate is not None and not isinstance(rate, float):
            raise ValueError(f"'rate' must be a float or None. Got '{type(rate)}'.")
        self._current_coupon_rate = rate

        if self._pricer is not None:
            self._pricer._current_coupon = None
            self._pricer._expected_coupons = None

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
            self._coupons_history = None
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

    def set_discount_curve(self, discount_curve) -> None:
        """
        Convenience wiring that mirrors FixedRateBond/ZeroCouponBond: wraps the curve
        in a plain (cap/floor-unaware) Pricer and sets it on the bond. For bonds with
        caps or floors set an option-aware pricer via 'set_pricer' instead.
        Args:
            discount_curve (DiscountCurve): instance of DiscountCurve class
        """
        self.set_pricer(Pricer(discount_curve))

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
        self._coupons_history = None

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
        if len(hist_reset) == 0:
            self._coupons_history = pd.DataFrame(
                columns=["resetDate", "couponStart", "couponEnd", "accrualFactor",
                         "resetRate", "spread", "couponRate", "floorlet", "caplet", "coupon"],
                index=pd.RangeIndex(0, name="couponNumber"))
            return self._coupons_history

        hist_starting = self.schedule.schedule["startingDate"][past_date_mask]
        hist_payment = self.schedule.schedule["paymentDate"][past_date_mask]
        hist_rate = self.historical_euribor.loc[hist_reset].to_numpy().squeeze() + self.spread
        hist_accrual = af[past_date_mask]
        floorlet = np.maximum(self.floor - hist_rate, 0) * hist_accrual * self.face_amount
        caplet = np.maximum(hist_rate - self.cap, 0) * hist_accrual * self.face_amount
        self._coupons_history = pd.DataFrame(
            {"resetDate": hist_reset, "couponStart": hist_starting, "couponEnd": hist_payment,
             "accrualFactor": hist_accrual, "resetRate": hist_rate - self.spread, "spread": self.spread,
             "couponRate": hist_rate, "floorlet": floorlet, "caplet": -caplet,
             "coupon": np.nansum([hist_rate * hist_accrual * self.face_amount, floorlet, -caplet],
                                 axis=0)}, index=pd.RangeIndex(1, len(hist_rate) + 1, name="couponNumber")
        ).replace(np.nan, "-")
        return self.coupons_history

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

        if shift_type not in ("parallel", "slope", "curvature"):
            raise ValueError("Admitted shift types are: 'parallel', 'slope', 'curvature'.")

        if kind not in ("symmetric", "oneside"):
            raise ValueError("Admitted kind types are: 'symmetric', 'oneside'.")

        price_key = "riskAdjustedValue" if self._cds_spread else "riskFreeValue"
        sr_curve = self.pricer.discount_curve.rate_curve
        original_data = sr_curve.spot_rates_data.copy()
        n = len(original_data)

        def _node_shift(size: float):
            match shift_type:
                case "parallel":
                    return np.full(n, size)
                case "slope":
                    return np.linspace(size, -size, n)
                case "curvature":
                    x = np.linspace(0.0, 1.0, n)
                    return 8.0 * size * x * (x - 1.0) + size
            return None

        def _reprice() -> float:
            self.pricer._forward_rates = None  # force recomputation from mutated spot_rates_data
            self.pricer._expected_coupons = None
            return self.prices()[price_key]["dirtyPrice"]

        try:
            match kind:
                case "symmetric":
                    up = original_data.copy()
                    up.iloc[:, 0] += _node_shift(shift_size)
                    sr_curve.spot_rates_data = up
                    price_up = _reprice()
                    down = original_data.copy()
                    down.iloc[:, 0] += _node_shift(-shift_size)
                    sr_curve.spot_rates_data = down
                    price_down = _reprice()
                    return (price_up - price_down) / (2 * shift_size) * 0.0001

                case "oneside":
                    price = _reprice()  # base price with original data
                    shifted = original_data.copy()
                    shifted.iloc[:, 0] += _node_shift(shift_size)
                    sr_curve.spot_rates_data = shifted
                    price_shifted = _reprice()
                    return (price_shifted - price) / shift_size * 0.0001
        finally:
            sr_curve.spot_rates_data = original_data
            self.pricer._forward_rates = None  # leave Pricer caches clean (original data)
            self.pricer._expected_coupons = None

    def duration(self) -> float:
        """
        Compute the Macaulay Duration of the bond — the present-value-weighted average time
        to receive cash flows (coupons + face amount), in years.

        Returns:
            Macaulay Duration in years.
        """
        exp_coupons = self.pricer.expected_coupons
        dc = self.pricer.discount_curve
        coupon_ends = exp_coupons.couponEnd
        df = dc.discount_factor_at(coupon_ends)
        t = accrual_factor(dc.dcc, self.evaluation_date, coupon_ends)
        coupons = exp_coupons.coupon.to_numpy()
        face_pv = self.face_amount * df[-1]
        dirty_price = coupons.dot(df) + face_pv
        weighted = (t * coupons * df).sum() + t[-1] * face_pv
        return float(weighted / dirty_price)

    def modified_duration(self, shift_size=0.0001, kind="symmetric") -> float:
        """
        Compute the Modified Duration of the bond via a parallel finite difference on the
        underlying spot rate curve, normalised by the risk-free dirty price.

        Modified Duration ≈ -dP/dy / P, estimated as DV01_parallel * 10 000 / price.

        Args:
            shift_size (float): parallel shock size for finite difference (default 1bp).
            kind (str): 'symmetric' (centred, more accurate) or 'oneside'.
        Returns:
            Modified Duration (years per unit of yield change).
        """
        dv01 = self.sensitivity(shift_type="parallel", shift_size=shift_size, kind=kind)
        dirty_price = self.prices()["riskFreeValue"]["dirtyPrice"]
        return float(-dv01 * 10000 / dirty_price)

    def key_rate_dv01(self, shift_size=0.0001, kind="symmetric") -> dict:
        """
        Calculate the Key Rate DV01 (KRD) of the bond by shocking, one at a time, each node of
        the underlying SpotRateCurve's raw tenor grid (spot_rates_data) and re-pricing the bond
        with the shocked curve. The tenors used are determined automatically from the curve
        itself, not hardcoded — whatever nodes are present in 'spot_rates_data' are the ones
        shocked. Requires the bond's discount_curve to be built on a SpotRateCurve (i.e.
        discount_curve.rate_curve must expose 'spot_rates_data').

        Each node is shocked independently from a clean copy of the original curve data — the
        curve is never left in a shocked state between iterations, avoiding stale-state bugs.
        The curve is restored to its original state once at the end.

        Args:
            shift_size (float): the per-node shock size to apply, default is 1 basis point (0.0001).
            kind (str): finite difference approximation type — 'symmetric' shocks the node both up
                       and down and takes the centered difference (more accurate, O(h^2) error,
                       2 reprices per node); 'oneside' shocks the node only upward and compares
                       against the unshocked base price (faster, O(h) error, 1 reprice per node).
        Returns:
            dict mapping each curve node's maturity date to the corresponding key rate DV01
            (price sensitivity to a 1bp move at that specific node, expressed per basis point).
        """
        if kind not in ("symmetric", "oneside"):
            raise ValueError("Admitted kind types are: 'symmetric', 'oneside'.")

        sr_curve = self.pricer.discount_curve.rate_curve
        col = sr_curve.spot_rates_data.columns[0]
        original_data = sr_curve.spot_rates_data.copy()

        dc = self.pricer.discount_curve
        compounding = dc.compounding
        trade_date = dc.trade_date
        curve_dcc = dc.dcc
        bond_dcc = self.dcc
        face_amount = self.face_amount
        spread = self.spread
        frequency = self.schedule.frequency

        reset_dates = self.schedule.schedule["resetDate"]
        past_mask = reset_dates <= self.evaluation_date
        future_mask = reset_dates > self.evaluation_date
        if not past_mask.any():
            raise ValueError("Evaluation date precedes the first reset date: no current coupon exists yet.")

        future_resets = pd.DatetimeIndex(reset_dates[future_mask])
        df2_dates = pd.DatetimeIndex(
            business_adjustment("modified_following",
                                future_resets + DateOffset(months=12 / frequency))
        )

        current_end = self.schedule.schedule["paymentDate"][past_mask][-1]
        future_payments = pd.DatetimeIndex(self.schedule.schedule["paymentDate"][future_mask])
        all_payment_dates = pd.DatetimeIndex([current_end]).append(future_payments)

        n_future = len(future_resets)
        n_pay = len(all_payment_dates)

        af_fwd = accrual_factor(curve_dcc, future_resets, df2_dates)  # per forward rate
        af_coupon = accrual_factor(bond_dcc,
                                   self.schedule.schedule["startingDate"][future_mask],
                                   future_payments)  # per cedole future

        term_resets = accrual_factor(curve_dcc, trade_date, future_resets)
        term_df2 = accrual_factor(curve_dcc, trade_date, df2_dates)
        term_pay = accrual_factor(curve_dcc, trade_date, all_payment_dates)

        if self.current_coupon_rate is not None:
            current_coupon_rate = self.current_coupon_rate
        else:
            current_reset = reset_dates[past_mask][-1]
            current_coupon_rate = self.historical_euribor.loc[current_reset].item() + spread

        current_start = self.schedule.schedule["startingDate"][past_mask][-1]
        af_current = accrual_factor(bond_dcc, current_start, current_end).item()
        current_coupon_value = current_coupon_rate * af_current * face_amount

        all_dates = pd.DatetimeIndex(
            list(future_resets) + list(df2_dates) + list(all_payment_dates)
        )
        all_terms = np.concatenate([term_resets, term_df2, term_pay])

        def _df(rates, terms):
            match compounding:
                case "simple":
                    return 1.0 / (1.0 + rates * terms)
                case "annually_compounded":
                    return 1.0 / (1.0 + rates) ** terms
                case "continuous":
                    return np.exp(-rates * terms)

        def _fwd(df1, df2):
            match compounding:
                case "simple":
                    return (df1 / df2 - 1.0) / af_fwd
                case "annually_compounded":
                    return (df1 / df2) ** (1.0 / af_fwd) - 1.0
                case "continuous":
                    return np.log(df1 / df2) / af_fwd

        def _reprice() -> float:
            all_rates = sr_curve.rate_at(all_dates)
            all_dfs = _df(all_rates, all_terms)

            df1 = all_dfs[:n_future]
            df2 = all_dfs[n_future:2 * n_future]
            df_pay = all_dfs[2 * n_future:]

            fwd = _fwd(df1, df2)
            future_coupons = (fwd + spread) * af_coupon * face_amount
            all_coupons = np.concatenate([[current_coupon_value], future_coupons])

            return all_coupons.dot(df_pay) + face_amount * df_pay[-1]

        # CDS: survival prob dipende dalla curva → fallback a prices() completo
        def _reprice_cds() -> float:
            self.pricer._forward_rates = None
            self.pricer._expected_coupons = None
            return self.prices()["riskAdjustedValue"]["dirtyPrice"]

        _do_reprice = _reprice_cds if self._cds_spread else _reprice

        krd = {}
        working = original_data.copy()

        try:
            match kind:
                case "symmetric":
                    for tenor_date in original_data.index:
                        orig_val = original_data.at[tenor_date, col]

                        working.at[tenor_date, col] = orig_val + shift_size
                        sr_curve.spot_rates_data = working
                        price_up = _do_reprice()

                        working.at[tenor_date, col] = orig_val - shift_size
                        sr_curve.spot_rates_data = working
                        price_down = _do_reprice()

                        working.at[tenor_date, col] = orig_val
                        krd[tenor_date] = (price_up - price_down) / (2 * shift_size) * 0.0001

                case "oneside":
                    sr_curve.spot_rates_data = original_data
                    price_base = _do_reprice()
                    for tenor_date in original_data.index:
                        orig_val = original_data.at[tenor_date, col]
                        working.at[tenor_date, col] = orig_val + shift_size
                        sr_curve.spot_rates_data = working
                        krd[tenor_date] = (_do_reprice() - price_base) / shift_size * 0.0001
                        working.at[tenor_date, col] = orig_val

        finally:
            sr_curve.spot_rates_data = original_data
            self.pricer._forward_rates = None
            self.pricer._expected_coupons = None

        return krd

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
        payments = self.schedule.schedule["paymentDate"]
        future_payments = payments[payments > self.evaluation_date]
        ttm = accrual_factor("ACT/365", self.evaluation_date, future_payments)
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
                                             periods=int(fixed_cash_flow_num))

        floating_cash_flow_date = pd.date_range(end=self.maturity,
                                                freq=DateOffset(
                                                    years=1 / self.floating_leg_frequency // 1,
                                                    months=(1 / self.floating_leg_frequency % 1) * 12 // 1,
                                                    days=round((1 / self.floating_leg_frequency % 1) * 12 % 1 * 30)),
                                                periods=int(floating_cash_flow_num))

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
        shift_method = {
            "parallel": self.discount_curve.apply_parallel_shift,
            "slope": self.discount_curve.apply_slope_shift,
            "curvature": self.discount_curve.apply_curvature_shift,
        }
        if shift_type not in shift_method:
            raise ValueError("Admitted shift type are: 'parallel', 'slope' or 'curvature'.")

        self.discount_curve.reset_shift()
        self._calculate_swap_rate()
        match kind:
            case "symmetric":
                shift_method[shift_type](shift_size)
                price_up = self.market_price()
                self.discount_curve.reset_shift()
                shift_method[shift_type](-shift_size)
                price_down = self.market_price()
                self.discount_curve.reset_shift()
                return (price_up - price_down) / (2 * shift_size) * 0.0001
            case "oneside":
                price = self.market_price()
                shift_method[shift_type](shift_size)
                price_shift = self.market_price()
                self.discount_curve.reset_shift()
                return (price_shift - price) / shift_size * 0.0001
            case _:
                raise ValueError("Admitted kind types are: 'symmetric', 'oneside'")


class FixedRateBond:
    """
    Bond class for fixed rate bond.
    """
    dcc = DayCountConvention(sterilize_attr=["_cash_flows"])
    face_amount = PositiveNumber(sterilize_attr=["_cash_flows"])
    coupon_rate = FloatNumber(sterilize_attr=["_cash_flows"])
    redemption = FloatNumber(sterilize_attr=["_cash_flows"], none_accepted=True, return_if_none=100.0)
    currency = String(none_accepted=True, return_if_none="EUR")

    def __init__(self, schedule, dcc, face_amount, coupon_rate, redemption=None, currency=None):
        """
        Args:
            schedule (Schedule): schedule object for the coupons
            dcc (str): day count convention
            face_amount (int | float): bond face amount
            coupon_rate (float): fixed annual coupon rate (e.g. 0.0375 for 3.75%)
            redemption (float): [optional] redemption price as a percentage of face amount, default is 100.0
            currency (str): [optional] face amount currency, defaults to EUR
        """

        self.dcc = dcc
        self.face_amount = face_amount
        self.coupon_rate = coupon_rate
        self.redemption = redemption
        self.currency = currency
        self.schedule = schedule
        self._evaluation_date = None
        self._discount_curve = None
        self._coupons_history = None
        self._cash_flows = None
        self._cds_spread = None
        self._recovery_rate = None
        self._survival_probabilities = None

    @property
    def schedule(self):
        return self._schedule

    @schedule.setter
    def schedule(self, schedule):
        if isinstance(schedule, Schedule):
            self._schedule = copy.deepcopy(schedule)
            self._cash_flows = None
        else:
            raise ValueError(f"'{schedule}' is not a Schedule object.")

    @property
    def evaluation_date(self):
        if self._evaluation_date is None:
            raise ValueError("Evaluation date has not been set. Call 'set_evaluation_date' method to set it.")
        return self._evaluation_date

    @property
    def discount_curve(self):
        if self._discount_curve is None:
            raise ValueError("No discount curve set. Call 'set_discount_curve' method to set it.")
        return self._discount_curve

    @property
    def cash_flows(self):
        if self._cash_flows is None:
            self._cash_flows = self._get_cash_flows()
        return self._cash_flows

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

    @property
    def coupons_history(self):
        if self._coupons_history is None:
            self._coupons_history = self.get_coupons_history()
        return self._coupons_history

    def __repr__(self):
        return f"FixedRateBond(faceAmount={self.face_amount}, couponRate={self.coupon_rate}, " \
               f"maturity={self.schedule.schedule['paymentDate'][-1].strftime(format('%Y-%m-%d'))}, " \
               f"redemption={self.redemption})"

    def set_evaluation_date(self, date) -> None:
        """
        Set evaluation date for market price calculation.
        Args:
            date (str | pandas.Timestamp): trade date
        """
        try:
            self._evaluation_date = pd.to_datetime(date)
            self._cash_flows = None
            self._coupons_history = None
        except Exception:
            raise ValueError(f"Can't convert {date} to datetime.")

    def set_discount_curve(self, discount_curve) -> None:
        """
        Set the discount curve to be used in the market value calculation.
        Args:
            discount_curve (DiscountCurve): instance of DiscountCurve class
        """
        if isinstance(discount_curve, DiscountCurve):
            self._discount_curve = discount_curve
        else:
            raise ValueError("'discount_curve' must be a DiscountCurve object.")

    def _get_cash_flows(self) -> pd.DataFrame:
        """
        Build the future cash flow schedule (coupons + redemption) as of the evaluation date.
        Returns:
            pandas.DataFrame of coupon start, coupon end, accrual factor and cash flow amount.
        """
        starts = self.schedule.schedule["startingDate"]
        payments = self.schedule.schedule["paymentDate"]
        future_mask = payments > self.evaluation_date
        starts, payments = starts[future_mask], payments[future_mask]
        af = accrual_factor(self.dcc, starts, payments)
        coupon = self.coupon_rate * af * self.face_amount
        if len(coupon) == 0:
            self._cash_flows = pd.DataFrame(
                columns=["couponStart", "couponEnd", "accrualFactor", "cashFlow"]
            )
            return self._cash_flows
        coupon[-1] += self.redemption / 100 * self.face_amount
        self._cash_flows = pd.DataFrame(
            {"couponStart": starts, "couponEnd": payments, "accrualFactor": af, "cashFlow": coupon},
            index=pd.RangeIndex(1, len(payments) + 1, name="couponNumber")
        )
        return self._cash_flows

    def get_coupons_history(self) -> pd.DataFrame:
        """
        Calculate the past history of coupons (deterministic for a fixed rate bond).
        Returns:
            pandas.DataFrame of coupon start, coupon end, accrual factor and cash flow.
        """
        starts = self.schedule.schedule["startingDate"]
        payments = self.schedule.schedule["paymentDate"]
        past_mask = payments <= self.evaluation_date
        hist_starting = starts[past_mask]
        hist_payments = payments[past_mask]
        af = accrual_factor(self.dcc, hist_starting, hist_payments)
        coupon = self.coupon_rate * af * self.face_amount

        self._coupons_history = pd.DataFrame(
            {"couponStart": hist_starting, "couponEnd": hist_payments, "accrualFactor": af, "cashFlow": coupon},
            index=pd.RangeIndex(1, len(hist_payments) + 1, name="couponNumber")
        )
        return self._coupons_history

    def prices(self) -> dict:
        """
        Compute fair market price as the sum of discounted future cash flows.
        Returns:
            dict with risk-free (and, if a CDS spread/recovery rate are set, risk-adjusted) dirty/clean price.
        """
        df = self.discount_curve.discount_factor_at(self.cash_flows.couponEnd)
        # df = self.discount_curve.discount_factors.loc[self.cash_flows.couponEnd].to_numpy().squeeze()
        start, end = self.cash_flows.couponStart.iloc[0], self.cash_flows.couponEnd.iloc[0]
        # Accrued = current-period coupon (rate × period accrual factor × face) scaled
        # by the elapsed fraction of the period at settlement (T+2). Scaling the full
        # annual rate instead would overstate accrued by a factor of `frequency` for
        # non-annual coupons.
        period_coupon = self.coupon_rate * accrual_factor(self.dcc, start, end).item() * self.face_amount
        elapsed = (self.evaluation_date + BDay(2) - start).days / (end - start).days
        accrued_interest = period_coupon * min(max(elapsed, 0.0), 1.0)

        cash_flow_pv = self.cash_flows.cashFlow.to_numpy().dot(df)
        prices = {"riskFreeValue": {"dirtyPrice": cash_flow_pv.item(),
                                    "accruedInterest": accrued_interest,
                                    "cleanPrice": (cash_flow_pv - accrued_interest).item()}}

        if self._cds_spread:
            if self._recovery_rate is None:
                raise ValueError("CDS spread set but no recovery rate found. Call 'set_recovery_rate'.")
            coupon_only = self.cash_flows.cashFlow.to_numpy().copy()
            coupon_only[-1] -= self.redemption / 100 * self.face_amount  # strip redemption out of last cash flow
            survival = self.survival_probabilities
            delta_prob = np.diff(-survival, prepend=-1)

            coupon_pv_on_survival = (coupon_only * survival).dot(df)
            coupon_pv_on_default = (self.recovery_rate * delta_prob).dot(df) * self.face_amount
            redemption_pv_on_survival = self.redemption / 100 * self.face_amount * df[-1] * survival[-1]

            risk_adj_dirty = coupon_pv_on_survival + coupon_pv_on_default + redemption_pv_on_survival
            prices = {**prices,
                      "riskAdjustedValue": {"dirtyPrice": risk_adj_dirty.item(),
                                            "accruedInterest": accrued_interest,
                                            "cleanPrice": (risk_adj_dirty - accrued_interest).item()}}
        return prices

    def key_rate_dv01(self, shift_size=0.0001, kind="symmetric") -> dict:
        """
        Calculate the Key Rate DV01 (KRD) of the bond by shocking, one at a time, each node of
        the underlying SpotRateCurve's raw tenor grid (spot_rates_data) and re-pricing the bond
        with the shocked curve. The tenors used are determined automatically from the curve
        itself, not hardcoded — whatever nodes are present in 'spot_rates_data' are the ones
        shocked. Requires the bond's discount_curve to be built on a SpotRateCurve (i.e.
        discount_curve.rate_curve must expose 'spot_rates_data').

        Each node is shocked independently from a clean copy of the original curve data — the
        curve is never left in a shocked state between iterations, avoiding stale-state bugs.
        The curve is restored to its original state once at the end.

        Args:
            shift_size (float): the per-node shock size to apply, default is 1 basis point (0.0001).
            kind (str): finite difference approximation type — 'symmetric' shocks the node both up
                       and down and takes the centered difference (more accurate, O(h^2) error,
                       2 reprices per node); 'oneside' shocks the node only upward and compares
                       against the unshocked base price (faster, O(h) error, 1 reprice per node).
        Returns:
            dict mapping each curve node's maturity date to the corresponding key rate DV01
            (price sensitivity to a 1bp move at that specific node, expressed per basis point).
        """
        if kind not in ("symmetric", "oneside"):
            raise ValueError("Admitted kind types are: 'symmetric', 'oneside'.")

        sr_curve = self.discount_curve.rate_curve
        col = sr_curve.spot_rates_data.columns[0]
        original_data = sr_curve.spot_rates_data.copy()

        dates = self.cash_flows.couponEnd
        term = accrual_factor(self.discount_curve.dcc, self.discount_curve.trade_date, dates)
        cf_arr = self.cash_flows.cashFlow.to_numpy()
        compounding = self.discount_curve.compounding

        def _reprice() -> float:

            rates = sr_curve.rate_at(dates)
            match compounding:
                case "simple":
                    return cf_arr.dot(1.0 / (1.0 + rates * term))
                case "annually_compounded":
                    return cf_arr.dot(1.0 / (1.0 + rates) ** term)
                case "continuous":
                    return cf_arr.dot(np.exp(-rates * term))
                case _:
                    raise ValueError(f"Unknown compounding: {compounding}")

        if self._cds_spread:
            coupon_only = cf_arr.copy()
            coupon_only[-1] -= self.redemption / 100 * self.face_amount
            survival = self.survival_probabilities
            delta_prob = np.diff(-survival, prepend=-1)
            redemption_notional = self.redemption / 100 * self.face_amount

            def _reprice_cds() -> float:
                rates = sr_curve.rate_at(dates)
                match compounding:
                    case "simple":
                        df = 1.0 / (1.0 + rates * term)
                    case "annually_compounded":
                        df = 1.0 / (1.0 + rates) ** term
                    case "continuous":
                        df = np.exp(-rates * term)
                    case _:
                        raise ValueError(f"Unknown compounding: {compounding}")
                coupon_pv = (coupon_only * survival).dot(df)
                default_pv = (self.recovery_rate * delta_prob).dot(df) * self.face_amount
                redempt_pv = redemption_notional * df[-1] * survival[-1]
                return coupon_pv + default_pv + redempt_pv

            _do_reprice = _reprice_cds
        else:
            _do_reprice = _reprice

        krd = {}
        working = original_data.copy()

        try:
            match kind:
                case "symmetric":
                    for tenor_date in original_data.index:
                        orig_val = original_data.at[tenor_date, col]

                        working.at[tenor_date, col] = orig_val + shift_size
                        sr_curve.spot_rates_data = working  # invalida _interpolator/_sr
                        price_up = _do_reprice()

                        working.at[tenor_date, col] = orig_val - shift_size
                        sr_curve.spot_rates_data = working
                        price_down = _do_reprice()

                        working.at[tenor_date, col] = orig_val  # restore nodo in working

                        krd[tenor_date] = (price_up - price_down) / (2 * shift_size) * 0.0001

                case "oneside":
                    sr_curve.spot_rates_data = original_data
                    price_base = _do_reprice()
                    for tenor_date in original_data.index:
                        orig_val = original_data.at[tenor_date, col]
                        working.at[tenor_date, col] = orig_val + shift_size
                        sr_curve.spot_rates_data = working
                        krd[tenor_date] = (_do_reprice() - price_base) / shift_size * 0.0001
                        working.at[tenor_date, col] = orig_val

        finally:
            sr_curve.spot_rates_data = original_data
            self.discount_curve.rate_curve = sr_curve

        return krd

    def sensitivity(self, shift_type="parallel", shift_size=0.01, kind="symmetric") -> float:
        """
        Calculate the DV01 of the bond to different types of curve shift by means of finite difference approximation.
        Args:
            shift_type (str): type of term structure shift (valid inputs are 'parallel', 'slope', 'curvature').
            shift_size (float): the term structure shift size to be applied to estimate the bond first derivatives.
            kind (str): finite difference approximation type (valid inputs are 'symmetric', 'oneside').
        Returns:
            The estimated bond DV01.
        """
        if shift_type not in ("parallel", "slope", "curvature"):
            raise ValueError("Admitted shift types are: 'parallel', 'slope', 'curvature'.")
        if kind not in ("symmetric", "oneside"):
            raise ValueError("Admitted kind types are: 'symmetric', 'oneside'.")

        price_key = "riskAdjustedValue" if self._cds_spread else "riskFreeValue"
        sr_curve = self.discount_curve.rate_curve
        original_data = sr_curve.spot_rates_data.copy()
        n = len(original_data)

        def _node_shift(size: float):

            match shift_type:
                case "parallel":
                    return np.full(n, size)
                case "slope":
                    return np.linspace(size, -size, n)
                case "curvature":
                    x = np.linspace(0.0, 1.0, n)
                    return 8.0 * size * x * (x - 1.0) + size  # f(0)=size, f(0.5)=-size, f(1)=size
            return None

        try:
            match kind:
                case "symmetric":
                    up = original_data.copy()
                    up.iloc[:, 0] += _node_shift(shift_size)
                    sr_curve.spot_rates_data = up
                    price_up = self.prices()[price_key]["dirtyPrice"]

                    down = original_data.copy()
                    down.iloc[:, 0] += _node_shift(-shift_size)
                    sr_curve.spot_rates_data = down
                    price_down = self.prices()[price_key]["dirtyPrice"]

                    return (price_up - price_down) / (2 * shift_size) * 0.0001

                case "oneside":
                    price = self.prices()[price_key]["dirtyPrice"]

                    shifted = original_data.copy()
                    shifted.iloc[:, 0] += _node_shift(shift_size)
                    sr_curve.spot_rates_data = shifted
                    price_shifted = self.prices()[price_key]["dirtyPrice"]

                    return (price_shifted - price) / shift_size * 0.0001

        finally:
            sr_curve.spot_rates_data = original_data
            self.discount_curve.rate_curve = sr_curve

    def duration(self) -> float:
        """
        Compute the Macaulay Duration of the bond — the present-value-weighted average time
        to receive cash flows (coupons + redemption), in years.

        Returns:
            Macaulay Duration in years.
        """
        df = self.discount_curve.discount_factor_at(self.cash_flows.couponEnd)
        t = accrual_factor(self.discount_curve.dcc, self.evaluation_date, self.cash_flows.couponEnd)
        cf = self.cash_flows.cashFlow.to_numpy()
        dirty_price = cf.dot(df)
        return float((t * cf * df).sum() / dirty_price)

    def modified_duration(self, shift_size=0.0001, kind="symmetric") -> float:
        """
        Compute the Modified Duration of the bond via a parallel finite difference on the
        underlying spot rate curve, normalised by the risk-free dirty price.

        Modified Duration ≈ -dP/dy / P, estimated as DV01_parallel * 10 000 / price.

        Args:
            shift_size (float): parallel shock size for finite difference (default 1bp).
            kind (str): 'symmetric' (centred, more accurate) or 'oneside'.
        Returns:
            Modified Duration (years per unit of yield change).
        """
        dv01 = self.sensitivity(shift_type="parallel", shift_size=shift_size, kind=kind)
        dirty_price = self.prices()["riskFreeValue"]["dirtyPrice"]
        return float(-dv01 * 10000 / dirty_price)

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
        if not isinstance(recovery_rate, (np.ndarray, list, float)) and recovery_rate is not None:
            raise ValueError("Wrong type for 'recovery_rate': it must be a float or an array.")
        self._survival_probabilities = None
        self._recovery_rate = recovery_rate

    def _get_survival_prob(self):
        ttm = accrual_factor("ACT/365", self.evaluation_date, self.cash_flows["couponEnd"])
        self._survival_probabilities = (np.exp(-self.cds_spread * ttm) - self.recovery_rate) / (1 - self.recovery_rate)


class CallableBond:
    """
    Wraps a FixedRateBond or FloatingRateBond with a Bermudan-style call schedule.
    Supports price-to-worst valuation and, for a single future call date on fixed-rate
    underlying, closed-form Hull-White valuation.
    """

    def __init__(self, bond, call_schedule, mean_reversion=0.03, volatility=0.1):
        """
        Args:
            bond (FixedRateBond | FloatingRateBond): the underlying non-callable bond.
            call_schedule (pandas.Series): index = call dates (must coincide with the
                                           bond's coupon payment dates), values = clean
                                           call price as % of face (e.g. 100.0 = at par).
            mean_reversion (float): Hull-White mean reversion speed 'a' (positive).
            volatility (float): Hull-White short rate volatility 'sigma' (positive).
        """
        if not isinstance(bond, (FixedRateBond, FloatingRateBond)):
            raise ValueError("'bond' must be a FixedRateBond or FloatingRateBond object.")
        self.bond = bond

        if not isinstance(call_schedule, pd.Series):
            raise ValueError("'call_schedule' must be a pandas.Series.")
        if not isinstance(call_schedule.index, pd.DatetimeIndex):
            raise ValueError("'call_schedule' index must be a pandas.DatetimeIndex.")
        payment_dates = pd.DatetimeIndex(bond.schedule.schedule["paymentDate"])

        if not call_schedule.index.isin(payment_dates).all():
            raise ValueError("Call dates must coincide with coupon payment dates.")
        self._call_schedule = call_schedule.sort_index()

        if not isinstance(mean_reversion, float) or mean_reversion <= 0:
            raise ValueError("'mean_reversion' must be a positive float.")
        if not isinstance(volatility, float) or volatility <= 0:
            raise ValueError("'volatility' must be a positive float.")
        self._mean_reversion = mean_reversion
        self._volatility = volatility

    @property
    def face_amount(self):
        return self.bond.face_amount

    @property
    def currency(self):
        return self.bond.currency

    @property
    def evaluation_date(self):
        return self.bond.evaluation_date

    @property
    def schedule(self):
        return self.bond.schedule

    @property
    def mean_reversion(self):
        return self._mean_reversion

    @property
    def volatility(self):
        return self._volatility

    def set_hw_params(self, mean_reversion=None, volatility=None):
        """
        Update Hull-White parameters.
        Args:
            mean_reversion (float): Hull-White mean reversion speed 'a'
            volatility (float): Hull-White short rate volatility 'sigma'
        """
        if mean_reversion is not None:
            if not isinstance(mean_reversion, float) or mean_reversion <= 0:
                raise ValueError("'mean_reversion' must be a positive float.")
            self._mean_reversion = mean_reversion
        if volatility is not None:
            if not isinstance(volatility, float) or volatility <= 0:
                raise ValueError("'volatility' must be a positive float.")
            self._volatility = volatility

    @property
    def call_schedule(self):
        """
        Returns:
            pandas.Series of future call dates (index > evaluation_date).
        """
        future = self._call_schedule[self._call_schedule.index > self.evaluation_date]
        if future.empty:
            raise ValueError("All call dates are in the past.")
        return future

    def _discount_curve(self):
        if isinstance(self.bond, FixedRateBond):
            return self.bond.discount_curve
        return self.bond.pricer.discount_curve

    def _future_cash_flows(self):
        """
        Normalize both underlying bond types to one shape: coupon-only cash flows
        (redemption/face amount excluded) and their past payment dates.
        Returns:
            (pandas.DatetimeIndex, numpy.ndarray) - dates, coupon amounts
        """

        if isinstance(self.bond, FixedRateBond):
            cf = self.bond.cash_flows
            coupons = cf.cashFlow.to_numpy().copy()
            coupons[-1] -= self.bond.redemption / 100 * self.bond.face_amount
            dates = pd.DatetimeIndex(cf.couponEnd)
        else:
            ec = self.bond.pricer.expected_coupons
            coupons = ec.coupon.to_numpy().astype(float)
            dates = pd.DatetimeIndex(ec.couponEnd)
        return dates, coupons

    def prices(self, method="worst") -> dict:
        """
        Compute straight, option and callable value of the bond.
        Args:
            method (str): 'worst' (price-to-worst, both underlying types), 'hw'
            (Hull-White Jamshidian, fixed rate underlying with a single future
             call date only) or 'tree' (Hull-White trinomial tree, fixed rate
             underlying, any number of call dates).
        """

        if self.bond._cds_spread:
            raise NotImplementedError("CallableBond does not support CDS-adjusted valuation yet.")

        straight = self.bond.prices()["riskFreeValue"]

        if method == "worst":
            option_value = self._option_value_worst(straight["dirtyPrice"])
        elif method == "hw":
            option_value = self._option_value_hw()
        elif method == "tree":
            option_value = self._option_value_tree()
        else:
            raise ValueError("'method' must be one of 'worst', 'hw', 'tree'.")

        callable_dirty = straight["dirtyPrice"] - option_value

        return {
            "straightValue": straight,
            "optionValue": option_value,
            "callableValue": {
                "dirtyPrice": callable_dirty,
                "accruedInterest": straight["accruedInterest"],
                "cleanPrice": callable_dirty - straight["accruedInterest"],
            },
        }

    def _option_value_tree(self, step_days=7) -> float:
        """
        Issuer's call option value from a Hull-White trinomial tree.

        Unlike the Jamshidian decomposition, which prices a single European
        exercise, the tree caps the bond value at every call date, so it handles
        Bermudan schedules and a call price that changes date by date - the two
        features high yield and hybrid capital paper actually have.

        The option is returned as the difference between two valuations on the
        SAME tree, one with the call cap and one without. Any discretisation
        error common to both cancels, so the option value stays clean even
        though the grid only approximates the exact payment dates.

        Args:
            step_days (int): calendar days per tree step. It is the knob that
                drives accuracy: cash flows land on the nearest node, so the
                timing error is at most half a step. Convergence is monotone in
                spirit but bumpy in practice, because the snapping jumps as the
                step changes by whole days. Seven days keeps the option value
                within a few hundredths of a percent of the closed form on a
                single call, at a fraction of the cost of a daily grid.
        """
        if isinstance(self.bond, FloatingRateBond):
            raise NotImplementedError("Tree valuation is only available for fixed rate underlyings.")

        dc = self._discount_curve()
        dates, coupons = self._future_cash_flows()
        redemption = self.bond.redemption / 100 * self.bond.face_amount

        # La griglia deve stare su un numero INTERO di giorni. Le convenzioni di
        # conteggio troncano ai giorni (act365 fa (d1 - d0).days), quindi una
        # griglia uniforme nel tempo non e' esprimibile in date: i fattori di
        # sconto cadrebbero su date arrotondate mentre l'albero assume un passo
        # esatto, e le due cose smettono di parlarsi.
        horizon_days = (dates[-1] - dc.trade_date).days
        step_days = max(1, int(step_days))
        n_steps = int(np.ceil(horizon_days / step_days))
        dt = accrual_factor(dc.dcc, dc.trade_date, dc.trade_date + pd.Timedelta(days=step_days)).item()

        grid_dates = pd.DatetimeIndex(
            [dc.trade_date + pd.Timedelta(days=i * step_days) for i in range(n_steps + 1)]
        )
        discount_factors = np.empty(n_steps + 1)
        discount_factors[0] = 1.0
        discount_factors[1:] = dc.discount_factor_at(grid_dates[1:])

        # Ogni flusso finisce sul nodo piu' vicino, con uno scarto di al massimo
        # mezzo passo. L'errore e' identico nelle due valutazioni e si elide
        # nella differenza.
        def slot(date):
            return min(max(round((date - dc.trade_date).days / step_days), 1), n_steps)

        cash_flows = np.zeros(n_steps + 1)
        for date, amount in zip(dates, coupons):
            cash_flows[slot(date)] += amount
        cash_flows[slot(dates[-1])] += redemption

        call_prices = np.full(n_steps + 1, np.nan)
        for call_date, clean_price in self.call_schedule.items():
            call_prices[slot(call_date)] = clean_price / 100 * self.bond.face_amount

        tree = hw_trinomial_tree(dt, n_steps, self.mean_reversion, self.volatility, discount_factors)
        straight_on_tree = hw_bermudan_bond_value(tree, dt, cash_flows)
        callable_on_tree = hw_bermudan_bond_value(tree, dt, cash_flows, call_prices)
        return straight_on_tree - callable_on_tree

    def _option_value_worst(self, straight_dirty) -> float:
        dc = self._discount_curve()
        dates, coupons = self._future_cash_flows()

        pvs = [straight_dirty]

        for call_date, clean_price in self.call_schedule.items():
            mask = dates <= call_date
            called_dates = dates[mask].append(pd.DatetimeIndex([call_date]))
            called_cfs = np.append(coupons[mask], clean_price / 100 * self.face_amount)
            df = dc.discount_factor_at(called_dates)
            pvs.append(called_cfs.dot(df))

        worst = min(pvs)
        return straight_dirty - worst

    def _option_value_hw(self) -> float:
        if isinstance(self.bond, FloatingRateBond):
            raise NotImplementedError("Hull-White valuation is only available for fixed rate underlyings.")

        call_sched = self.call_schedule
        if len(call_sched) > 1:
            raise NotImplementedError("Bermudan callables not supported yet; use method='worst'.")

        call_date = call_sched.index[0]
        K = call_sched.iloc[0] / 100.0

        dc = self._discount_curve()
        a = self.mean_reversion
        sigma = self.volatility

        # t_call = option expiry (call date), t_cf = each cash flow date after the
        # call — named to match hw_b(t, T, a)'s (valuation, target) argument order.
        t_call = accrual_factor(dc.dcc, dc.trade_date, call_date).item()

        cf = self.bond.cash_flows
        mask = pd.DatetimeIndex(cf.couponEnd) > call_date
        c = cf.cashFlow.to_numpy()[mask] / self.face_amount
        t_dates = pd.DatetimeIndex(cf.couponEnd)[mask]
        t_cf = accrual_factor(dc.dcc, dc.trade_date, t_dates)

        df_T = dc.discount_factor_at(pd.DatetimeIndex([call_date])).item()
        df_t = dc.discount_factor_at(t_dates)

        eps = 1.0 / 365.0
        df_up = dc.discount_factor_at(pd.DatetimeIndex([call_date + pd.Timedelta(days=1)])).item()
        df_down = dc.discount_factor_at(pd.DatetimeIndex([call_date - pd.Timedelta(days=1)])).item()
        f0T = -(np.log(df_up) - np.log(df_down)) / (2.0 * eps)

        B = hw_b(t_call, t_cf, a)
        lnA = (np.log(df_t / df_T) + B * f0T
               - (sigma ** 2 / (4.0 * a)) * B ** 2 * (1.0 - np.exp(-2.0 * a * t_call)))

        def _bond_price(r):
            return np.exp(lnA - B * r)

        def _f(r):
            return c.dot(_bond_price(r)) - K

        lo, hi = -1.0, 1.0
        while _f(lo) * _f(hi) > 0 and max(abs(lo), abs(hi)) <= 10.0:
            lo *= 2.0
            hi *= 2.0
        if _f(lo) * _f(hi) > 0:
            raise ValueError("Could not bracket root for r* in Jamshidian decomposition.")

        r_star = brentq(_f, lo, hi)
        K_i = _bond_price(r_star)

        option_per_unit = sum(
            ci * hw_zcb_option(df_T, df_ti, Ki, t_call, ti, a, sigma, kind="call")
            for ci, df_ti, Ki, ti in zip(c, df_t, K_i, t_cf)
        )
        return option_per_unit * self.face_amount

    def option_value(self, method="worst") -> float:
        """
        Returns:
            The call option value (per the wrapped bond's face amount).
        """
        return self.prices(method)["optionValue"]

    def bullet_price(self, call_index=0) -> dict:
        """
        Price the bond as a bullet maturing at the given call date, with redemption
        set to that call's price. No option-adjustment: assumes the issuer always
        calls (standard convention for capital instruments where non-call is a
        stress signal, not an economic decision). Fixed-rate underlying only.

        Args:
            call_index (int): index into 'call_schedule' of the call date to bullet to
                              (0 = earliest future call date).
        Returns:
            Same dict shape as FixedRateBond.prices() (riskFreeValue, ...).
        """
        if isinstance(self.bond, FloatingRateBond):
            raise NotImplementedError("Bullet-to-call is only implemented for fixed rate underlyings.")

        call_date = self.call_schedule.index[call_index]
        clean_price = self.call_schedule.iloc[call_index]

        truncated = copy.deepcopy(self.bond)
        sched = truncated.schedule.schedule
        mask = sched["paymentDate"] <= call_date
        truncated.schedule._schedule = {k: v[mask] for k, v in sched.items()}
        truncated._cash_flows = None
        truncated.redemption = float(clean_price)
        truncated.set_evaluation_date(self.evaluation_date)
        truncated.set_discount_curve(self._discount_curve())
        return truncated.prices()

    def sensitivity(self, shift_type="parallel", shift_size=0.01, kind="symmetric", method="worst") -> float:
        """
        Calculate the DV01 of the callable bond to different types of curve shift by
        means of finite difference approximation.
        Args:
            shift_type (str): 'parallel', 'slope' or 'curvature'.
            shift_size (float): term structure shift size.
            kind (str): 'symmetric' or 'oneside'.
            method (str): 'worst' or 'hw'.
        Returns:
            The estimated callable bond DV01.
        """
        if shift_type not in ("parallel", "slope", "curvature"):
            raise ValueError("Admitted shift types are: 'parallel', 'slope', 'curvature'.")
        if kind not in ("symmetric", "oneside"):
            raise ValueError("Admitted kind types are: 'symmetric', 'oneside'.")

        sr_curve = self._discount_curve().rate_curve
        if not hasattr(sr_curve, "spot_rates_data"):
            raise ValueError("sensitivity requires a SpotRateCurve-based discount curve.")

        original_data = sr_curve.spot_rates_data.copy()
        n = len(original_data)

        def _node_shift(size):
            match shift_type:
                case "parallel":
                    return np.full(n, size)
                case "slope":
                    return np.linspace(size, -size, n)
                case "curvature":
                    x = np.linspace(0.0, 1.0, n)
                    return 8.0 * size * x * (x - 1.0) + size
            return None

        def _reprice() -> float:
            if isinstance(self.bond, FloatingRateBond):
                self.bond.pricer._forward_rates = None
                self.bond.pricer._expected_coupons = None
            return self.prices(method)["callableValue"]["dirtyPrice"]

        try:
            match kind:
                case "symmetric":
                    up = original_data.copy()
                    up.iloc[:, 0] += _node_shift(shift_size)
                    sr_curve.spot_rates_data = up
                    price_up = _reprice()

                    down = original_data.copy()
                    down.iloc[:, 0] += _node_shift(-shift_size)
                    sr_curve.spot_rates_data = down
                    price_down = _reprice()

                    return (price_up - price_down) / (2 * shift_size) * 0.0001

                case "oneside":
                    price = _reprice()
                    shifted = original_data.copy()
                    shifted.iloc[:, 0] += _node_shift(shift_size)
                    sr_curve.spot_rates_data = shifted
                    price_shifted = _reprice()

                    return (price_shifted - price) / shift_size * 0.0001
        finally:
            sr_curve.spot_rates_data = original_data
            if isinstance(self.bond, FloatingRateBond):
                self.bond.pricer._forward_rates = None
                self.bond.pricer._expected_coupons = None

    def key_rate_dv01(self, shift_size=0.0001, kind="symmetric", method="worst") -> dict:
        """
        Calculate the Key Rate DV01 of the callable bond by shocking, one at a time,
        each node of the underlying SpotRateCurve's raw tenor grid.
        Args:
            shift_size (float): per-node shock size, default 1bp.
            kind (str): 'symmetric' or 'oneside'.
            method (str): 'worst' or 'hw'.
        Returns:
            dict mapping each curve node's maturity date to the key rate DV01.
        """
        if kind not in ("symmetric", "oneside"):
            raise ValueError("Admitted kind types are: 'symmetric', 'oneside'.")

        sr_curve = self._discount_curve().rate_curve
        if not hasattr(sr_curve, "spot_rates_data"):
            raise ValueError("sensitivity requires a SpotRateCurve-based discount curve.")

        col = sr_curve.spot_rates_data.columns[0]
        original_data = sr_curve.spot_rates_data.copy()

        def _reprice() -> float:
            if isinstance(self.bond, FloatingRateBond):
                self.bond.pricer._forward_rates = None
                self.bond.pricer._expected_coupons = None
            return self.prices(method)["callableValue"]["dirtyPrice"]

        krd = {}
        working = original_data.copy()

        try:
            match kind:
                case "symmetric":
                    for tenor_date in original_data.index:
                        orig_val = original_data.at[tenor_date, col]

                        working.at[tenor_date, col] = orig_val + shift_size
                        sr_curve.spot_rates_data = working
                        price_up = _reprice()

                        working.at[tenor_date, col] = orig_val - shift_size
                        sr_curve.spot_rates_data = working
                        price_down = _reprice()

                        working.at[tenor_date, col] = orig_val
                        krd[tenor_date] = (price_up - price_down) / (2 * shift_size) * 0.0001

                case "oneside":
                    sr_curve.spot_rates_data = original_data
                    price_base = _reprice()
                    for tenor_date in original_data.index:
                        orig_val = original_data.at[tenor_date, col]
                        working.at[tenor_date, col] = orig_val + shift_size
                        sr_curve.spot_rates_data = working
                        krd[tenor_date] = (_reprice() - price_base) / shift_size * 0.0001
                        working.at[tenor_date, col] = orig_val
        finally:
            sr_curve.spot_rates_data = original_data
            if isinstance(self.bond, FloatingRateBond):
                self.bond.pricer._forward_rates = None
                self.bond.pricer._expected_coupons = None

        return krd

    def effective_duration(self, method="worst") -> float:
        """
        Compute the option-adjusted (effective) duration of the callable bond via a
        parallel finite difference, normalised by the callable dirty price.
        Args:
            method (str): 'worst' or 'hw'.
        Returns:
            Effective duration in years.
        """
        dv01 = self.sensitivity(shift_type="parallel", method=method)
        price = self.prices(method)["callableValue"]["dirtyPrice"]
        return float(-dv01 * 10000 / price)


class ZeroCouponBond:
    """
    Bond class for zero coupon bond: single cash flow (face_amount) at maturity_date.
    """
    maturity_date = Date(sterilize_attr=[])
    face_amount = PositiveNumber(sterilize_attr=[])
    currency = String(none_accepted=True, return_if_none="EUR")

    def __init__(self, maturity_date, face_amount, currency=None):
        """
        Args:
            maturity_date (str | pandas.Timestamp): maturity date.
            face_amount (int | float): face amount.
            currency (str): [optional] face amount currency, defaults to EUR
        """
        self.maturity_date = maturity_date
        self.face_amount = face_amount
        self.currency = currency
        self._evaluation_date = None
        self._discount_curve = None
        self._cds_spread = None
        self._recovery_rate = None

    @property
    def evaluation_date(self):
        if self._evaluation_date is None:
            raise ValueError("Evaluation date has not been set. Call 'set_evaluation_date' method to set it.")
        return self._evaluation_date

    @property
    def discount_curve(self):
        if self._discount_curve is None:
            raise ValueError("No discount curve set. Call 'set_discount_curve' method to set it.")
        return self._discount_curve

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

    def __repr__(self):
        return f"ZeroCouponBond(faceAmount={self.face_amount}, maturity={self.maturity_date.strftime('%Y-%m-%d')})"

    def set_evaluation_date(self, date) -> None:
        try:
            self._evaluation_date = pd.to_datetime(date)
        except Exception:
            raise ValueError(f"Can't convert {date} to datetime.")

    def set_discount_curve(self, discount_curve) -> None:
        if isinstance(discount_curve, DiscountCurve):
            self._discount_curve = discount_curve
        else:
            raise ValueError("'discount_curve' must be a DiscountCurve object.")

    def set_cds_spread(self, spread) -> None:
        if not isinstance(spread, float) and spread is not None:
            raise ValueError("Wrong type for parameter 'spread', valid type is float.")
        self._cds_spread = spread

    def set_recovery_rate(self, recovery_rate) -> None:
        if not isinstance(recovery_rate, float) and recovery_rate is not None:
            raise ValueError("Wrong type for 'recovery_rate': it must be a float.")
        self._recovery_rate = recovery_rate

    def prices(self) -> dict:
        df = self.discount_curve.discount_factor_at(pd.DatetimeIndex([self.maturity_date])).item()
        dirty = self.face_amount * df
        prices = {"riskFreeValue": {"dirtyPrice": dirty, "accruedInterest": 0.0, "cleanPrice": dirty}}

        if self._cds_spread:
            if self._recovery_rate is None:
                raise ValueError("CDS spread set but no recovery rate found. Call 'set_recovery_rate'.")
            ttm = accrual_factor("ACT/365", self.evaluation_date, pd.DatetimeIndex([self.maturity_date])).item()
            survival = np.exp(-self.cds_spread * ttm)
            default_pv = self.recovery_rate * (1 - survival) * df * self.face_amount
            risk_adj = self.face_amount * survival * df + default_pv
            prices["riskAdjustedValue"] = {"dirtyPrice": risk_adj, "accruedInterest": 0.0, "cleanPrice": risk_adj}
        return prices

    def sensitivity(self, shift_type="parallel", shift_size=0.01, kind="symmetric") -> float:
        if shift_type not in ("parallel", "slope", "curvature"):
            raise ValueError("Admitted shift types are: 'parallel', 'slope', 'curvature'.")
        if kind not in ("symmetric", "oneside"):
            raise ValueError("Admitted kind types are: 'symmetric', 'oneside'.")

        price_key = "riskAdjustedValue" if self._cds_spread else "riskFreeValue"
        sr_curve = self.discount_curve.rate_curve
        original_data = sr_curve.spot_rates_data.copy()
        n = len(original_data)

        def _node_shift(size):
            match shift_type:
                case "parallel":
                    return np.full(n, size)
                case "slope":
                    return np.linspace(size, -size, n)
                case "curvature":
                    x = np.linspace(0.0, 1.0, n)
                    return 8.0 * size * x * (x - 1.0) + size
            return None

        try:
            match kind:
                case "symmetric":
                    up = original_data.copy()
                    up.iloc[:, 0] += _node_shift(shift_size)
                    sr_curve.spot_rates_data = up
                    price_up = self.prices()[price_key]["dirtyPrice"]

                    down = original_data.copy()
                    down.iloc[:, 0] += _node_shift(-shift_size)
                    sr_curve.spot_rates_data = down
                    price_down = self.prices()[price_key]["dirtyPrice"]

                    return (price_up - price_down) / (2 * shift_size) * 0.0001

                case "oneside":
                    price = self.prices()[price_key]["dirtyPrice"]
                    shifted = original_data.copy()
                    shifted.iloc[:, 0] += _node_shift(shift_size)
                    sr_curve.spot_rates_data = shifted
                    price_shifted = self.prices()[price_key]["dirtyPrice"]

                    return (price_shifted - price) / shift_size * 0.0001
        finally:
            sr_curve.spot_rates_data = original_data
            self.discount_curve.rate_curve = sr_curve

    def key_rate_dv01(self, shift_size=0.0001, kind="symmetric") -> dict:
        if kind not in ("symmetric", "oneside"):
            raise ValueError("Admitted kind types are: 'symmetric', 'oneside'.")

        sr_curve = self.discount_curve.rate_curve
        col = sr_curve.spot_rates_data.columns[0]
        original_data = sr_curve.spot_rates_data.copy()
        maturity = pd.DatetimeIndex([self.maturity_date])
        term = accrual_factor(self.discount_curve.dcc, self.discount_curve.trade_date, maturity)
        compounding = self.discount_curve.compounding

        def _reprice() -> float:
            rate = sr_curve.rate_at(maturity)
            match compounding:
                case "simple":
                    df = 1.0 / (1.0 + rate * term)
                case "annually_compounded":
                    df = 1.0 / (1.0 + rate) ** term
                case "continuous":
                    df = np.exp(-rate * term)
                case _:
                    raise ValueError(f"Unknown compounding: {compounding}")
            return (self.face_amount * df).item()

        krd = {}
        working = original_data.copy()

        try:
            match kind:
                case "symmetric":
                    for tenor_date in original_data.index:
                        orig_val = original_data.at[tenor_date, col]

                        working.at[tenor_date, col] = orig_val + shift_size
                        sr_curve.spot_rates_data = working
                        price_up = _reprice()

                        working.at[tenor_date, col] = orig_val - shift_size
                        sr_curve.spot_rates_data = working
                        price_down = _reprice()

                        working.at[tenor_date, col] = orig_val
                        krd[tenor_date] = (price_up - price_down) / (2 * shift_size) * 0.0001

                case "oneside":
                    sr_curve.spot_rates_data = original_data
                    price_base = _reprice()
                    for tenor_date in original_data.index:
                        orig_val = original_data.at[tenor_date, col]
                        working.at[tenor_date, col] = orig_val + shift_size
                        sr_curve.spot_rates_data = working
                        krd[tenor_date] = (_reprice() - price_base) / shift_size * 0.0001
                        working.at[tenor_date, col] = orig_val
        finally:
            sr_curve.spot_rates_data = original_data
            self.discount_curve.rate_curve = sr_curve

        return krd

    def duration(self) -> float:
        return accrual_factor(self.discount_curve.dcc, self.evaluation_date,
                              pd.DatetimeIndex([self.maturity_date])).item()

    def modified_duration(self, shift_size=0.0001, kind="symmetric") -> float:
        dv01 = self.sensitivity(shift_type="parallel", shift_size=shift_size, kind=kind)
        dirty_price = self.prices()["riskFreeValue"]["dirtyPrice"]
        return float(-dv01 * 10000 / dirty_price)