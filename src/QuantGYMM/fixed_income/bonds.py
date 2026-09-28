import copy

import numpy as np
import pandas as pd

from scipy.optimize import brentq

from .risk_factors import CreditRisk, RateRisk
from ..calendar import Schedule
from ..descriptors import Date, DayCountConvention, FloatNumber, PositiveNumber, String
from ..utils import accrual_factor, business_days_after
from .indexes import IborIndex
from .models import hw_b, hw_bermudan_bond_value, hw_trinomial_tree, hw_zcb_option
from .pricers import BlackCouponPricer, Pricer

__all__ = ["BondPortfolio", "Bond", "FixedRateBond", "FloatingRateBond", "ZeroCouponBond",
           "CallableBond"]


class BondPortfolio(RateRisk, CreditRisk):
    """
    Class for implementing bond portfolio
    """

    def __init__(self, bonds):
        """
        Constructor.
        Args:
            bonds (list[Bond | FloatingRateBond | ZeroCouponBond | FixedRateBond | CallableBond]: list of bonds.
        """
        self.bonds = bonds
        self.projection_curves = list(dict.fromkeys([curve for bond in self.bonds for curve in bond.projection_curves]))
        disc_curves = list(dict.fromkeys([bond.discount_curve for bond in self.bonds]))
        if len(disc_curves) != 1:
            raise ValueError("All bonds must be discounted on the same curve.")
        self.discount_curve = disc_curves[0]

    @property
    def cash_flows(self):
        """
        Return the cash flow of the bond portfolio.
        """
        cash_flow = [bond.cash_flows[["paymentDate", "cashFlow"]].set_index("paymentDate") for bond in self.bonds]
        return pd.concat(cash_flow).groupby(level=0).sum()

    def _dirty_price(self) -> float:
        return sum(bond._dirty_price() for bond in self.bonds)

    def prices(self):
        return pd.DataFrame([bond.prices() for bond in self.bonds]).sum().to_dict()


class Bond(RateRisk, CreditRisk):
    """
    Base class for bond instruments.
    """

    face_amount = PositiveNumber(sterilize_attr=["_cash_flows", "_coupons_history"])
    redemption = FloatNumber(sterilize_attr=["_cash_flows", "_coupons_history"],
                             none_accepted=True, return_if_none=100.0)
    currency = String(none_accepted=True, return_if_none="EUR")
    z_spread = FloatNumber()

    def __init__(self, face_amount, redemption=None, currency=None, z_spread=0.0, identifier=None):
        """
        Args:
            face_amount (int | float): bond face amount
            redemption (float): [optional] redemption price as a percentage of face amount, default is 100.0
            currency (str): [optional] face amount currency, defaults to EUR
            z_spread (flat): credit spread
            identifier (str) [optional] bond identifier
        """
        self.face_amount = face_amount
        self.redemption = redemption
        self.currency = currency
        self.z_spread = z_spread
        self.identifier = identifier
        self._evaluation_date = None
        self._pricer = None
        self._cash_flows = None
        self._coupons_history = None
        self._cds_spread = None
        self._recovery_rate = None
        self._survival_probabilities = None
        self._cache_pillars = None

    @property
    def evaluation_date(self):
        if self._evaluation_date is None:
            raise ValueError("Evaluation date has not been set. Call 'set_evaluation_date' method to set it.")
        return self._evaluation_date

    @property
    def settlement_date(self):
        return business_days_after(self.evaluation_date, 2)

    @property
    def pricer(self):
        if self._pricer is None:
            raise ValueError("No pricer set. Call 'set_pricer' method to set it.")
        return self._pricer

    @property
    def discount_curve(self):
        return self.pricer.discount_curve

    @property
    def cash_flows(self):
        if self._cash_flows is None or self._cache_pillars != self._projection_pillars:
            self._cash_flows = self._get_cash_flows()
            self._cache_pillars = self._projection_pillars
        return self._cash_flows

    @property
    def projection_curves(self):
        return []

    @property
    def _projection_pillars(self):
        return tuple(curve.pillars for curve in self.projection_curves)

    def _dirty_price(self) -> float:
        return self.pricer.present_value(self)

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
    def bonds(self):
        return [self]

    def ytm(self, clean_price=None) -> float:
        settlement = self.settlement_date
        clean = self.prices()["cleanPrice"] if clean_price is None else clean_price / 100 * self.face_amount
        dirty = clean + self.accrued_interest()
        flows = self.cash_flows[self.cash_flows.paymentDate > settlement]
        t = accrual_factor(self.discount_curve.dcc, settlement, flows.paymentDate)
        cash_flows = flows.cashFlow.to_numpy()
        return brentq(lambda y: cash_flows.dot((1 + y) ** -t) - dirty, -0.99, 10, full_output=False)

    def ytm_duration(self, clean_price=None):
        y = self.ytm(clean_price=clean_price)
        settlement = self.settlement_date
        clean = self.prices()["cleanPrice"] if clean_price is None else clean_price / 100 * self.face_amount
        dirty = clean + self.accrued_interest()
        flows = self.cash_flows[self.cash_flows.paymentDate > settlement]
        t = accrual_factor(self.discount_curve.dcc, settlement, flows.paymentDate)
        cash_flows = flows.cashFlow.to_numpy()
        macaulay = (t * cash_flows * (1 + y) ** -t).sum() / dirty
        return macaulay / (1 + y)

    def set_evaluation_date(self, date) -> None:
        """
        Set evaluation date for market price calculation.
        Args:
            date (str | pandas.Timestamp): trade date
        """
        try:
            self._evaluation_date = pd.to_datetime(date)
        except Exception:
            raise ValueError(f"Can't convert {date} to datetime.") from None
        self._cash_flows = None
        self._coupons_history = None
        self._survival_probabilities = None

    def set_pricer(self, pricer) -> None:
        """
        Set the pricer to be used in the market value calculation.
        Args:
            pricer (Pricer): instance of Pricer class
        """
        if not isinstance(pricer, Pricer):
            raise ValueError(f"Pricer must be a Pricer object, got {pricer.__class__.__name__}.")
        self._pricer = pricer

    def set_cds_spread(self, spread) -> None:
        """
        Args:
            spread (float): CDS spread, in decimal.
        """
        if not isinstance(spread, float) and spread is not None:
            raise ValueError("Wrong type for parameter 'spread', valid type is float.")
        self._survival_probabilities = None
        self._cds_spread = spread

    def set_recovery_rate(self, recovery_rate) -> None:
        """
        Args:
            recovery_rate (float | list | numpy.ndarray): either a recovery rate or an array of
                                                          recovery rates (if the RR is assumed to
                                                          be time-varying).
        """
        if not isinstance(recovery_rate, (np.ndarray, list, float)) and recovery_rate is not None:
            raise ValueError("Wrong type for 'recovery_rate': it must be a float or an array.")
        if recovery_rate is not None:
            values = np.asarray(recovery_rate, dtype=float)
            if not np.all((values >= 0) & (values < 1)):
                raise ValueError(f"'recovery_rate' must lie in [0, 1). Got {recovery_rate}.")
        self._survival_probabilities = None
        self._recovery_rate = recovery_rate

    def accrued_interest(self):
        settlement = self.settlement_date
        current = self.cash_flows[self.cash_flows.paymentDate > settlement]
        if current.empty or current.coupon.iloc[0] == 0:
            return 0.0
        period = current.iloc[0]
        end = min(settlement, period.accrualEnd)
        if end <= period.accrualStart:
            return 0.0
        elapsed = accrual_factor(self.dcc, period.accrualStart, end,
                                 reference=(period.referenceStart, period.referenceEnd)).item()
        return period.coupon * elapsed / period.accrualFactor

    def prices(self) -> dict:
        """
        Compute fair market price at settlement as the sum of the cash flows paid after it, discounted to it.
        Returns:
            dict with dirty/clean price and accrued interest.
        """
        dirty = self.pricer.present_value(self, self.settlement_date)
        accrued = self.accrued_interest()
        return {"dirtyPrice": dirty, "accruedInterest": accrued, "cleanPrice": dirty - accrued}

    def _get_cash_flows(self):
        raise NotImplementedError

    def duration(self) -> float:
        """
        Present value of value weighted average time of the cash flows.
        """
        curve = self.discount_curve
        dates = self.cash_flows.paymentDate
        df = self.pricer.discount_factor_at(self, dates)
        t = accrual_factor(curve.dcc, self.evaluation_date, dates)
        cash_flows = self.cash_flows.cashFlow.to_numpy()
        return float((t * cash_flows * df).sum() / cash_flows.dot(df))

    def _get_survival_prob(self):
        """
        Survival probability implied by the CDS spread.
        """
        self._survival_probabilities = self._survival_at(self.cash_flows["paymentDate"])

    def _survival_at(self, dates):
        term = accrual_factor("ACT/365", self.evaluation_date, dates)
        hazard_rate = self.cds_spread / (1 - np.asarray(self.recovery_rate, dtype=float))
        return np.exp(-hazard_rate * term)


class FixedRateBond(Bond):
    """
    Bond class for fixed rate bond.
    """
    dcc = DayCountConvention(sterilize_attr=["_cash_flows", "_coupons_history"])
    coupon_rate = FloatNumber(sterilize_attr=["_cash_flows", "_coupons_history"])

    def __init__(self, schedule, dcc, face_amount, coupon_rate, redemption=None, currency=None, z_spread=0.0,
                 identifier=None):
        """
        Args:
            schedule (Schedule): schedule object for the coupons
            dcc (str): day count convention
            face_amount (int | float): bond face amount
            coupon_rate (float): fixed annual coupon rate (e.g. 0.0375 for 3.75%)
            redemption (float): [optional] redemption price as a percentage of face amount, default is 100.0
            currency (str): [optional] face amount currency, defaults to EUR
            z_spread (float): credit spread
            identifier (str): [optional] identifier for the bond
        """

        super().__init__(face_amount, redemption, currency, z_spread, identifier)
        self.dcc = dcc
        self.coupon_rate = coupon_rate
        self.schedule = schedule

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
    def coupons_history(self):
        if self._coupons_history is None:
            self._coupons_history = self.get_coupons_history()
        return self._coupons_history

    def __repr__(self):
        return f"FixedRateBond(id={self.identifier}, faceAmount={self.face_amount}, couponRate={self.coupon_rate}, " \
               f"maturity={self.schedule.schedule['paymentDate'][-1].strftime(format('%Y-%m-%d'))}, " \
               f"redemption={self.redemption})"

    def _get_cash_flows(self) -> pd.DataFrame:
        """
        Build the coupons outstanding at the evaluation date.
        Returns:
            pandas.DataFrame of coupon start, coupon end, accrual factor, coupon, redemption
            and cash flow.
        """
        schedule = self.schedule.schedule
        future = schedule["paymentDate"] > self.evaluation_date
        periods = {key: dates[future] for key, dates in schedule.items()}
        af = accrual_factor(self.dcc, periods["startingDate"], periods["endingDate"],
                            reference=(periods["referenceStart"], periods["referenceEnd"]))
        coupon = self.coupon_rate * af * self.face_amount
        if len(coupon) == 0:
            return pd.DataFrame(
                columns=["accrualStart", "accrualEnd", "paymentDate", "referenceStart", "referenceEnd",
                         "accrualFactor", "coupon", "redemption", "cashFlow"]
            )

        redemption = np.zeros(len(coupon))
        redemption[-1] = self.redemption / 100 * self.face_amount
        return pd.DataFrame(
            {"accrualStart": periods["startingDate"], "accrualEnd": periods["endingDate"],
             "paymentDate": periods["paymentDate"], "referenceStart": periods["referenceStart"],
             "referenceEnd": periods["referenceEnd"], "accrualFactor": af, "coupon": coupon,
             "redemption": redemption, "cashFlow": coupon + redemption},
            index=pd.RangeIndex(1, len(coupon) + 1, name="couponNumber")
        )

    def get_coupons_history(self) -> pd.DataFrame:
        """
        Calculate the past history of coupons (deterministic for a fixed rate bond).
        Returns:
            pandas.DataFrame of coupon start, coupon end, accrual factor and cash flow.
        """
        schedule = self.schedule.schedule
        past = schedule["paymentDate"] <= self.evaluation_date
        periods = {key: dates[past] for key, dates in schedule.items()}
        af = accrual_factor(self.dcc, periods["startingDate"], periods["endingDate"],
                            reference=(periods["referenceStart"], periods["referenceEnd"]))
        coupon = self.coupon_rate * af * self.face_amount

        return pd.DataFrame(
            {"accrualStart": periods["startingDate"], "accrualEnd": periods["endingDate"],
             "paymentDate": periods["paymentDate"], "accrualFactor": af, "coupon": coupon},
            index=pd.RangeIndex(1, len(coupon) + 1, name="couponNumber")
        )


class ZeroCouponBond(Bond):
    """
    Bond class for zero coupon bond: single cash flow (face_amount) at maturity_date.
    """
    maturity_date = Date(sterilize_attr=["_cash_flows"])

    def __init__(self, maturity_date, face_amount, redemption=None, currency=None, z_spread=0.0, identifier=None):
        """
        Args:
            maturity_date (str | pandas.Timestamp): maturity date.
            face_amount (int | float): face amount.
            redemption (float): [optional] redemption price as a percentage of face amount, default is 100.0
            currency (str): [optional] face amount currency, defaults to EUR
            z_spread (float): credit spread
        """
        super().__init__(face_amount, redemption, currency, z_spread, identifier)
        self.maturity_date = maturity_date

    def __repr__(self):
        return (f"ZeroCouponBond(id={self.identifier}, faceAmount={self.face_amount}, "
                f"maturity={self.maturity_date.strftime('%Y-%m-%d')})")

    def _get_cash_flows(self) -> pd.DataFrame:
        """
        Build the single cash flow the bond pays at maturity.
        Returns:
            pandas.DataFrame with the same columns every other bond exposes.
        """
        maturity = pd.Timestamp(self.maturity_date)
        redemption = self.redemption / 100 * self.face_amount
        return pd.DataFrame(
            {"accrualStart": [self.evaluation_date], "accrualEnd": [maturity], "paymentDate": [maturity],
             "accrualFactor": accrual_factor("ACT/365", self.evaluation_date, maturity),
             "coupon": [0.0], "redemption": [redemption], "cashFlow": [redemption]},
            index=pd.RangeIndex(1, 2, name="couponNumber"))


class FloatingRateBond(Bond):
    """
    Bond class for floating rate bond.
    """
    dcc = DayCountConvention(sterilize_attr=["_coupons_history", "_cash_flows"])
    spread = FloatNumber(sterilize_attr=["_coupons_history", "_cash_flows"], none_accepted=True, return_if_none=0.0)
    cap = FloatNumber(sterilize_attr=["_coupons_history", "_cash_flows"], none_accepted=True, return_if_none=np.nan)
    floor = FloatNumber(sterilize_attr=["_coupons_history", "_cash_flows"], none_accepted=True, return_if_none=np.nan)

    def __init__(self, schedule, index, dcc, face_amount, spread=0.0, floor=None, cap=None,
                 redemption=None, currency=None, z_spread=0.0, identifier=None):
        """
        Args:
            schedule (Schedule): schedule object for the coupons
            index (IborIndex): index of the coupons
            dcc (str): day count convention
            face_amount (int | float): bond face amount
            spread (float): [optional] spread over the floating rate (contractual)
            floor (float): [optional] floor rate for the coupon
            cap (float): [optional] cap rate for the coupon
            redemption (float): [optional] redemption price as a percentage of face amount, default is 100.0
            currency (str): [optional] face amount currency, defaults to EUR
            z_spread (float): credit spread
            identifier (str): identifier for the bond
        """
        super().__init__(face_amount, redemption, currency, z_spread, identifier)
        self.dcc = dcc
        self.spread = spread
        self.floor = floor
        self.cap = cap
        self.schedule = schedule
        self.index = index
        self._coupon_pricer = None

    @property
    def schedule(self):
        return self._schedule

    @schedule.setter
    def schedule(self, schedule):
        if isinstance(schedule, Schedule):
            bond_schedule = copy.deepcopy(schedule)
            self._schedule = bond_schedule
            self._cash_flows = None
            self._coupons_history = None
        else:
            raise ValueError(f"'{schedule}' is not a Schedule object.")

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        if not isinstance(index, IborIndex):
            raise ValueError(f"'index' must be an IborIndex object. Got {index.__class__.__name__}.")
        self._index = index
        self._cash_flows = None
        self._coupons_history = None

    @property
    def coupon_pricer(self):
        return self._coupon_pricer

    @property
    def projection_curves(self):
        curve = self.index.projection_curve
        return [] if curve is None else [curve]

    def __repr__(self):
        return f"Bond(id={self.identifier}, faceAmount={self.face_amount}, spread={self.spread}, " \
               f"maturity={self.schedule.schedule['paymentDate'][-1].strftime(format('%Y-%m-%d'))}," \
               f" floor={self.floor}, cap={self.cap})"

    def set_coupon_pricer(self, coupon_pricer) -> None:
        """
        Set the model the caplets and floorlets are priced with.
        Args:
            coupon_pricer (BlackCouponPricer): instance of a coupon pricer class
        """
        if not isinstance(coupon_pricer, BlackCouponPricer):
            raise ValueError(f"'coupon_pricer' must be a coupon pricer object."
                             f" Got {coupon_pricer.__class__.__name__}.")
        self._coupon_pricer = coupon_pricer
        self._cash_flows = None

    def _get_cash_flows(self) -> pd.DataFrame:
        """
        Build the coupons outstanding at the evaluation date.
        """
        schedule = self.schedule.schedule
        future = schedule["paymentDate"] > self.evaluation_date
        periods = {key: dates[future] for key, dates in schedule.items()}
        starts, payments = periods["startingDate"], periods["paymentDate"]
        resets = self.index.fixing_date(starts)
        af = accrual_factor(self.dcc, starts, periods["endingDate"],
                            reference=(periods["referenceStart"], periods["referenceEnd"]))
        reset_rate = np.array([self.index.fixing(start, self.evaluation_date) for start in starts])
        coupon_rate = reset_rate + self.spread
        coupon = coupon_rate * af * self.face_amount
        redemption = np.zeros(len(payments))
        redemption[-1] = self.redemption / 100 * self.face_amount
        numbering = pd.RangeIndex(1, len(payments) + 1, name="couponNumber")
        columns = {"resetDate": resets, "accrualStart": starts, "accrualEnd": periods["endingDate"],
                   "paymentDate": payments, "referenceStart": periods["referenceStart"],
                   "referenceEnd": periods["referenceEnd"], "accrualFactor": af, "resetRate": reset_rate,
                   "spread": self.spread, "couponRate": coupon_rate}

        if np.isnan(self.cap) and np.isnan(self.floor):
            return pd.DataFrame({**columns, "coupon": coupon, "redemption": redemption,
                                 "cashFlow": coupon + redemption}, index=numbering)

        if self._coupon_pricer is None:
            raise ValueError("Bond has a cap or a floor but no coupon pricer. Call"
                             " 'set_coupon_pricer' to set the model they are priced with.")

        fixed = resets <= self.evaluation_date
        caplet, floorlet = np.empty(len(payments)), np.empty(len(payments))
        caplet[fixed] = np.maximum(coupon_rate[fixed] - self.cap, 0) * af[fixed] * self.face_amount
        floorlet[fixed] = np.maximum(self.floor - coupon_rate[fixed], 0) * af[fixed] * self.face_amount
        caplet[~fixed], floorlet[~fixed] = self._coupon_pricer.forward_premiums(
            self, resets[~fixed], af[~fixed], reset_rate[~fixed])

        optioned = np.nansum([coupon, floorlet, -caplet], axis=0)
        return pd.DataFrame(
            {**columns, "floorlet": floorlet, "caplet": -caplet, "coupon": optioned,
             "redemption": redemption, "cashFlow": optioned + redemption},
            index=numbering).replace(np.nan, "-")

    def get_coupons_history(self) -> pd.DataFrame:
        """
        Calculate the past history of coupons.
        Returns:
            pandas.DataFrame of coupons reset date, coupon staring date, coupon payment date, coupon accrual factor,
            coupon rate.
        """
        schedule = self.schedule.schedule
        af = accrual_factor(self.dcc, schedule["startingDate"], schedule["endingDate"],
                            reference=(schedule["referenceStart"], schedule["referenceEnd"]))
        past_date_mask = schedule["paymentDate"] <= self.evaluation_date
        hist_reset = self.index.fixing_date(schedule["startingDate"])[past_date_mask]

        if len(hist_reset) == 0:
            return pd.DataFrame(
                columns=["resetDate", "accrualStart", "accrualEnd", "paymentDate", "accrualFactor",
                         "resetRate", "spread", "couponRate", "floorlet", "caplet", "coupon"],
                index=pd.RangeIndex(0, name="couponNumber"))

        hist_starting = schedule["startingDate"][past_date_mask]
        hist_ending = schedule["endingDate"][past_date_mask]
        hist_payment = schedule["paymentDate"][past_date_mask]
        hist_rate = np.array([self.index.fixing(start, self.evaluation_date) for start in hist_starting]) + self.spread
        hist_accrual = af[past_date_mask]
        floorlet = np.maximum(self.floor - hist_rate, 0) * hist_accrual * self.face_amount
        caplet = np.maximum(hist_rate - self.cap, 0) * hist_accrual * self.face_amount
        return pd.DataFrame(
            {"resetDate": hist_reset, "accrualStart": hist_starting, "accrualEnd": hist_ending,
             "paymentDate": hist_payment,
             "accrualFactor": hist_accrual, "resetRate": hist_rate - self.spread, "spread": self.spread,
             "couponRate": hist_rate, "floorlet": floorlet, "caplet": -caplet,
             "coupon": np.nansum([hist_rate * hist_accrual * self.face_amount, floorlet, -caplet],
                                 axis=0)}, index=pd.RangeIndex(1, len(hist_rate) + 1, name="couponNumber")
        ).replace(np.nan, "-")

    def ytm_duration(self, clean_price=None) -> float:
        raise NotImplementedError("Yield to maturity duration is not a valid metric for a floating rate bond.")


class CallableBond(RateRisk, CreditRisk):
    """
    Callable bond wrapper.
    """
    _METHODS = ("worst", "hw", "tree")

    def __init__(self, bond, call_schedule, mean_reversion=0.03, volatility=0.01, method="tree"):
        """
        Args:
            bond (FixedRateBond | FloatingRateBond): the underlying non-callable bond.
            call_schedule (pandas.Series): index = call dates (must coincide with the
                                           bond's coupon payment dates), values = clean
                                           call price as % of face (e.g. 100.0 = at par).
            mean_reversion (float): Hull-White mean reversion speed 'a' (positive).
            volatility (float): Hull-White short rate volatility 'sigma' (positive).
            method (str): 'tree', 'hw', 'tree', method to price the embedded option.
        """
        if method not in self._METHODS:
            raise ValueError("'method' must be one of 'worst', 'hw', 'tree'.")
        self.method = method
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

    def __repr__(self):
        calls = self.call_schedule.index
        return f"CallableBond(id={self.identifier}, {self.bond!r}, calls={len(calls)}, " \
               f"firstCall={calls[0].strftime('%Y-%m-%d')}, lastCall={calls[-1].strftime('%Y-%m-%d')})"

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

    @property
    def discount_curve(self):
        return self.bond.discount_curve

    @property
    def projection_curves(self):
        return self.bond.projection_curves

    @property
    def cash_flows(self):
        return self.bond.cash_flows

    @property
    def bonds(self):
        return [self.bond]

    @property
    def identifier(self):
        return self.bond.identifier

    @property
    def z_spread(self):
        return self.bond.z_spread

    @z_spread.setter
    def z_spread(self, value):
        self.bond.z_spread = value

    def _dirty_price(self):
        return float(self.bond._dirty_price() - self._option_value())

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

    def _future_cash_flows(self):
        """
        Normalize both underlying bond types to one shape: coupon-only cash flows
        (redemption/face amount excluded) and their past payment dates.
        Returns:
            (pandas.DatetimeIndex, numpy.ndarray) - dates, coupon amounts
        """

        cash_flows = self.bond.cash_flows
        return pd.DatetimeIndex(cash_flows.paymentDate), cash_flows.coupon.to_numpy().astype(float)

    def prices(self) -> dict:
        """
        Compute straight, option and callable value of the bond at settlement.
        """
        settlement = [self.bond.settlement_date]
        option_value = self._option_value() / self.bond.pricer.discount_factor_at(self.bond, settlement)[0]
        straight = self.bond.prices()
        callable_dirty = straight["dirtyPrice"] - option_value

        return {
            "dirtyPrice": callable_dirty,
            "accruedInterest": straight["accruedInterest"],
            "cleanPrice": callable_dirty - straight["accruedInterest"],
            "optionValue": option_value,
        }

    def _option_value(self) -> float:
        if self.bond._cds_spread:
            raise NotImplementedError("CallableBond does not support CDS-adjusted valuation yet.")
        if self.method == "worst":
            return self._option_value_worst(self.bond._dirty_price())
        if self.method == "hw":
            return self._option_value_hw()
        return self._option_value_tree()

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

        dc = self.discount_curve
        dates, coupons = self._future_cash_flows()
        redemption = self.bond.cash_flows.redemption.iloc[-1]
        horizon_days = (dates[-1] - dc.trade_date).days
        step_days = max(1, int(step_days))
        n_steps = int(np.ceil(horizon_days / step_days))
        dt = accrual_factor(dc.dcc, dc.trade_date, dc.trade_date + pd.Timedelta(days=step_days)).item()

        grid_dates = pd.DatetimeIndex(
            [dc.trade_date + pd.Timedelta(days=i * step_days) for i in range(n_steps + 1)]
        )
        discount_factors = np.empty(n_steps + 1)
        discount_factors[0] = 1.0
        discount_factors[1:] = self.bond.pricer.discount_factor_at(self.bond, grid_dates[1:])

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
        dates, coupons = self._future_cash_flows()

        pvs = [straight_dirty]

        for call_date, clean_price in self.call_schedule.items():
            mask = dates <= call_date
            called_dates = dates[mask].append(pd.DatetimeIndex([call_date]))
            called_cfs = np.append(coupons[mask], clean_price / 100 * self.face_amount)
            df = self.bond.pricer.discount_factor_at(self.bond, called_dates)
            pvs.append(called_cfs.dot(df))

        worst = min(pvs)
        return straight_dirty - worst

    def _option_value_hw(self) -> float:
        if isinstance(self.bond, FloatingRateBond):
            raise NotImplementedError("Hull-White valuation is only available for fixed rate underlyings.")

        call_sched = self.call_schedule
        if len(call_sched) > 1:
            raise NotImplementedError(
                "The Jamshidian decomposition prices a single exercise date only. "
                "For a Bermudan schedule use method='tree'."
            )

        call_date = call_sched.index[0]
        K = call_sched.iloc[0] / 100.0

        dc = self.discount_curve
        a = self.mean_reversion
        sigma = self.volatility

        # t_call = option expiry (call date), t_cf = each cash flow date after the
        # call — named to match hw_b(t, T, a)'s (valuation, target) argument order.
        t_call = accrual_factor(dc.dcc, dc.trade_date, call_date).item()

        cf = self.bond.cash_flows
        mask = pd.DatetimeIndex(cf.paymentDate) > call_date
        c = cf.cashFlow.to_numpy()[mask] / self.face_amount
        t_dates = pd.DatetimeIndex(cf.paymentDate)[mask]
        t_cf = accrual_factor(dc.dcc, dc.trade_date, t_dates)

        df_T = self.bond.pricer.discount_factor_at(self.bond, pd.DatetimeIndex([call_date])).item()
        df_t = self.bond.pricer.discount_factor_at(self.bond, t_dates)

        eps = 1.0 / 365.0
        df_up = self.bond.pricer.discount_factor_at(self.bond,
                                                    pd.DatetimeIndex([call_date + pd.Timedelta(days=1)])).item()
        df_down = self.bond.pricer.discount_factor_at(self.bond,
                                                      pd.DatetimeIndex([call_date - pd.Timedelta(days=1)])).item()
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

    def option_value(self) -> float:
        """
        Returns:
            The call option value (per the wrapped bond's face amount).
        """
        return self.prices()["optionValue"]

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
            raise NotImplementedError("Bullet-to-call is only implemented for fixed rate underlying.")

        call_date = self.call_schedule.index[call_index]
        clean_price = self.call_schedule.iloc[call_index]

        truncated = copy.deepcopy(self.bond)
        sched = truncated.schedule.schedule
        mask = sched["paymentDate"] <= call_date
        truncated.schedule._schedule = {k: v[mask] for k, v in sched.items()}
        truncated._cash_flows = None
        truncated.redemption = float(clean_price)
        truncated.set_evaluation_date(self.evaluation_date)
        truncated.set_pricer(self.bond.pricer)
        return truncated.prices()
