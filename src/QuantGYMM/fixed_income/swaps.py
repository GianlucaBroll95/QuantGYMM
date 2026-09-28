import numpy as np
import pandas as pd

from .indexes import IborIndex
from .pricers import Pricer
from .risk_factors import RateRisk
from ..calendar import Schedule
from ..descriptors import DayCountConvention, FloatNumber, PositiveNumber
from ..utils import accrual_factor

__all__ = ["VanillaSwap"]


class VanillaSwap(RateRisk):
    """
    Fixed rate exchanged against an Ibor index plus a spread.
    """
    _SIDES = ("payer", "receiver")

    fixed_rate = FloatNumber()
    spread = FloatNumber()
    nominal = PositiveNumber()
    fixed_dcc = DayCountConvention()
    float_dcc = DayCountConvention()

    def __init__(self, fixed_schedule, fixed_rate, float_schedule, index, nominal, fixed_dcc="30/360",
                 float_dcc=None, spread=0.0, side="payer"):
        """
        Args:
            fixed_schedule (Schedule): accrual and payment dates of the fixed leg.
            fixed_rate (float): fixed rate, in decimal.
            float_schedule (Schedule): accrual and payment dates of the floating leg.
            index (IborIndex): index the floating leg pays.
            nominal (float): notional of both legs.
            fixed_dcc (str): day count convention of the fixed leg.
            float_dcc (str): [optional] day count convention of the floating leg, defaults to the index's.
            spread (float): spread over the index, in decimal.
            side (str): 'payer' pays the fixed rate, 'receiver' receives it.
        """
        if not isinstance(fixed_schedule, Schedule) or not isinstance(float_schedule, Schedule):
            raise ValueError("'fixed_schedule' and 'float_schedule' must be Schedule objects.")
        if not isinstance(index, IborIndex):
            raise ValueError(f"'index' must be an IborIndex object. Got {index.__class__.__name__}.")
        if side not in self._SIDES:
            raise ValueError("'side' must be 'payer' or 'receiver'.")
        self.fixed_schedule = fixed_schedule
        self.float_schedule = float_schedule
        self.fixed_rate = fixed_rate
        self.index = index
        self.nominal = nominal
        self.fixed_dcc = fixed_dcc
        self.float_dcc = index.dcc if float_dcc is None else float_dcc
        self.spread = spread
        self.side = side
        self._evaluation_date = None
        self._pricer = None

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
    def discount_curve(self):
        return self.pricer.discount_curve

    @property
    def projection_curves(self):
        curve = self.index.projection_curve
        return [] if curve is None else [curve]

    def set_evaluation_date(self, date) -> None:
        """
        Args:
            date (str | pandas.Timestamp): valuation date.
        """
        self._evaluation_date = pd.Timestamp(date)

    def set_pricer(self, pricer) -> None:
        """
        Args:
            pricer (Pricer): pricer holding the curve both legs are discounted on.
        """
        if not isinstance(pricer, Pricer):
            raise ValueError(f"Pricer must be a Pricer object, got {pricer.__class__.__name__}.")
        self._pricer = pricer

    def _periods(self, schedule, dcc):
        dates = schedule.schedule
        future = dates["paymentDate"] > self.evaluation_date
        periods = {key: value[future] for key, value in dates.items()}
        af = accrual_factor(dcc, periods["startingDate"], periods["endingDate"],
                            reference=(periods["referenceStart"], periods["referenceEnd"]))
        return periods, af

    @property
    def fixed_leg(self) -> pd.DataFrame:
        """
        Coupons of the fixed leg still to be paid.
        """
        periods, af = self._periods(self.fixed_schedule, self.fixed_dcc)
        return pd.DataFrame({"accrualStart": periods["startingDate"], "accrualEnd": periods["endingDate"],
                             "paymentDate": periods["paymentDate"], "accrualFactor": af, "rate": self.fixed_rate,
                             "amount": self.fixed_rate * af * self.nominal})

    @property
    def floating_leg(self) -> pd.DataFrame:
        """
        Coupons of the floating leg still to be paid: fixed ones at the published rate, the others at the
        forward over their accrual period.
        """
        periods, af = self._periods(self.float_schedule, self.float_dcc)
        starts, ends = periods["startingDate"], periods["endingDate"]
        rate = np.array([self.index.fixing(start, self.evaluation_date, end) for start, end in zip(starts, ends)])
        return pd.DataFrame({"resetDate": self.index.fixing_date(starts), "accrualStart": starts,
                             "accrualEnd": ends, "paymentDate": periods["paymentDate"], "accrualFactor": af,
                             "resetRate": rate, "spread": self.spread,
                             "amount": (rate + self.spread) * af * self.nominal})

    def _legs_value(self):
        if self.evaluation_date != self.discount_curve.trade_date:
            raise ValueError(f"The swap is evaluated on {self.evaluation_date.date()}, the discount curve is "
                             f"built on {self.discount_curve.trade_date.date()}.")
        fixed, floating = self.fixed_leg, self.floating_leg
        discount = self.discount_curve.discount_factor_at
        fixed_value = float(fixed.amount.to_numpy() @ discount(fixed.paymentDate))
        floating_value = float(floating.amount.to_numpy() @ discount(floating.paymentDate))
        annuity = float(fixed.accrualFactor.to_numpy() @ discount(fixed.paymentDate)) * self.nominal
        return fixed_value, floating_value, annuity

    def _dirty_price(self) -> float:
        fixed_value, floating_value, _ = self._legs_value()
        value = floating_value - fixed_value
        return value if self.side == "payer" else -value

    def npv(self) -> float:
        """
        Value at the evaluation date: floating leg less fixed leg for a payer, the opposite for a receiver.
        """
        return self._dirty_price()

    def par_rate(self) -> float:
        """
        Fixed rate that makes the swap worth zero.
        """
        _, floating_value, annuity = self._legs_value()
        return floating_value / annuity

    def __repr__(self):
        return (f"VanillaSwap({self.side}, nominal={self.nominal}, fixedRate={self.fixed_rate}, "
                f"index={self.index.name}, maturity={self.fixed_schedule.end_date.strftime('%Y-%m-%d')})")
