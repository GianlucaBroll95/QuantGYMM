import pandas as pd
from pandas.tseries.offsets import DateOffset
from ..descriptors import DayCountConvention, PositiveInteger
from ..utils import accrual_factor, business_adjustment, business_days_before

__all__ = ["IborIndex", "Euribor1M", "Euribor3M", "Euribor6M", "Euribor12M"]


class IborIndex:
    """
    Interbank offered rate index
    """

    _DCC = "ACT/360"
    _BUSINESS_CONVENTION = "modified_following"
    _FIXING_DAYS = 2

    tenor_months = PositiveInteger()
    dcc = DayCountConvention()

    def __init__(self, tenor_months, projection_curve=None, fixings=None, dcc=None):
        """
        Args:
            tenor_months (int): tenor of the index in months (3 for Euribor 3M, 6 for Euribor 6M, etc.).
            projection_curve (SpotRateCurve | DiscountCurve | None): curve the future fixings are read off.
            fixings (pandas.Series | None): published fixings, in decimal, indexed by date.
            dcc (str | None): day count convention, defaults to the index's own.
        """
        self.tenor_months = tenor_months
        self.dcc = dcc if dcc is not None else self._DCC
        self._fixings = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        self.projection_curve = projection_curve
        if fixings is not None:
            self.add_fixings(fixings)

    @property
    def name(self):
        if type(self) is IborIndex:
            return f"IborIndex({self.tenor_months}M)"
        return type(self).__name__

    @property
    def projection_curve(self):
        return self._projection_curve

    @projection_curve.setter
    def projection_curve(self, projection_curve):
        if projection_curve is not None and not hasattr(projection_curve, "discount_factor_at"):
            raise ValueError(f"'projection_curve' must expose 'discount_factor_at'."
                             f" Got {projection_curve.__class__.__name__}.")
        self._projection_curve = projection_curve

    @property
    def fixings(self):
        return self._fixings

    def fixing_date(self, start):
        """
        Date the rate for a period starting at 'start' is published on.
        Args:
            start (str | pandas.Timestamp | Iterable): first day of the coupon period.
        Returns:
            dates moved back the index's fixing days on the TARGET calendar.
        """
        return business_days_before(start, self._FIXING_DAYS)

    def add_fixings(self, fixings) -> None:
        """
        Add published fixings to the archive.
        Args:
            fixings (pandas.Series): fixings in decimal, indexed by date.
        """
        if not isinstance(fixings, pd.Series):
            raise ValueError(f"'fixings' must be a pandas.Series. Got {fixings.__class__.__name__}.")
        if not isinstance(fixings.index, pd.DatetimeIndex):
            raise IndexError("'fixings' must have a pandas.DatetimeIndex.")
        merged = pd.concat([self._fixings, fixings.astype(float)])
        self._fixings = merged[~merged.index.duplicated(keep="last")].sort_index()

    def maturity(self, reset) -> pd.Timestamp:
        """
        End of the period a fixing at 'reset' applies to.
        Args:
            reset (str | pandas.Timestamp): first day of the coupon period.
        """
        end = pd.Timestamp(reset) + DateOffset(months=self.tenor_months)
        return pd.Timestamp(business_adjustment(self._BUSINESS_CONVENTION, end))

    def fixing(self, start, evaluation_date, end=None) -> float:
        """
        The rate the index fixes for a coupon period: the published one if the fixing date is past, or is
        today and already published, the forward otherwise.
        Args:
            start (str | pandas.Timestamp): first day of the coupon period.
            evaluation_date (str | pandas.Timestamp): date the valuation is made at.
            end (str | pandas.Timestamp): [optional] end of the period a future fixing is estimated over,
                                          defaults to the index's own maturity.
        Returns:
            float, the rate in decimal.
        """
        fixing_date = self.fixing_date(start)
        evaluation_date = pd.Timestamp(evaluation_date)
        published = fixing_date in self._fixings.index
        if fixing_date < evaluation_date or (fixing_date == evaluation_date and published):
            if not published:
                raise KeyError(f"Missing {self.name} fixing for {fixing_date.date()}.")
            return float(self._fixings.loc[fixing_date])
        return self.forward(start, end)

    def forward(self, reset, end=None) -> float:
        """
        The rate implied by the projection curve for a period starting at 'start'.
        Args:
            reset (str | pandas.Timestamp): first day of the coupon period.
            end (str | pandas.Timestamp): [optional] end of the period, defaults to the index's own maturity.
        Returns:
            float, simple rate on the index's own day count.
        """
        if self._projection_curve is None:
            raise ValueError(f"No projection curve set on {self.name}: future fixings cannot"
                             f" be estimated.")
        start = pd.Timestamp(reset)
        end = self.maturity(reset) if end is None else pd.Timestamp(end)
        df = self._projection_curve.discount_factor_at(pd.DatetimeIndex([start, end]))
        af = accrual_factor(self.dcc, start, end).item()
        return float((df[0] / df[1] - 1) / af)

    def __repr__(self):
        return (f"{type(self).__name__}(tenor_months={self.tenor_months}, dcc='{self.dcc}',"
                f" fixings={len(self._fixings)})")


class Euribor1M(IborIndex):
    def __init__(self, projection_curve=None, fixings=None):
        super().__init__(1, projection_curve=projection_curve, fixings=fixings)


class Euribor3M(IborIndex):
    def __init__(self, projection_curve=None, fixings=None):
        super().__init__(3, projection_curve=projection_curve, fixings=fixings)


class Euribor6M(IborIndex):
    def __init__(self, projection_curve=None, fixings=None):
        super().__init__(6, projection_curve=projection_curve, fixings=fixings)


class Euribor12M(IborIndex):
    def __init__(self, projection_curve=None, fixings=None):
        super().__init__(12, projection_curve=projection_curve, fixings=fixings)
