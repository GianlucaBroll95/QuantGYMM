import bisect
from contextlib import contextmanager

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.linalg import solve_triangular
from scipy.optimize import brentq

from ..calendar import Schedule
from ..descriptors import BusinessConvention, Date, DayCountConvention
from ..utils import accrual_factor, business_adjustment, business_days_after, imm_date, tenor_offset

__all__ = ["YieldCurve", "ForwardRate", "Deposit", "ForwardRateAgreement", "Future", "Swap"]


class YieldCurve:
    """
    Term structure of discount factors implied by a set of quoted instruments.
    """
    _JACOBIAN_STEP = 1e-7
    trade_date = Date(sterilize_attr=["_interpolator"])
    dcc = DayCountConvention(sterilize_attr=["_interpolator"])

    def __init__(self, instruments: list, name, trade_date: pd.Timestamp, dcc="ACT/365", interpolation="linear"):
        """
        Args:
            instruments (list): quoted instruments the curve is solved against; each one must expose
                'maturity', 'quote' and 'implied_quote'.
            trade_date (str | pandas.Timestamp): date the discount factor is one at.
            dcc (str): day count convention turning dates into year fractions.
            interpolation (str): interpolation method over the pillars.
        """

        self.trade_date = trade_date
        self.instruments = sorted(instruments, key=lambda x: x.maturity)
        self.dcc = dcc
        self.interpolation = interpolation
        self.name = name
        self._interpolator = None
        self._jacobian = None
        self._jacobian_key = None
        self._times = [0.0]
        self._log_df = [0.0]
        self._frozen = False
        self._source_pillars = []
        self.sources = []
        for instrument in self.instruments:
            source = getattr(instrument, "discount", None)
            if source is not None and source not in self.sources:
                self.sources.append(source)
        self._bootstrap()

    def __repr__(self):
        return (f"YieldCurve({self.name}, trade_date = {self.trade_date.strftime('%Y-%m-%d')}, "
                f"pillars = {len(self._times) - 1})")

    @property
    def pillars(self):
        """
        Dates and log discount factors the curve is made of. This property is needed to ensure proper version control
        handling during sensitivity analysis.
        """
        self._ensure_current()
        return tuple(self._times), tuple(self._log_df)

    @property
    def interpolator(self):
        """
        Interpolator over the pillars, built on first read and discarded whenever a pillar moves.
        """
        if self._interpolator is None:
            self._interpolator = interp1d(self._times, self._log_df, kind=self.interpolation, bounds_error=False,
                                          fill_value="extrapolate")
        return self._interpolator

    @property
    def spot_rates(self):
        dates = [instrument.maturity for instrument in self.instruments]
        return pd.DataFrame({"date": dates, "spotRate": -np.asarray(self._log_df[1:]) / np.asarray(self._times[1:])})

    @property
    def discount_factors(self):
        dates = [instrument.maturity for instrument in self.instruments]
        return pd.DataFrame(
            {"date": dates, "discountFactor": np.exp(self._log_df[1:])})

    @property
    def jacobian(self):
        self._ensure_current()
        key = self.pillars
        if self._jacobian_key != key:
            self._jacobian = self._build_jacobian()
            self._jacobian_key = key
        return self._jacobian

    def _build_jacobian(self):
        J = np.zeros((len(self.instruments), len(self._log_df[1:])))
        base = np.array([instrument.implied_quote(self) for instrument in self.instruments])
        for i, time in enumerate(self._times[1:], start=1):
            self._log_df[i] -= self._JACOBIAN_STEP * time
            self._interpolator = None
            J[:, i - 1] = ([instrument.implied_quote(self) for instrument in
                            self.instruments] - base) / self._JACOBIAN_STEP
            self._log_df[i] += self._JACOBIAN_STEP * time
            self._interpolator = None
        return J

    def _bootstrap(self):
        """
        Solve the pillars one at a time, in maturity order.

        Every instrument contributes a single pillar at its own maturity, so each step leaves one
        unknown and is closed by a scalar root search on the log discount factor.
        """
        self._source_pillars = [source.pillars for source in self.sources]
        self._times = [0.0]
        self._log_df = [0.0]
        self._interpolator = None
        for instrument in self.instruments:
            self._times.append(accrual_factor(self.dcc, self.trade_date, instrument.maturity).item())
            self._log_df.append(self._log_df[-1])

            def residual(x, instrument=instrument):
                self._log_df[-1] = x
                self._interpolator = None
                return instrument.implied_quote(self) - instrument.quote

            self._log_df[-1] = brentq(residual, -5, 5, xtol=1e-15)
            self._interpolator = None

    def _ensure_current(self):
        if self._frozen or not self.sources:
            return
        if [source.pillars for source in self.sources] != self._source_pillars:
            self._bootstrap()

    def discount_factor_at(self, date):
        """
        Present value of one unit paid at each date.
        Args:
            date (str | pandas.Timestamp | Iterable): dates to discount.
        Returns:
            float for a single date, numpy.ndarray otherwise.
        """
        self._ensure_current()
        af = accrual_factor(self.dcc, self.trade_date, date)
        df = np.exp(self.interpolator(af))
        return df.item() if np.ndim(date) == 0 else df

    def zero_rates_at(self, date):
        """
        Continuously compounded rate from the trade date to a date.
        Args:
            date (str | pandas.Timestamp): date to measure to.
        Returns:
            float
        """
        self._ensure_current()
        af = accrual_factor(self.dcc, self.trade_date, date).item()
        return -float(self.interpolator(af)) / af

    @contextmanager
    def frozen(self):
        was_frozen = self._frozen
        self._frozen = True
        try:
            yield self
        finally:
            self._frozen = was_frozen

    @contextmanager
    def shocked_quotes(self, size: float | np.ndarray, node: int | None = None):
        """
        Move one market quote or the whole curve, and solve again.
        Args:
            size (float | np.ndarray): amount added to market quote
            node (int | None): [optional] market quote node
        """
        targets = self.instruments if node is None else [self.instruments[node]]
        sizes = np.broadcast_to(size, len(targets))
        saved_quotes = [instrument.quote for instrument in targets]
        saved_times, saved_log_df, saved_source_pillars = list(self._times), list(self._log_df), list(
            self._source_pillars)

        for instrument, amount in zip(targets, sizes):
            instrument.quote += float(amount)
        try:
            self._bootstrap()
            yield self
        finally:
            for instrument, quote in zip(targets, saved_quotes):
                instrument.quote = quote
            self._times, self._log_df, self._source_pillars = saved_times, saved_log_df, saved_source_pillars
            self._interpolator = None

    @contextmanager
    def shocked_zeros(self, size: float | np.ndarray, node: int | str | None = None):
        """
        Move one zero rate or the whole curve.
        Args:
            size (float): amount added to zero
            node (int | None): [optional] zero node to shock
        """
        saved_times, saved_log = list(self._times), list(self._log_df)
        if node is not None and not isinstance(node, int):
            t = accrual_factor(self.dcc, self.trade_date,
                               self.trade_date + tenor_offset(node) if isinstance(node, str) else node).item()
            k = bisect.bisect_left(self._times, t)
            if k == len(self._times) or self._times[k] != t:
                self._log_df.insert(k, float(self.interpolator(t)))
                self._times.insert(k, t)
            nodes = [k]
        else:
            nodes = range(1, len(self._times)) if node is None else [node + 1]
        sizes = np.broadcast_to(size, len(nodes))
        try:
            with self.frozen():
                for i, amount in zip(nodes, sizes):
                    self._log_df[i] -= float(amount) * self._times[i]
                self._interpolator = None
                yield self
        finally:
            self._times, self._log_df = saved_times, saved_log
            self._interpolator = None

    def dependency_jacobian(self, source):
        """
        Derivative of the own zero rates with respect to the zero rates of 'source', own market quotes held:
        -J^-1 . dS/dz_source (Henrard 2013, par. 3.6, implicit function theorem).
        Args:
            source (YieldCurve): curve this one is calibrated on.
        Returns:
            numpy.ndarray of shape (own nodes, source nodes).
        """
        self._ensure_current()
        base = np.array([instrument.implied_quote(self) for instrument in self.instruments])
        D = np.zeros((len(self.instruments), len(source.instruments)))
        with self.frozen():
            for j in range(len(source.instruments)):
                with source.shocked_zeros(self._JACOBIAN_STEP, node=j):
                    D[:, j] = (np.array([instrument.implied_quote(self) for instrument in
                                         self.instruments]) - base) / self._JACOBIAN_STEP
        return solve_triangular(self.jacobian, -D, lower=True)


class ForwardRate:
    """
    Base class for instruments quoting a simple rate over a single period.
    """
    _SPOT_LAG = 2
    trade_date = Date()
    dcc = DayCountConvention()
    convention = BusinessConvention()
    starting_date: pd.Timestamp
    maturity: pd.Timestamp
    accrual: float

    def __init__(self, quote, trade_date, dcc="ACT/360", convention="modified_following"):
        """
        Args:
            quote (float): rate quoted by the market.
            trade_date (str | pandas.Timestamp): trade date.
            dcc (str): day count convention of the period.
            convention (str): business day convention.
        """
        self.quote = quote
        self.trade_date = trade_date
        self.dcc = dcc
        self.convention = convention
        self.spot_date = business_days_after(self.trade_date, self._SPOT_LAG)

    def implied_quote(self, curve: YieldCurve):
        """
        Simple rate the curve implies over the period the instrument covers.
        Args:
            curve (YieldCurve): curve the discount factors are read from.
        Returns:
            float
        """
        p_start = curve.discount_factor_at(self.starting_date)
        p_end = curve.discount_factor_at(self.maturity)
        return (p_start / p_end - 1) / self.accrual


class Deposit(ForwardRate):
    """
    Cash lent at spot and repaid with interest at a single future date.
    """

    def __init__(self, tenor, quote, trade_date, dcc="ACT/360", convention="modified_following"):
        """
        Args:
            tenor (str): length of the deposit, measured from spot.
            quote (float): rate quoted by the market.
            trade_date (str | pandas.Timestamp): trade date.
            dcc (str): day count convention of the period.
            convention (str): business day convention.
        """
        super().__init__(quote, trade_date, dcc, convention)
        self.tenor = tenor
        self.maturity = business_adjustment(self.convention, self.spot_date + tenor_offset(self.tenor))
        self.accrual = accrual_factor(self.dcc, self.spot_date, self.maturity).item()
        self.starting_date = self.spot_date

    def __repr__(self):
        return f"Deposit(tenor = {self.tenor}, quote = {self.quote:.4%})"


class ForwardRateAgreement(ForwardRate):
    """
    Rate fixed today for a period that starts in the future.
    """

    def __init__(self, start_tenor, end_tenor, quote, trade_date, dcc="ACT/360", convention="modified_following"):
        """
        Args:
            start_tenor (str): time from spot to the start of the period.
            end_tenor (str): time from spot to the end of the period.
            quote (float): rate quoted by the market.
            trade_date (str | pandas.Timestamp): trade date.
            dcc (str): day count convention of the period.
            convention (str): business day convention.
        """
        super().__init__(quote, trade_date, dcc, convention)
        self.start_tenor = start_tenor
        self.end_tenor = end_tenor
        self.starting_date = business_adjustment(self.convention, self.spot_date + tenor_offset(self.start_tenor))
        self.maturity = business_adjustment(self.convention, self.spot_date + tenor_offset(self.end_tenor))
        self.accrual = accrual_factor(self.dcc, self.starting_date, self.maturity).item()

    def __repr__(self):
        return f"ForwardRateAgreement({self.start_tenor}x{self.end_tenor}, quote = {self.quote:.4%})"


class Future(ForwardRate):
    """
    Exchange traded contract on a future rate, dated on the IMM calendar.
    """

    def __init__(self, delivery, quote, trade_date, tenor, convexity=0.0, dcc="ACT/365",
                 convention="modified_following"):
        """
        Args:
            delivery (str | pandas.Timestamp): any day of the delivery month.
            quote (float): rate quoted by the market.
            trade_date (str | pandas.Timestamp): trade date.
            tenor (str): length of the underlying period.
            convexity (float): [optional] adjustment added to the implied forward, defaults to 0.0
            dcc (str): day count convention of the period.
            convention (str): business day convention.
        """
        super().__init__(quote, trade_date, dcc, convention)
        self.tenor = tenor
        self.delivery = delivery
        self.convexity = convexity
        self.starting_date = business_adjustment(self.convention, imm_date(self.delivery))
        self.maturity = business_adjustment(self.convention,
                                            imm_date(pd.Timestamp(self.delivery) + tenor_offset(self.tenor)))
        self.accrual = accrual_factor(self.dcc, self.starting_date, self.maturity).item()

    def __repr__(self):
        return f"Future({self.starting_date.strftime('%Y-%m-%d')}, quote = {self.quote:.4%})"

    def implied_quote(self, curve: YieldCurve):
        return super().implied_quote(curve) + self.convexity


class Swap:
    """
    Fixed rate exchanged against a floating rate over a common period.
    """
    _SPOT_LAG = 2
    trade_date = Date()
    dcc = DayCountConvention()
    convention = BusinessConvention()

    def __init__(self, tenor, quote, trade_date, frequency=1, float_frequency=None, dcc="30/360",
                 convention="modified_following", discount_curve=None):
        """
        Args:
            tenor (str): length of the swap, measured from spot.
            quote (float): par rate quoted by the market.
            trade_date (str | pandas.Timestamp): trade date.
            frequency (int): payments per year on the fixed leg.
            float_frequency (int): [optional] payments per year on the floating leg, defaults to the
                fixed leg frequency.
            dcc (str): day count convention of the fixed leg.
            convention (str): business day convention.
            discount_curve (YieldCurve): [optional] curve discounting both legs, defaults to the curve
                the floating leg is projected off.
        """
        self.tenor = tenor
        self.quote = quote
        self.trade_date = trade_date
        self.frequency = frequency
        self.float_frequency = float_frequency or frequency
        self.dcc = dcc
        self.convention = convention
        self.discount = discount_curve
        self.spot_date = business_days_after(self.trade_date, self._SPOT_LAG)
        self.maturity = business_adjustment(self.convention, self.spot_date + tenor_offset(self.tenor))
        self.schedule = Schedule(self.spot_date, self.maturity, self.frequency, self.convention)
        self.floating_schedule = Schedule(self.spot_date, self.maturity, self.float_frequency, self.convention)
        self.float_starting_date = self.floating_schedule.schedule["startingDate"]
        self.float_payment_date = self.floating_schedule.schedule["paymentDate"]
        self.payment_date = self.schedule.schedule["paymentDate"]
        self.accrual = accrual_factor(self.dcc, self.schedule.schedule["startingDate"],
                                      self.schedule.schedule["endingDate"])

    def __repr__(self):
        return f"Swap(tenor= {self.tenor}, quote = {self.quote:.4%})"

    def implied_quote(self, curve: YieldCurve):
        """
        Fixed rate that makes the two legs worth the same.
        Args:
            curve (YieldCurve): curve the floating leg is projected off.
        Returns:
            float
        """
        discount = self.discount or curve
        p_start = curve.discount_factor_at(self.float_starting_date)
        p_end = curve.discount_factor_at(self.float_payment_date)
        d_float = discount.discount_factor_at(self.float_payment_date)
        d_fixed = discount.discount_factor_at(self.payment_date)

        return ((p_start / p_end - 1) @ d_float) / (d_fixed @ self.accrual)
