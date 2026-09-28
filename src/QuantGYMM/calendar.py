from .descriptors import *
from .utils import *
from pandas.tseries.offsets import DateOffset, MonthEnd
import numpy as np

__all__ = ["Schedule"]


class Schedule:
    """
    Schedule object for coupon planning.
    """
    start_date = Date(sterilize_attr=["_schedule"])
    end_date = Date(sterilize_attr=["_schedule"])
    frequency = PositiveNumber(sterilize_attr=["_schedule"])
    convention = BusinessConvention(sterilize_attr=["_schedule"])
    eom = Boolean(sterilize_attr=["_schedule"])
    payment_convention = BusinessConvention(sterilize_attr=["_schedule"])
    first_date = Date(sterilize_attr=["_schedule"], none_accepted=True)
    next_to_last_date = Date(sterilize_attr=["_schedule"], none_accepted=True)

    def __init__(self, start_date, end_date, frequency, convention="modified_following", eom=True,
                 payment_convention="following", first_date=None, next_to_last_date=None):
        """
        Args:
            start_date (str | pandas.Timestamp): "YYYY-MM-DD" string indicating starting date
            end_date (str | pandas.Timestamp): "YYYY-MM-DD" string indicating ending date
            frequency (float | int): frequency of payment
            convention (str): business day convention of the accrual dates (default is "modified_following");
                              "unadjusted" keeps the contractual dates
            eom (bool): end of month rule (default is True)
            payment_convention (str): business day convention moving each accrual end to its payment date
                                      (default is "following")
            first_date (str | pandas.Timestamp): [optional] first coupon date, for an irregular first period
            next_to_last_date (str | pandas.Timestamp): [optional] coupon date before maturity, for an irregular
                                                        last period
        """
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency
        self.convention = convention
        self.eom = eom
        self.payment_convention = payment_convention
        self.first_date = first_date
        self.next_to_last_date = next_to_last_date
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
        """
        Coupon dates generated backward from maturity, as QuantLib's DateGeneration.Backward.
        Returns:
            dict of numpy.ndarray: accrual start and end, payment date, and the regular period each accrual
            period is measured against (ACT/ACT ICMA).
        """
        bounds = [d for d in (self.start_date, self.first_date, self.next_to_last_date, self.end_date) if d is not None]
        if any(a >= b for a, b in zip(bounds, bounds[1:])):
            raise ValueError("Dates must satisfy start_date < first_date < next_to_last_date < end_date.")

        def adjust(d):
            return business_adjustment(self.convention, d)

        def to_month_end(d, anchor):
            return d + MonthEnd(0) if self.eom and anchor.is_month_end else d

        step = DateOffset(years=int((1 / self.frequency) // 1), months=int((1 / self.frequency) % 1 * 12))
        seed = self.end_date if self.next_to_last_date is None else self.next_to_last_date
        date = [self.end_date] if self.next_to_last_date is None else [self.end_date, self.next_to_last_date]
        exit_date = self.start_date if self.first_date is None else self.first_date
        k = 1
        while (d := to_month_end(seed - k * step, seed)) >= exit_date:
            if adjust(d) != adjust(date[-1]):
                date.append(d)
            k += 1
        if self.first_date is not None and adjust(self.first_date) != adjust(date[-1]):
            date.append(self.first_date)
        irregular_first = adjust(self.start_date) != adjust(date[-1])
        if irregular_first:
            date.append(self.start_date)
        date = date[::-1]
        irregular_last = (self.next_to_last_date is not None and len(date) > 2
                          and to_month_end(self.end_date - step, self.end_date) != self.next_to_last_date)

        accrual = np.array([adjust(d).normalize() for d in date])
        payment = np.array([business_adjustment(self.payment_convention, d).normalize() for d in accrual[1:]])
        reference_start, reference_end = accrual[:-1].copy(), accrual[1:].copy()
        if irregular_first:
            reference_start[0] = adjust(to_month_end(accrual[1] - step, date[1])).normalize()
        if irregular_last:
            reference_end[-1] = adjust(to_month_end(accrual[-2] + step, date[-2])).normalize()
        return {"startingDate": accrual[:-1], "endingDate": accrual[1:], "paymentDate": payment,
                "referenceStart": reference_start, "referenceEnd": reference_end}
