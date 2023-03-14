from .descriptors import *
from .utils import *
from pandas.tseries.offsets import DateOffset, MonthEnd
import numpy as np
import pandas as pd

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
