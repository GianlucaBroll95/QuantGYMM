import functools
import re

import numpy as np
import pandas as pd
import datetime as dt
from collections.abc import Iterable
from pandas.tseries.offsets import DateOffset
from dateutil.easter import easter

__all__ = ["tenor_offset", "is_bd", "is_target_holiday", "modified_following", "modified_following_bimonthly",
           "preceding", "following", "business_adjustment", "business_days_before", "business_days_after",
           "number_of_month", "thirty360", "thirty_e_360", "act365", "act360", "act_act", "act_act_icma", "nl365",
           "accrual_factor", "imm_date"]
_TENOR_UNITS = {"W": "weeks", "M": "months", "Y": "years"}
_TENOR = re.compile(r"\s*(\d+)\s*([WMY])\s*", re.IGNORECASE)


def tenor_offset(tenor) -> DateOffset:
    """
    Date offset a tenor stands for.
    Args:
        tenor (str): a number followed by 'W', 'M' or 'Y', e.g. '1W', '18M', '50Y'.
    Returns:
        pandas.tseries.offsets.DateOffset
    """
    match = _TENOR.fullmatch(str(tenor))
    if match is None:
        raise ValueError(f"Invalid tenor '{tenor}': expected a number followed by 'W', 'M' or 'Y'.")
    size, unit = match.groups()
    return DateOffset(**{_TENOR_UNITS[unit.upper()]: int(size)})


@functools.lru_cache(maxsize=None)
def _target_holidays(year):
    easter_sunday = easter(year)
    return frozenset({
        dt.date(year, 1, 1),
        easter_sunday - dt.timedelta(days=2),
        easter_sunday,
        easter_sunday + dt.timedelta(days=1),
        dt.date(year, 5, 1),
        dt.date(year, 12, 25),
        dt.date(year, 12, 26),
    })


def is_target_holiday(date) -> bool:
    return date.date() in _target_holidays(date.year)


def imm_date(date) -> pd.Timestamp:
    """
    Third Wednesday of the month a date falls in.
    Args:
        date (str | pandas.Timestamp): any day of the month.
    Returns:
        pandas.Timestamp
    """
    first = pd.Timestamp(date).replace(day=1)
    return first + pd.Timedelta(days=(2 - first.weekday()) % 7 + 14)


def is_bd(date) -> bool:
    """
    Checks if date is business day considering TARGET calendar.
    Args:
        date (pandas.Timestamp): date to check.
    Returns:
        bool
    """

    return (date.weekday() < 5) and not is_target_holiday(date)


def modified_following(date):
    """
    Performs date adjustment according to the modified following convention.
    Args:
        date (pandas.Timestamp): date to be modified.
    Returns:
        date adjusted for business convention.
    """
    adjusted = following(date)
    return adjusted if adjusted.month == date.month else preceding(date)


def following(date):
    """
    Performs date adjustment according to the following convention.
    Args:
        date (pandas.Timestamp): date to be modified
    Returns:
        date adjusted for business convention.
    """
    while not is_bd(date):
        date += pd.Timedelta(days=1)
    return date


def modified_following_bimonthly(date):
    """
    Performs date adjustment according to the modified following bimonthly convention.
    Args:
        date (pandas.Timestamp): date to be modified
    Returns:
        date adjusted for business convention.
    """
    adjusted = following(date)
    if adjusted.month != date.month or date.day <= 15 < adjusted.day:
        return preceding(date)
    return adjusted


def preceding(date):
    """
    Performs date adjustment according to the preceding convention.
    Args:
        date (pandas.Timestamp): date to be modified
    Returns:
        date adjusted for business convention.
    """
    while not is_bd(date):
        date -= pd.Timedelta(days=1)
    return date


def business_days_before(dates, n) -> pd.Timestamp | np.ndarray:
    """
    Step back 'n' business days on the TARGET calendar.
    Args:
        dates (Iterable | pandas.Timestamp): dates to step back from;
        n (int): number of business days.
    Returns:
        dates moved back 'n' business days.
    """
    return _business_days_offset(dates, n, -pd.Timedelta(days=1))


def business_days_after(dates, n) -> pd.Timestamp | np.ndarray:
    """
     Step ahead 'n' business days on the TARGET calendar.
     Args:
         dates (Iterable | pandas.Timestamp): dates to step back from;
         n (int): number of business days.
     Returns:
         dates ahead 'n' business days.
     """
    return _business_days_offset(dates, n, pd.Timedelta(days=1))


def _business_days_offset(dates, n, step):
    if isinstance(dates, Iterable) and not isinstance(dates, str):
        return np.asarray([_business_days_offset(date, n, step) for date in dates])
    date = pd.Timestamp(dates)
    for _ in range(n):
        date += step
        while not is_bd(date):
            date += step
    return date


def business_adjustment(convention, dates):
    """
    Wrapper for business convention adjustment.
    Args:
        convention (str): business convention;
        dates (Iterable | pandas.Timestamp): dates to be modified.
    Returns:
        dates adjusted according to the business convention chosen.
    """
    if isinstance(dates, Iterable):
        match convention:
            case "unadjusted":
                return list(dates)
            case "preceding":
                return [preceding(m) for m in dates]
            case "following":
                return [following(m) for m in dates]
            case "modified_following":
                return [modified_following(m) for m in dates]
            case "modified_following_bimonthly":
                return [modified_following_bimonthly(m) for m in dates]
            case _:
                raise ValueError(f"Business convention '{convention}' not implemented.")
    else:
        match convention:
            case "unadjusted":
                return dates
            case "preceding":
                return preceding(dates)
            case "following":
                return following(dates)
            case "modified_following":
                return modified_following(dates)
            case "modified_following_bimonthly":
                return modified_following_bimonthly(dates)
            case _:
                raise ValueError(f"Business convention '{convention}' not implemented.")


def _is_sequence(x):
    """
    Helper function to check if x is a sequence, not just an iterable.
    """
    return isinstance(x, (list, tuple, np.ndarray, pd.Series, pd.Index))


def _apply_pairwise(dcc_function, *dates):
    """
    Dispatcher for day count convention
    """
    result = []

    if len(dates) == 1:
        dates = dates[0]
        for start, end in zip(dates, dates[1:]):
            result.append(dcc_function(start, end))
    elif len(dates) == 2:
        d0, d1 = dates
        if not _is_sequence(d0) and not _is_sequence(d1):
            result.append(dcc_function(d0, d1))
        elif not _is_sequence(d0) and _is_sequence(d1):
            for end in d1:
                result.append(dcc_function(d0, end))
        elif _is_sequence(d0) and _is_sequence(d1):
            if len(d0) != len(d1):
                raise ValueError(f"Mismatch in dates. d0 is {len(d0)}, d1 is {len(d1)}.")
            for start, end in zip(d0, d1):
                result.append(dcc_function(start, end))
        else:
            raise ValueError("Wrong dimension for dates.")
    else:
        raise ValueError("Wrong dimension for dates.")

    return np.asarray(result)


def _thirty360_single(start, end):
    d1 = 30 if start.day == 31 else start.day

    if (end.day == 31) and ((start.day == 31) or (start.day == 30)):
        d2 = 30
    else:
        d2 = end.day

    return (360 * (end.year - start.year) + 30 * (end.month - start.month) + (d2 - d1)) / 360


def thirty360(*dates):
    """
    Compute accrual factor according to day count convention 30/360.
    Args:
        dates (Iterable, pandas.Timestamp): two dates or a list of dates.
    Returns:
        numpy.ndarray of accrual factors.
    """
    return _apply_pairwise(_thirty360_single, *dates)


def _thirty_e_360_single(start, end):
    d1 = 30 if start.day == 31 else start.day
    d2 = 30 if end.day == 31 else end.day
    return (360 * (end.year - start.year) + 30 * (end.month - start.month) + (d2 - d1)) / 360


def thirty_e_360(*dates):
    """
    Compute accrual factor according to day count convention 30E/360 (Eurobond basis).
    Unlike 30/360 (US basis), the day-31 adjustment on the end date does not depend on the start date.
    Args:
        dates (Iterable, pandas.Timestamp): two dates or a list of dates.
    Returns:
        numpy.ndarray of accrual factors.
    """
    return _apply_pairwise(_thirty_e_360_single, *dates)


def act360(*dates):
    """
    Compute accrual factor according to day count convention ACT/360.
    Args:
        dates (Iterable, pandas.Timestamp): two dates or a list of dates.
    Returns:
        numpy.ndarray of accrual factors.
    """

    return _apply_pairwise(lambda d0, d1: (d1 - d0).days / 360, *dates)


def act365(*dates):
    """
    Compute accrual factor according to day count convention ACT/365.
    Args:
        dates (Iterable, pandas.Timestamp): two dates or a list of dates.
    Returns:
        numpy.ndarray of accrual factors.
    """
    return _apply_pairwise(lambda d0, d1: (d1 - d0).days / 365, *dates)


def _act_act_single(start, end):
    """
    ACT/ACT ISDA day count fraction between a single (start, end) date pair.
    Splits the period at year boundaries, dividing days falling in a leap year
    by 366 and days falling in a non-leap year by 365.
    """
    if start == end:
        return 0.0
    years = range(start.year, end.year + 1)
    total = 0.0
    for y in years:
        y_start = max(start, pd.Timestamp(year=y, month=1, day=1))
        y_end = min(end, pd.Timestamp(year=y + 1, month=1, day=1))
        if y_end <= y_start:
            continue
        days_in_year = 366 if pd.Timestamp(year=y, month=12, day=31).is_leap_year else 365
        total += (y_end - y_start).days / days_in_year
    return total


def act_act(*dates):
    """
    Compute accrual factor according to day count convention ACT/ACT (ISDA).
    Args:
        dates (Iterable, pandas.Timestamp): two dates or a list of dates.
    Returns:
        numpy.ndarray of accrual factors.
    """
    return _apply_pairwise(_act_act_single, *dates)


def _nl365_single(start, end):
    # count calendar days then subtract every Feb 29th falling strictly within (start, end]
    days = (end - start).days
    feb29_count = sum(
        1 for y in range(start.year, end.year + 1)
        if pd.Timestamp(y, 1, 1).is_leap_year and start < pd.Timestamp(y, 2, 29) <= end
    )
    return (days - feb29_count) / 365


def nl365(*dates):
    """
    Compute accrual factor according to day count convention NL/365 (No Leap / Actual 365 No Leap).
    Counts actual calendar days but excludes any February 29th falling within the period, dividing by 365.
    Args:
        dates (Iterable, pandas.Timestamp): two dates or a list of dates.
    Returns:
        numpy.ndarray of accrual factors.
    """
    return _apply_pairwise(_nl365_single, *dates)


def _act_act_icma_single(start, end, reference_start=None, reference_end=None):
    if end == start:
        return 0.0
    if end < start:
        return -_act_act_icma_single(end, start, reference_start, reference_end)
    reference_start = start if reference_start is None else reference_start
    reference_end = end if reference_end is None else reference_end
    months = int(np.floor(12 * (reference_end - reference_start).days / 365 + 0.5))
    if months == 0:
        reference_start, reference_end, months = start, start + DateOffset(years=1), 12
    period = months / 12
    if end <= reference_end:
        if start >= reference_start:
            return period * (end - start).days / (reference_end - reference_start).days
        previous = reference_start - DateOffset(months=months)
        if end > reference_start:
            return (_act_act_icma_single(start, reference_start, previous, reference_start)
                    + _act_act_icma_single(reference_start, end, reference_start, reference_end))
        return _act_act_icma_single(start, end, previous, reference_start)
    if reference_start > start:
        raise ValueError("Invalid dates: start < reference start < reference end < end.")
    fraction = _act_act_icma_single(start, reference_end, reference_start, reference_end)
    i = 0
    while True:
        next_start = reference_end + DateOffset(months=months * i)
        next_end = reference_end + DateOffset(months=months * (i + 1))
        if end < next_end:
            break
        fraction += period
        i += 1
    return fraction + _act_act_icma_single(next_start, end, next_start, next_end)


def act_act_icma(*dates, reference=None):
    """
    Compute accrual factor according to day count convention ACT/ACT ICMA (ISMA-251, bond basis).
    Args:
        dates (Iterable, pandas.Timestamp): two dates or a list of dates.
        reference (tuple): [optional] start and end of the regular coupon period each (start, end) pair
                           belongs to; defaults to the pair itself, which is right for regular periods.
    Returns:
        numpy.ndarray of accrual factors.
    """
    if reference is None:
        return _apply_pairwise(_act_act_icma_single, *dates)
    columns = [*dates, *reference]
    n = max(len(column) if _is_sequence(column) else 1 for column in columns)
    columns = [list(column) if _is_sequence(column) else [column] * n for column in columns]
    return np.asarray([_act_act_icma_single(*row) for row in zip(*columns)])


def accrual_factor(dcc, *dates, reference=None):
    """
    Wrapper for accrual factor calculation according to different business conventions.
    Args:
        dcc (str): day count convention;
        dates (Iterable | pandas.Timestamp): dates with respect to which determine accrual factor.
        reference (tuple): [optional] start and end of the regular coupon period of each pair, used by
                           ACT/ACT ICMA only.
    Returns:
        np.ndarray of accrual factors.
    """
    match dcc:
        case "ACT/360":
            return act360(*dates)
        case "ACT/365":
            return act365(*dates)
        case "ACT/ACT" | "ACT/ACT ISDA":
            return act_act(*dates)
        case "ACT/ACT ICMA":
            return act_act_icma(*dates, reference=reference)
        case "30/360":
            return thirty360(*dates)
        case "30E/360":
            return thirty_e_360(*dates)
        case "NL/365":
            return nl365(*dates)
        case _:
            raise ValueError(f"Day count convention '{dcc}' not implemented.")


def number_of_month(start, end) -> float:
    """
    Calculate the exact number of month between start and date.
    Args:
        start (pandas.Timestamp): start date
        end (pandas.Timestamp): end date
    Returns:
    Number of months between two date.
    """
    month = (end.year - start.year) * 12 + (end.month - start.month)
    day_correction = (end.day - start.day) / end.days_in_month
    return month + day_correction
