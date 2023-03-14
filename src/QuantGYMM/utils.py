import numpy as np
from collections.abc import Iterable
from pandas.tseries.offsets import BDay, Day
from dateutil.easter import easter


def is_easter(date) -> bool:
    return date.date() == easter(date.year)


def is_holy_friday(date) -> bool:
    return date == (easter(date.year) - Day(2))


def is_holy_monday(date) -> bool:
    return date == (easter(date.year) + Day(1))


def is_christmas(date) -> bool:
    return (date.month == 12) and (date.day == 25)


def is_saint_stephen(date) -> bool:
    return (date.month == 12) and (date.day == 26)


def is_labour_day(date) -> bool:
    return (date.month == 5) and (date.day == 1)


def is_new_year_day(date) -> bool:
    return (date.month == 1) and (date.day == 1)


def is_target_holiday(date) -> bool:
    e = is_easter(date)
    hf = is_holy_friday(date)
    hm = is_holy_monday(date)
    c = is_christmas(date)
    ss = is_saint_stephen(date)
    ld = is_labour_day(date)
    ny = is_new_year_day(date)
    return e + hf + hm + c + ss + ld + ny != 0


def is_bd(date) -> bool:
    """
    Checks if date is business day considering TARGET calendar.
    Args:
        date (pandas.Timestamp): date to check.
    Returns:
        bool
    """

    return (date == date + Day(1) - BDay(1)) and not is_target_holiday(date)


def modified_following(date):
    """
    Performs date adjustment according to the modified following convention.
    Args:
        date (pandas.Timestamp): date to be modified.
    Returns:
        date adjusted for business convention.
    """
    if is_bd(date):
        return date
    elif (date + BDay(1)).month != date.month:
        return date - BDay(1)
    else:
        return date + BDay(1)


def following(date):
    """
    Performs date adjustment according to the following convention.
    Args:
        date (pandas.Timestamp): date to be modified
    Returns:
        date adjusted for business convention.
    """
    if is_bd(date):
        return date
    else:
        return date + BDay(1)


def modified_following_bimonthly(date):
    """
    Performs date adjustment according to the modified following bimonthly convention.
    Args:
        date (pandas.Timestamp): date to be modified
    Returns:
        date adjusted for business convention.
    """
    if is_bd(date):
        return date
    elif (date + BDay(1)).month != date.month or (date.day <= 15 < (date + BDay(1)).day):
        return date - BDay(1)
    else:
        return date + BDay(1)


def preceding(date):
    """
    Performs date adjustment according to the preceding convention.
    Args:
        date (pandas.Timestamp): date to be modified
    Returns:
        date adjusted for business convention.
    """
    if is_bd(date):
        return date
    else:
        return date - BDay(1)


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


def thirty360(*dates):
    """
    Compute accrual factor according to day count convention 30/360.
    Args:
        dates (Iterable, pandas.Timestamp): two dates or a list of dates.
    Returns:
        numpy.ndarray of accrual factors.
    """
    af = np.array([])
    if len(dates) == 1:
        dates = dates[0]
        for start, end in zip(dates, dates[1:]):
            if start.day == 31:
                d1 = 30
            else:
                d1 = start.day
            if (end.day == 31) and ((start.day == 31) or (start.day == 30)):
                d2 = 30
            else:
                d2 = end.day
            y1, y2 = start.year, end.year
            m1, m2 = start.month, end.month
            af = np.append(af, ((360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)) / 360))
    elif len(dates) == 2:
        if not isinstance(dates[0], Iterable) and not isinstance(dates[1], Iterable):
            d1 = 30 if dates[0].day == 31 else dates[0].day
            if (dates[1].day == 31) and ((d1 == 31) or (d1 == 30)):
                d2 = 30
            else:
                d2 = dates[1].day
            m1, y1 = dates[0].month, dates[0].year
            m2, y2 = dates[1].month, dates[1].year
            af = np.append(af, ((360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)) / 360))
        elif not isinstance(dates[0], Iterable) and isinstance(dates[1], Iterable):
            d1 = 30 if dates[0].day == 31 else dates[0].day
            m1, y1 = dates[0].month, dates[0].year
            for end in dates[1]:
                if (end.day == 31) and ((d1 == 31) or (d1 == 30)):
                    d2 = 30
                else:
                    d2 = end.day
                m2, y2 = end.month, end.year
                af = np.append(af, ((360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)) / 360))
        else:
            for start, end in zip(dates[0], dates[1]):
                if start.day == 31:
                    d1 = 30
                else:
                    d1 = start.day
                if (end.day == 31) and ((start.day == 31) or (start.day == 30)):
                    d2 = 30
                else:
                    d2 = end.day
                y1, y2 = start.year, end.year
                m1, m2 = start.month, end.month
                af = np.append(af, ((360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)) / 360))
    else:
        raise ValueError("Wrong dimension for dates.")
    return af


def act360(*dates):
    """
    Compute accrual factor according to day count convention ACT/360.
    Args:
        dates (Iterable, pandas.Timestamp): two dates or a list of dates.
    Returns:
        numpy.ndarray of accrual factors.
    """
    af = np.array([])
    if len(dates) == 1:
        dates = dates[0]
        for start, end in zip(dates, dates[1:]):
            af = np.append(af, (end - start).days / 360)
    elif len(dates) == 2:
        if not isinstance(dates[0], Iterable) and not isinstance(dates[1], Iterable):
            start, end = dates[0], dates[1]
            af = np.append(af, (end - start).days / 360)
        elif not isinstance(dates[0], Iterable) and isinstance(dates[1], Iterable):
            start = dates[0]
            for end in dates[1]:
                af = np.append(af, (end - start).days / 360)
        else:
            for start, end in zip(dates[0], dates[1]):
                af = np.append(af, (end - start).days / 360)
    else:
        raise ValueError("Wrong dimension for dates.")
    return af


def act365(*dates):
    """
    Compute accrual factor according to day count convention ACT/365.
    Args:
        dates (Iterable, pandas.Timestamp): two dates or a list of dates.
    Returns:
        numpy.ndarray of accrual factors.
    """
    accrual_factor = np.array([])
    if len(dates) == 1:
        dates = dates[0]
        for start, end in zip(dates, dates[1:]):
            accrual_factor = np.append(accrual_factor, (end - start).days / 365)
    elif len(dates) == 2:
        if not isinstance(dates[0], Iterable) and not isinstance(dates[1], Iterable):
            start, end = dates[0], dates[1]
            accrual_factor = np.append(accrual_factor, (end - start).days / 365)
        elif not isinstance(dates[0], Iterable) and isinstance(dates[1], Iterable):
            start = dates[0]
            for end in dates[1]:
                accrual_factor = np.append(accrual_factor, (end - start).days / 365)
        else:
            for start, end in zip(dates[0], dates[1]):
                accrual_factor = np.append(accrual_factor, (end - start).days / 365)
    else:
        raise ValueError("Wrong dimension for dates.")
    return accrual_factor


def accrual_factor(dcc, *dates):
    """
    Wrapper for accrual factor calculation according to different business conventions.
    Args:
        dcc (str): day count convention;
        dates (Iterable | pandas.Timestamp): dates with respect to which determine accrual factor.
    Returns:
        np.ndarray of accrual factors.
    """
    match dcc:
        case "ACT/360":
            return act360(*dates)
        case "ACT/365":
            return act365(*dates)
        case "30/360":
            return thirty360(*dates)
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
