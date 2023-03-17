import pandas

from .descriptors import *
import numpy as np
import pandas as pd
import scipy
from .utils import *
from pandas.tseries.offsets import DateOffset, BDay

__all__ = ["SwapRateCurve", "DiscountCurve", "EuriborCurve"]


class SwapRateCurve:
    """
    Swap rate curve object for handling swap rate.
    """
    _SPOT_LAG = 2

    convention = BusinessConvention(sterilize_attr=["_interpolated_rates"])
    dcc = DayCountConvention(sterilize_attr=["_interpolated_rates"])
    trade_date = Date(sterilize_attr=["_interpolated_rates"])
    frequency = PositiveNumber(sterilize_attr=["_interpolated_rates"])

    def __init__(self, swap_rate, trade_date, frequency, business_convention="modified_following", dcc="30/360",
                 interpolation="linear"):
        """
        Args:
            swap_rate (pandas.DataFrame): swap rate dataframe, should have datetime index and numbers as column
                                        names (e.g. 1, 1.5, 2, ...) corresponding to swap maturities.
            frequency (int): swap fixed leg payment frequency
            business_convention (str): business convention
            trade_date (str | pandas.Timestamp): "YYYY-MM-DD" string representing the trade date
            dcc (str): day count convention of the fixed swap leg
            interpolation (str): method to be used to interpolate swap rate
        """
        self.interpolation = interpolation
        self.convention = business_convention
        self.dcc = dcc
        self.frequency = frequency
        self.trade_date = trade_date
        self.swap_rates = swap_rate
        self._interpolated_rates = None

    @property
    def spot_lag(self):
        return self._SPOT_LAG

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
        interpolating_function = scipy.interpolate.interp1d(days_in_term, self.swap_rates.swapRate,
                                                            kind=self.interpolation)
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
        if isinstance(swap_rate.index, pandas.DatetimeIndex):
            if len(swap_rate.columns) != 1:
                raise ValueError(f"Swap rate should be a DataFrame or a Series of one columns of swap rates."
                                 f" Got {len(swap_rate.columns)} columns.")
            return swap_rate
        if not swap_rate.index.is_monotonic_increasing:
            raise IndexError(
                "Index must be a DatetimeIndex or an increasing numeric index of year (e.g. 0.5, 1, 1.5, ...).")
        starting_date = self.trade_date + BDay(self.spot_lag)
        maturity = business_adjustment(self.convention,
                                       [starting_date + DateOffset(years=maturity // 1, months=12 * (maturity % 1)) for
                                        maturity in swap_rate.index])
        return pd.DataFrame({"term": maturity, "swapRate": swap_rate.iloc[:, 0]})


class DiscountCurve:
    """
    Discount curve object for extrapolation from swap rate curve.
    """

    dcc = DayCountConvention(sterilize_attr=["_spot_rates", "_discount_factors"])
    compounding = CompoundingConvention(sterilize_attr=["_spot_rates", "_discount_factors"])

    def __init__(self, rate_curve, compounding="annually_compounded", dcc="ACT/365", interpolation="cubic"):
        """
        Args:
            rate_curve (SwapRateCurve | SpotRateCurve | EuriborCurve): swap rate curve object or SpotRateCurve
            compounding (str): compounding of the discount curve (default is 'annually_compounded')
            dcc (str): day count convention for the market rates (default is ACT/365)
            interpolation (str): interpolation method to be used on spot rates interpolation
        """
        self._df = None
        self._sr = None
        self._discount_factors = None
        self._spot_rates = None
        self._shift = 0
        self._shift_flag = False
        self.dcc = dcc
        self.compounding = compounding
        self.interpolation = interpolation
        self.rate_curve = rate_curve

    @property
    def rate_curve(self):
        return self._rate_curve

    @rate_curve.setter
    def rate_curve(self, rate_curve):

        if isinstance(rate_curve, SwapRateCurve):
            self.trade_date = rate_curve.trade_date
            self._rate_curve = rate_curve
            self._starting_date = rate_curve.trade_date + BDay(rate_curve.spot_lag)
            self._discount_factors = None
            self._spot_rates = None
            self._get_df_from_swap_rate()
            self._get_spot()
        elif isinstance(rate_curve, (SpotRateCurve, EuriborCurve)):
            self.trade_date = rate_curve.trade_date
            self._rate_curve = rate_curve
            self._starting_date = rate_curve.trade_date + BDay(rate_curve.spot_lag)
            self._discount_factors = None
            self._spot_rates = None
            self._sr = rate_curve.sr

        else:
            raise ValueError("You need to pass a 'SwapRateCurve' object or a 'SpotRateCurve.")

    @property
    def discount_factors(self):
        if self._discount_factors is None:
            self._get_df_from_spot_rate()
        return self._discount_factors

    @discount_factors.setter
    def discount_factors(self, value):
        raise ValueError("Can't set discount factors directly, use 'set_discount_factors' method.")

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
        return f"DiscountCurve(trade_date={self.rate_curve.trade_date.strftime('%Y-%m-%d')}, " \
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

    def set_discount_factors(self, df) -> None:
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
        maturity = [self._starting_date] + self.rate_curve.interpolated_rates.term.to_list()
        afs = accrual_factor(self.rate_curve.dcc, maturity)
        annuity, df = np.array([0]), np.array([1])
        for sr, af in zip(self.rate_curve.interpolated_rates.interpolatedSwapRate, afs):
            df = np.append(df, (1 - annuity[-1] * sr) / (1 + af * sr))
            annuity = np.append(annuity, annuity[-1] + df[-1] * af)
        self._df = pd.DataFrame(
            {"maturity": maturity[1:], "annuity": annuity[1:], "discountFactor": df[1:],
             "accrualFactor": afs})

    def _interpolate_spot_rates(self):
        day_since_start = np.array([d.days for d in self.sr.maturity - self._starting_date])
        interpolating_function = scipy.interpolate.interp1d(day_since_start, self.sr.spotRate,
                                                            kind=self.interpolation)

        date_to_interpolate = pd.date_range(self.sr.maturity.iloc[0], self.sr.maturity.iloc[-1])
        days_to_interpolate = np.array([d.days for d in date_to_interpolate - self._starting_date])

        af_since_trade = accrual_factor(self.dcc, self.trade_date, date_to_interpolate)

        spot_rates = pd.DataFrame({"maturity": date_to_interpolate, "term": af_since_trade,
                                   "spotRate": interpolating_function(days_to_interpolate)})

        # for shorter maturity, simply back-propagate first euribor rate available
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
        if isinstance(other, (EuriborCurve, SpotRateCurve)):
            self._sr = pd.concat([other.sr, self.sr],
                                 ignore_index=True).drop_duplicates("maturity", keep="first").sort_values("maturity")
            self._spot_rates = None
        else:
            raise ValueError(f"DiscountCurve can be added only to EuriborCurve or SpotRateCurve."
                             f" Got '{other.__class__.__name__}'.")
        return self


class EuriborCurve:
    """
    Euribor curve object for money market rates.
    """
    _DCC = "ACT/360"
    _BUSINESS_CONVENTION = "modified_following"
    _SPOT_LAG = 2
    _ALLOWED_MATURITIES = ["1W", "1M", "3M", "6M", "12M",
                           "1 week", "1 month", "3 month", "6 month", "12 month"]
    _OFFSETS = {"1W": DateOffset(days=7), "1M": DateOffset(months=1), "3M": DateOffset(months=3),
                "6M": DateOffset(months=6), "12M": DateOffset(months=12),
                "1 week": DateOffset(days=7), "1 month": DateOffset(months=1), "3 month": DateOffset(months=3),
                "6 month": DateOffset(months=6), "12 month": DateOffset(months=12)}

    trade_date = Date(sterilize_attr=["_spot_rates", "_discount_factors", "_sr"])
    euribor = DataFrame(sterilize_attr=["_spot_rates", "_discount_factors", "_sr"])

    def __init__(self, euribor, trade_date, interpolation="cubic"):
        """
        Args:
            euribor (pandas.DataFrame): dataframe with euribor rate 1M, 3M, 6M and 12M.
            trade_date (str | pandas.Timestamp): "YYYY-MM-DD" or pandas.Timestamp for the starting date
        """

        self.trade_date = trade_date
        self.euribor = euribor
        self._starting_date = self.trade_date + BDay(self._SPOT_LAG)
        self.interpolation = interpolation
        self._sr = None
        self._discount_factors = None
        self._spot_rates = None

    @property
    def spot_lag(self):
        return self._SPOT_LAG

    @property
    def sr(self):
        if self._sr is None:
            self._process_spot_rate()
        return self._sr

    @property
    def spot_rates(self):
        if self._spot_rates is None:
            self._interpolate_spot_rates()
        return self._spot_rates

    @property
    def discount_factors(self):
        if self._discount_factors is None:
            self._get_discount_factors()
            return self._discount_factors

    @discount_factors.setter
    def discount_factors(self, value):
        raise ValueError("Can't do this!")

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

    def __repr__(self):
        return f"EuriborCurve(trade_date={self.trade_date.strftime('%Y-%m-%d')})"

    def _process_spot_rate(self):
        if not all([name in self._ALLOWED_MATURITIES for name in self.euribor.columns]):
            raise ValueError(f"Column names of Euribor data not valid. Valid column names"
                             f" are {', '.join(self._ALLOWED_MATURITIES)}.")
        maturity = self._starting_date + self.euribor.columns.map(self._OFFSETS)
        maturity = pd.Index(business_adjustment(self._BUSINESS_CONVENTION, maturity))
        self._sr = pd.DataFrame({"maturity": maturity,
                                 "term": accrual_factor(self._DCC, self.trade_date, maturity),
                                 "spotRate": self.euribor.loc[self.trade_date]})

    def _get_discount_factors(self):
        term = accrual_factor(self._DCC, self.trade_date, self.spot_rates.maturity)
        spot_rate = self.spot_rates.spotRate.to_numpy()
        self._discount_factors = pd.DataFrame({"maturity": self.spot_rates.maturity,
                                               "discountFactor": 1 / (1 + spot_rate * term)})

    def _interpolate_spot_rates(self):
        day_since_start = np.array([d.days for d in self.sr.maturity - self._starting_date])
        interpolator = scipy.interpolate.interp1d(day_since_start, self.sr.spotRate, kind=self.interpolation)
        date_to_interpolate = pd.date_range(self.sr.maturity.iloc[0], self.sr.maturity.iloc[-1])
        days_to_interpolate = np.array([d.days for d in date_to_interpolate - self._starting_date])
        af_since_trade = accrual_factor(self._DCC, self.trade_date, date_to_interpolate)
        spot_rates = pd.DataFrame({"maturity": date_to_interpolate, "term": af_since_trade,
                                   "spotRate": interpolator(days_to_interpolate)})
        dates = pd.date_range(self.trade_date, date_to_interpolate[0], inclusive="left")
        lm = pd.DataFrame({"maturity": dates, "term": accrual_factor(self._DCC, dates[0], dates)})
        self._spot_rates = pd.concat([lm, spot_rates], ignore_index=True).bfill()

    def __add__(self, other):

        if isinstance(other, (DiscountCurve, SpotRateCurve)):
            other._sr = pd.concat([self.sr, other.sr],
                                  ignore_index=True).drop_duplicates("maturity", keep="first").sort_values("maturity")
            other._spot_rates = None
        else:
            raise ValueError(f"EuriborCurve can be added only to DiscountCurve or SpotRateCurve. "
                             f"Got '{other.__class__.__name__}'.")
        return other


class SpotRateCurve:
    """
    Spot rate curve for handling spot rates
    """
    _SPOT_LAG = 2
    spot_rates_data = DataFrame(index_type=pd.DatetimeIndex, sterilize_attr=["_spot_rates", "_sr", "_discount_factors"])
    trade_date = Date(sterilize_attr=["_spot_rates", "_sr", "_discount_factors"])
    dcc = DayCountConvention(sterilize_attr=["_spot_rates", "_sr", "_discount_factors"])
    business_convention = BusinessConvention(sterilize_attr=["_spot_rates", "_sr", "_discount_factors"])
    compounding = CompoundingConvention(sterilize_attr=["_spot_rates", "_sr", "_discount_factors"])

    def __init__(self, spot_rates, trade_date, dcc="ACT/365", business_convention="modified_following",
                 interpolation="cubic", compounding="annually_compounded"):
        """
        Args:
            spot_rates (pandas.DataFrame): dataframe of spot rates
            trade_date (str | pandas.Timestamp): "YYYY-MM-DD" or pandas.Timestamp for the starting date
            dcc (str): day count convention
            business_convention (str): business convention
        """

        self._discount_factors = None
        self._spot_rates = None
        self._sr = None
        self.compounding = compounding
        self.spot_rates_data = spot_rates
        self.trade_date = trade_date
        self.dcc = dcc
        self.business_convention = business_convention
        self.interpolation = interpolation
        self._starting_date = self.trade_date + BDay(self._SPOT_LAG)

    @property
    def spot_lag(self):
        return self._SPOT_LAG

    @property
    def sr(self):
        if self._sr is None:
            self._process_spot_rate()
        return self._sr

    @property
    def spot_rates(self):
        if self._spot_rates is None:
            self._interpolate_spot_rates()
        return self._spot_rates

    @property
    def discount_factors(self):
        if self._discount_factors is None:
            self._get_discount_factors()
        return self._discount_factors

    @discount_factors.setter
    def discount_factors(self, value):
        raise ValueError("Can't do this!")

    def _process_spot_rate(self):
        af_since_trade = accrual_factor(self.dcc, self.trade_date, self.spot_rates_data.index)
        self._sr = pd.DataFrame({"maturity": self.spot_rates_data.index, "term": af_since_trade,
                                 "spotRate": self.spot_rates_data.iloc[:, 0].values})

    def _interpolate_spot_rates(self):
        day_since_start = np.array([d.days for d in self.sr.maturity - self._starting_date])
        interpolator = scipy.interpolate.interp1d(day_since_start, self.sr.spotRate, kind=self.interpolation)
        date_to_interpolate = pd.date_range(self.sr.maturity.iloc[0], self.sr.maturity.iloc[-1])
        days_to_interpolate = np.array([d.days for d in date_to_interpolate - self._starting_date])
        af_since_trade = accrual_factor(self.dcc, self.trade_date, date_to_interpolate)
        spot_rates = pd.DataFrame({"maturity": date_to_interpolate, "term": af_since_trade,
                                   "spotRate": interpolator(days_to_interpolate)})
        dates = pd.date_range(self.trade_date, date_to_interpolate[0], inclusive="left")
        lm = pd.DataFrame({"maturity": dates, "term": accrual_factor(self.dcc, dates[0], dates)})
        self._spot_rates = pd.concat([lm, spot_rates], ignore_index=True).bfill().drop_duplicates("maturity",
                                                                                                  keep="last")

    def _get_discount_factors(self):
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

        if isinstance(other, (DiscountCurve, EuriborCurve)):
            other._sr = pd.concat([self.sr, other.sr],
                                  ignore_index=True).drop_duplicates("maturity", keep="first").sort_values("maturity")
            other._spot_rates = None
        else:
            raise ValueError(f"EuriborCurve can be added only to DiscountCurve or SpotRateCurve. "
                             f"Got '{other.__class__.__name__}'.")
        return other

    def __repr__(self):
        return f"SpotRateCurve(trade_date={self.trade_date.strftime('%Y-%m-%d')}, compounding={self.compounding}, dcc={self.dcc})"
