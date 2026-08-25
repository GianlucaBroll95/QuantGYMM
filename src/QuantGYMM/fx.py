"""
Foreign-exchange rates.

Shared, asset-class-agnostic market data: any asset class (equity, fixed
income, private markets) that needs to convert a position's market value into
a common base/reporting currency uses 'FXRate' the same way.
"""
import pandas as pd
import numpy as np

from .descriptors import *

__all__ = ["FXRate"]


class FXRate:
    """
    Exchange rate between two currencies. 'rate' is the number of units of
    'quote_currency' per one unit of 'base_currency' (matches Yahoo Finance's
    own 'BASEQUOTECY=X' ticker convention, e.g. EURUSD=X is USD per 1 EUR).
    'convert(amount)' converts an amount denominated in 'base_currency' into
    'quote_currency'.
    """
    base_currency = String()
    quote_currency = String()

    def __init__(self, base_currency, quote_currency):
        self.base_currency = base_currency
        self.quote_currency = quote_currency
        self._evaluation_date = None
        self._historical_rates = None

    @property
    def ticker(self):
        return f"{self.base_currency}{self.quote_currency}=X"

    @property
    def evaluation_date(self):
        if self._evaluation_date is None:
            raise ValueError("Evaluation date has not been set. Call 'set_evaluation_date' method to set it.")
        return self._evaluation_date

    @property
    def historical_rates(self):
        if self._historical_rates is None:
            raise ValueError(
                "Historical rates have not been set. Call 'set_historical_rates' or 'load_yahoo_history' method to set them.")
        return self._historical_rates

    @property
    def rate(self):
        rate = self.historical_rates.asof(self.evaluation_date)
        if np.isnan(rate):
            raise ValueError(f"Evaluation date {self.evaluation_date.strftime('%Y-%m-%d')} precedes the first "
                             f"available rate observation ({self.historical_rates.index[0].strftime('%Y-%m-%d')})."
                             )
        return float(rate)

    @property
    def returns(self):
        rates = self.historical_rates.loc[:self.evaluation_date]
        return np.log(rates / rates.shift(1)).dropna()

    def convert(self, amount) -> float:
        """
        Convert an amount denominated in 'base_currency' into 'quote_currency'
        at the evaluation date's rate.
        Args:
            amount (float): amount in 'base_currency'.
        Returns:
            float: equivalent amount in 'quote_currency'.
        """
        return amount * self.rate

    def set_evaluation_date(self, date) -> None:
        """
        Set evaluation date for rate lookup.
        Args:
            date (str | pandas.Timestamp): trade date
        """
        try:
            self._evaluation_date = pd.to_datetime(date)
        except Exception:
            raise ValueError(f"Can't convert {date} to datetime.")

    def load_yahoo_history(self, lookback=pd.DateOffset(years=5)) -> None:
        """
        Download the historical exchange rate from Yahoo Finance, ending at the
        evaluation date. Requires 'set_evaluation_date' to have been called first.
        Args:
            lookback (pandas.DateOffset | pandas.Timedelta, default 5 years): how far
                back from the evaluation date to fetch data.
        """
        end = self.evaluation_date
        start = end - lookback
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "Yahoo API requires yfinance to be installed. (pip install QuantGYMM[yahoo] or pip install yfinance)")
        history = yf.Ticker(self.ticker).history(start=start, end=end + pd.Timedelta(days=1), auto_adjust=False)
        if history.empty:
            raise ValueError(f"Can't download FX history from Yahoo API for pair {self.ticker}.")
        self.set_historical_rates(history["Close"])

    def set_historical_rates(self, rates) -> None:
        """
        Set the historical exchange rate series (units of 'quote_currency' per
        one unit of 'base_currency').
        Args:
            rates (pandas.Series | pandas.DataFrame): FX rates indexed by date.
        """
        if isinstance(rates, pd.DataFrame):
            if rates.shape[1] != 1:
                raise ValueError("'rates' must be a Series or a single-column DataFrame.")
            rates = rates.iloc[:, 0]
        if not isinstance(rates, pd.Series):
            raise ValueError("'rates' must be a Series or a single-column DataFrame.")
        if not isinstance(rates.index, pd.DatetimeIndex):
            raise ValueError("'rates' must be indexed by a DatetimeIndex.")
        rates = rates.dropna()
        if rates.empty:
            raise ValueError("'rates' must not be empty.")
        if (rates <= 0).any():
            raise ValueError("'rates' must be strictly positive.")
        if rates.index.tz is not None:
            rates = rates.tz_localize(None)
        self._historical_rates = rates.sort_index().copy()
