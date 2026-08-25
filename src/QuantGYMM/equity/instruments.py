import pandas as pd
import numpy as np

from ..descriptors import *

__all__ = ["Stock"]


class Stock:
    ticker = String()
    quantity = PositiveNumber()
    currency = String(none_accepted=True, return_if_none="EUR")
    isin = String(none_accepted=True)
    dividend_yield = FloatNumber(none_accepted=True)

    def __init__(self, ticker, quantity=1, isin=None, currency=None, dividend_yield=None):
        self.ticker = ticker
        self.quantity = quantity
        self.isin = isin
        self.currency = currency
        self.dividend_yield = dividend_yield
        self._evaluation_date = None
        self._historical_prices = None
        self._dividends = None

    @property
    def evaluation_date(self):
        if self._evaluation_date is None:
            raise ValueError("Evaluation date has not been set. Call 'set_evaluation_date' method to set it.")
        return self._evaluation_date

    @property
    def historical_prices(self):
        if self._historical_prices is None:
            raise ValueError(
                "Historical prices have not been set. Call 'set_historical_prices' or 'load_yahoo_history' method to set them.")
        return self._historical_prices

    @property
    def dividends(self):
        if self._dividends is None:
            raise ValueError(
                "Dividend history has not been set. Call 'set_dividend_history' or 'load_yahoo_history' method to set it.")
        return self._dividends

    @property
    def price(self):
        price = self.historical_prices.asof(self.evaluation_date)
        if np.isnan(price):
            raise ValueError(f"Evaluation date {self.evaluation_date.strftime('%Y-%m-%d')} precedes the first "
                             f"available price observation ({self.historical_prices.index[0].strftime('%Y-%m-%d')})."
                             )
        return float(price)

    @property
    def market_value(self):
        return self.price * self.quantity

    @property
    def simple_returns(self):
        """Price-only simple (percentage) returns. Use this, not 'log_returns',
        wherever returns must be combined ACROSS positions (e.g. portfolio
        aggregation) — only simple returns are exact under weighted summation."""
        prices = self.historical_prices.loc[:self.evaluation_date]
        return prices.pct_change().dropna()

    @property
    def log_returns(self):
        """Price-only log returns. Use this, not 'simple_returns', for a single
        position's own volatility/time-series statistics (additive across
        time, not across positions)."""
        prices = self.historical_prices.loc[:self.evaluation_date]
        return np.log(prices / prices.shift(1)).dropna()

    @property
    def simple_total_returns(self):
        """Dividend-adjusted simple (percentage) returns."""
        prices = self.historical_prices.loc[:self.evaluation_date]
        dividends = self.dividends.reindex(prices.index, fill_value=0.0)
        return ((prices + dividends) / prices.shift(1) - 1).dropna()

    @property
    def log_total_returns(self):
        """Dividend-adjusted log returns."""
        prices = self.historical_prices.loc[:self.evaluation_date]
        dividends = self.dividends.reindex(prices.index, fill_value=0.0)
        return np.log((prices + dividends) / prices.shift(1)).dropna()

    def set_evaluation_date(self, date) -> None:
        """
        Set evaluation date for market price calculation.
        Args:
            date (str | pandas.Timestamp): trade date
        """
        try:
            self._evaluation_date = pd.to_datetime(date)
        except Exception:
            raise ValueError(f"Can't convert {date} to datetime.")

    def load_yahoo_history(self, lookback=pd.DateOffset(years=5)) -> None:
        """
        Download historical prices and dividends from Yahoo Finance, ending at the
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
            raise ValueError(f"Can't download history data from Yahoo API for ticker {self.ticker}.")
        self.set_historical_prices(history["Close"])
        self.set_dividend_history(history["Dividends"])

    def set_historical_prices(self, prices) -> None:
        """
        Set the historical prices series for the stock.
        Args:
            prices (pandas.Series | pandas.DataFrame): historical prices (usually closing price) indexed by date.
        """

        if isinstance(prices, pd.DataFrame):
            if prices.shape[1] != 1:
                raise ValueError("'prices' must be a Series or a single columns DataFrame.")
            prices = prices.iloc[:, 0]
        if not isinstance(prices, pd.Series):
            raise ValueError("'prices' must be a Series or a single columns DataFrame.")
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("'prices' must be indexed by a DatetimeIndex.")
        prices = prices.dropna()
        if prices.empty:
            raise ValueError("'prices' must not be empty.")
        if (prices <= 0).any():
            raise ValueError("'prices' must be strictly positive.")
        if prices.index.tz is not None:
            prices = prices.tz_localize(None)
        self._historical_prices = prices.sort_index().copy()

    def set_dividend_history(self, dividends) -> None:
        """
        Set the per-share dividend history for the stock (dividend paid on each
        date, 0 on non-ex-dividend dates). Used only for 'simple_total_returns' /
        'log_total_returns' / dividend-adjusted volatility; an empty history (no
        dividends ever paid) is valid.
        Args:
            dividends (pandas.Series | pandas.DataFrame): dividend payments indexed by date.
        """
        if isinstance(dividends, pd.DataFrame):
            if dividends.shape[1] != 1:
                raise ValueError("'dividends' must be a Series or a single-column DataFrame.")
            dividends = dividends.iloc[:, 0]
        if not isinstance(dividends, pd.Series):
            raise ValueError("'dividends' must be a Series or a single-column DataFrame.")
        if not isinstance(dividends.index, pd.DatetimeIndex):
            raise ValueError("'dividends' must be indexed by a DatetimeIndex.")
        dividends = dividends.dropna()
        if (dividends < 0).any():
            raise ValueError("'dividends' must be non-negative.")
        if dividends.index.tz is not None:
            dividends = dividends.tz_localize(None)
        self._dividends = dividends.sort_index().copy()

    def realized_volatility(self, n_obs=None, total_return=False) -> float:
        """
        Compute realized volatility (std of log returns) up to the evaluation date.
        Args:
            n_obs (int, default None): number of most recent return observations to use.
                If 'None', the full available history is used.
            total_return (bool, default False): if True, use dividend-adjusted
                log total returns ('log_total_returns') instead of price-only
                log returns ('log_returns'). Requires 'dividends' to have been set.
        Returns:
            float: realized volatility.
        """
        returns = self.log_total_returns if total_return else self.log_returns

        if n_obs is None:
            window = returns
        else:
            if n_obs < 2:
                raise ValueError(f"n_obs must be at least 2, got {n_obs}.")
            if returns.shape[0] < n_obs:
                raise ValueError(f"Requested n_obs={n_obs}, but only {returns.shape[0]} obs are available.")
            window = returns.iloc[-n_obs:]

        if window.shape[0] < 2:
            raise ValueError("At least 2 return observations are required to compute a standard deviation.")
        return window.std(ddof=1)
