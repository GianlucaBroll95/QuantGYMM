"""
Equity portfolio aggregation, for risk management purposes.

'EquityPortfolio' groups several 'Stock' positions into a single book: an
aggregate market value in one base/reporting currency, a portfolio-level
historical return series, and historical-simulation VaR/CVaR.
"""
import pandas as pd
from sklearn.covariance import LedoitWolf
import numpy as np
from scipy.stats import norm
from .instruments import Stock
from ..fx import FXRate

__all__ = ["EquityPortfolio"]


class EquityPortfolio:
    """
    Collection of Stock positions for equity risk management.

    Portfolio returns are combined as SIMPLE (percentage) returns of the
    portfolio's own market-value level, not as a weighted sum of constituent
    LOG returns: simple returns of a sum of levels aggregate correctly across
    positions at a point in time, while log returns only aggregate correctly
    across time for a single position.
    """

    def __init__(self, positions, base_currency=None, lookback_window=pd.DateOffset(years=5)):
        """
        Args:
            positions (list[Stock]): equity positions making up the portfolio.
            base_currency (str): [optional] reporting currency, defaults to EUR.
        """
        positions = list(positions)
        if not positions:
            raise ValueError("'positions' must contain at least one Stock.")
        if not all(isinstance(p, Stock) for p in positions):
            raise TypeError("'positions' must all be Stock instances.")
        self.positions = positions
        self.base_currency = base_currency or "EUR"
        self.lookback_window = lookback_window
        self._evaluation_date = None
        self._fx_rates = {}

    @property
    def currencies(self):
        """Position currencies other than the base currency, needing an FXRate."""
        return sorted({p.currency for p in self.positions} - {self.base_currency})

    @property
    def evaluation_date(self):
        if self._evaluation_date is None:
            raise ValueError("Evaluation date has not been set. Call 'set_evaluation_date' method to set it.")
        return self._evaluation_date

    @property
    def market_value(self):
        """Aggregate market value of every position, in the base currency."""
        _ = self.evaluation_date  # enforce 'set_evaluation_date' has been called on the portfolio
        for p in self.positions:
            if p._historical_prices is None:
                p.load_yahoo_history(self.lookback_window)
        return sum(self._position_value_in_base(p) for p in self.positions)

    @property
    def weights(self):
        """Each position's share of 'market_value', keyed by ticker."""
        total = self.market_value
        return {p.ticker: self._position_value_in_base(p) / total for p in self.positions}

    @property
    def returns(self):
        """Portfolio simple (percentage) returns, up to the evaluation date."""
        return self._historical_value().pct_change().dropna()

    @property
    def asset_returns(self):
        """Portfolio components returns."""
        return self._historical_asset_value().pct_change().dropna()

    @property
    def variance(self):
        """Portfolio variance."""
        cov = self.covariance_matrix()
        weights = pd.Series(self.weights)
        return weights @ cov @ weights

    @property
    def volatility(self):
        """Portfolio volatility."""
        return np.sqrt(self.variance)

    @property
    def contribution_to_variance(self):
        """Single stock contribution to variance."""
        port_var = self.variance
        cov = self.covariance_matrix()
        w = pd.Series(self.weights)
        return w * (cov @ w) / port_var


    def set_evaluation_date(self, date) -> None:
        """
        Set the evaluation date for every position and every registered FXRate.
        Args:
            date (str | pandas.Timestamp): trade date
        """
        try:
            date = pd.to_datetime(date)
        except Exception as err:
            raise ValueError(f"Can't convert {date} to datetime.") from err
        for position in self.positions:
            position.set_evaluation_date(date)
        for fx_rate in self._fx_rates.values():
            fx_rate.set_evaluation_date(date)
        self._evaluation_date = date

    def set_fx_rate(self, fx_rate) -> None:
        """
        Register the FXRate used to convert a position currency into the
        portfolio's base currency.
        Args:
            fx_rate (FXRate): must have 'quote_currency' equal to the
                portfolio's 'base_currency'.
        """
        if not isinstance(fx_rate, FXRate):
            raise TypeError("'fx_rate' must be an FXRate instance.")
        if fx_rate.quote_currency != self.base_currency:
            raise ValueError(
                f"'fx_rate.quote_currency' must be the portfolio's base currency ({self.base_currency}).")
        self._fx_rates[fx_rate.base_currency] = fx_rate

    def _fx_rate_for(self, currency):
        if currency == self.base_currency:
            return None
        if currency not in self._fx_rates:
            raise ValueError(
                f"No FXRate registered to convert {currency} into {self.base_currency}. "
                f"Call 'set_fx_rate' first.")
        return self._fx_rates[currency]

    def _position_value_in_base(self, position) -> float:
        value = position.market_value
        fx_rate = self._fx_rate_for(position.currency)
        return fx_rate.convert(value) if fx_rate is not None else value


    def _historical_value(self):
        """
        Portfolio value in base currency at each historical date, using
        TODAY'S quantities applied retroactively (buy-and-hold historical
        simulation: 'what would today's book have been worth on each past
        day, holding it unchanged'). Missing observations (e.g. a market
        holiday on one exchange) are forward-filled before combining.
        """
        return self._historical_asset_value().sum(axis=1)

    def _historical_asset_value(self) -> pd.DataFrame:
        """
        Historical portfolio asset values in base currency.
        """
        end = self.evaluation_date
        series_list = []
        for position in self.positions:
            try:
                values = position.historical_prices.loc[:end] * position.quantity
            except ValueError:
                position.load_yahoo_history(self.lookback_window)
                values = position.historical_prices.loc[:end] * position.quantity
            fx_rate = self._fx_rate_for(position.currency)
            if fx_rate is not None:
                values = (values * fx_rate.historical_rates.loc[:end]).dropna()
            series_list.append(values.rename(position.ticker))
        combined = pd.concat(series_list, axis=1).sort_index().ffill().dropna()
        if combined.empty:
            raise ValueError("No overlapping historical price dates across positions (and FX rates).")
        return combined


    def _return_window(self, n_obs=None):
        returns = self.returns
        if n_obs is None:
            window = returns
        else:
            if n_obs < 2:
                raise ValueError(f"n_obs must be at least 2, got {n_obs}.")
            if returns.shape[0] < n_obs:
                raise ValueError(f"Requested n_obs={n_obs}, but only {returns.shape[0]} obs are available.")
            window = returns.iloc[-n_obs:]
        if window.shape[0] < 2:
            raise ValueError("At least 2 return observations are required to compute VaR.")
        return window

    def historical_var(self, confidence=0.95, n_obs=None) -> float:
        """
        Historical-simulation VaR: the confidence-level loss quantile of the
        portfolio's own historical return distribution, applied to today's
        market value.
        Args:
            confidence (float, default 0.95): confidence level (e.g. 0.95, 0.99).
            n_obs (int, default None): number of most recent return observations
                to use. If 'None', the full available history is used.
        Returns:
            float: VaR in the base currency (positive = loss).
        """
        window = self._return_window(n_obs)
        quantile = window.quantile(1 - confidence)
        return float(max(-quantile, 0.0) * self.market_value)

    def historical_cvar(self, confidence=0.95, n_obs=None) -> float:
        """
        Historical-simulation CVaR / Expected Shortfall: average loss beyond
        the VaR quantile, applied to today's market value.
        Args:
            confidence (float, default 0.95): confidence level (e.g. 0.95, 0.99).
            n_obs (int, default None): number of most recent return observations
                to use. If 'None', the full available history is used.
        Returns:
            float: CVaR in the base currency (positive = loss).
        """
        window = self._return_window(n_obs)
        quantile = window.quantile(1 - confidence)
        tail = window[window <= quantile]
        if tail.empty:
            tail = window.nsmallest(1)
        return float(max(-tail.mean(), 0.0) * self.market_value)

    def covariance_matrix(self, shrinkage=True) -> pd.DataFrame:
        """
        Calculate historical portfolio covariance matrix. Optionally, it applies Ledoit-Wolf shrinkage.
        Args:
            shrinkage (bool, default True): shrink portfolio covariance matrix.
        Returns:
            pandas.DataFrame: covariance matrix.
        """
        returns = self.asset_returns
        if shrinkage:
            lw  = LedoitWolf()
            lw.fit(returns)
            cov = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
            cov.attrs["shrinkage"] = float(lw.shrinkage_)
        else:
            cov = returns.cov()
            cov.attrs["shrinkage"] = 0.0
        cov.attrs["n_obs"] = len(returns)
        return cov

    def parametric_var(self, confidence=0.95) -> float:
        """
        Calculate portfolio parametric var.
        """
        return float(self.market_value * (norm.ppf(confidence) * self.volatility - self.returns.mean()))




