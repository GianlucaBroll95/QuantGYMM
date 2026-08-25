"""
Tests for the Stock equity instrument.

All tests are offline: history is a synthetic AAPL close series (first trading
days of 2023), so neither yfinance nor network access is required.
"""
import numpy as np
import pandas as pd
import pytest

from QuantGYMM.equity.instruments import Stock

# AAPL closes, first six trading days of 2023 (2023-01-07/08 is a weekend).
AAPL_DATES = pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05",
                             "2023-01-06", "2023-01-09", "2023-01-10"])
AAPL_CLOSES = [125.07, 126.36, 125.02, 129.62, 130.15, 130.73]


@pytest.fixture
def history():
    return pd.Series(AAPL_CLOSES, index=AAPL_DATES)


@pytest.fixture
def aapl(history):
    stock = Stock("AAPL", quantity=100, isin="US0378331005", currency="USD", dividend_yield=0.005)
    stock.set_evaluation_date("2023-01-10")
    stock.set_historical_prices(history)
    return stock


class TestConstruction:

    def test_defaults(self):
        stock = Stock("AAPL")
        assert stock.ticker == "AAPL"
        assert stock.quantity == 1
        assert stock.isin is None
        assert stock.currency == "EUR"
        assert stock.dividend_yield is None

    def test_explicit_attributes(self, aapl):
        assert aapl.ticker == "AAPL"
        assert aapl.quantity == 100
        assert aapl.isin == "US0378331005"
        assert aapl.currency == "USD"
        assert aapl.dividend_yield == 0.005

    def test_exported_at_top_level(self):
        from QuantGYMM import Stock as TopLevelStock
        assert TopLevelStock is Stock

    @pytest.mark.parametrize("ticker", [1, None, True, ["AAPL"]])
    def test_invalid_ticker(self, ticker):
        with pytest.raises(TypeError):
            Stock(ticker)

    @pytest.mark.parametrize("quantity", [0, -1, "2", None])
    def test_invalid_quantity(self, quantity):
        with pytest.raises(TypeError):
            Stock("AAPL", quantity=quantity)

    @pytest.mark.parametrize("dividend_yield", ["0.5%", ["0.005"]])
    def test_invalid_dividend_yield(self, dividend_yield):
        with pytest.raises(TypeError):
            Stock("AAPL", dividend_yield=dividend_yield)

    @pytest.mark.parametrize("isin", [1, False])
    def test_invalid_isin(self, isin):
        with pytest.raises(TypeError):
            Stock("AAPL", isin=isin)


class TestEvaluationDate:

    @pytest.mark.parametrize("date", ["2023-01-10", pd.Timestamp("2023-01-10")])
    def test_set_and_read(self, date):
        stock = Stock("AAPL")
        stock.set_evaluation_date(date)
        assert stock.evaluation_date == pd.Timestamp("2023-01-10")

    def test_unset_raises(self):
        with pytest.raises(ValueError, match="set_evaluation_date"):
            Stock("AAPL").evaluation_date

    def test_unparsable_date_raises(self):
        with pytest.raises(ValueError, match="Can't convert"):
            Stock("AAPL").set_evaluation_date("not a date")


class TestHistoricalPrices:

    def test_series_round_trip(self, aapl, history):
        pd.testing.assert_series_equal(aapl.historical_prices, history)

    def test_unset_raises(self):
        with pytest.raises(ValueError, match="set_historical_prices"):
            Stock("AAPL").historical_prices

    def test_single_column_dataframe_accepted(self, history):
        stock = Stock("AAPL")
        stock.set_historical_prices(history.to_frame("Close"))
        assert stock.historical_prices.tolist() == AAPL_CLOSES

    def test_multi_column_dataframe_rejected(self, history):
        frame = pd.DataFrame({"Close": history, "Open": history})
        with pytest.raises(ValueError, match="single"):
            Stock("AAPL").set_historical_prices(frame)

    @pytest.mark.parametrize("prices", [AAPL_CLOSES, np.array(AAPL_CLOSES), "125.07"])
    def test_non_pandas_rejected(self, prices):
        with pytest.raises(ValueError):
            Stock("AAPL").set_historical_prices(prices)

    def test_non_datetime_index_rejected(self):
        with pytest.raises(ValueError, match="DatetimeIndex"):
            Stock("AAPL").set_historical_prices(pd.Series(AAPL_CLOSES))

    @pytest.mark.parametrize("bad_price", [0.0, -1.0])
    def test_non_positive_prices_rejected(self, history, bad_price):
        corrupted = history.copy()
        corrupted.iloc[2] = bad_price
        with pytest.raises(ValueError, match="positive"):
            Stock("AAPL").set_historical_prices(corrupted)

    def test_nan_observations_dropped(self, history):
        with_nan = history.copy()
        with_nan.iloc[1] = np.nan
        stock = Stock("AAPL")
        stock.set_historical_prices(with_nan)
        assert len(stock.historical_prices) == len(history) - 1
        assert not stock.historical_prices.isna().any()

    def test_all_nan_rejected(self, history):
        with pytest.raises(ValueError, match="empty"):
            Stock("AAPL").set_historical_prices(pd.Series(np.nan, index=AAPL_DATES))

    def test_unsorted_input_is_sorted(self, history):
        stock = Stock("AAPL")
        stock.set_historical_prices(history.iloc[::-1])
        assert stock.historical_prices.index.is_monotonic_increasing
        pd.testing.assert_series_equal(stock.historical_prices, history)

    def test_timezone_aware_index_normalized(self, history):
        aware = history.copy()
        aware.index = aware.index.tz_localize("America/New_York")
        stock = Stock("AAPL")
        stock.set_historical_prices(aware)
        assert stock.historical_prices.index.tz is None

    def test_defensive_copy(self, history):
        stock = Stock("AAPL")
        stock.set_historical_prices(history)
        history.iloc[-1] = 1.0
        assert stock.historical_prices.iloc[-1] == AAPL_CLOSES[-1]


class TestDividends:

    def test_unset_raises(self):
        with pytest.raises(ValueError, match="set_dividend_history"):
            Stock("AAPL").dividends

    def test_valid_dividend_history(self):
        divs = pd.Series([0.0, 0.0, 0.0, 0.5, 0.0, 0.0], index=AAPL_DATES)
        stock = Stock("AAPL")
        stock.set_dividend_history(divs)
        assert stock.dividends.tolist() == divs.tolist()

    def test_empty_dividend_history_is_valid(self):
        stock = Stock("AAPL")
        stock.set_dividend_history(pd.Series([], index=pd.DatetimeIndex([]), dtype=float))
        assert stock.dividends.empty

    def test_negative_dividend_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            Stock("AAPL").set_dividend_history(pd.Series([-0.1], index=[AAPL_DATES[0]]))

    def test_non_datetime_index_rejected(self):
        with pytest.raises(ValueError, match="DatetimeIndex"):
            Stock("AAPL").set_dividend_history(pd.Series([0.5]))

    def test_timezone_aware_index_normalized(self):
        divs = pd.Series([0.5], index=pd.DatetimeIndex(["2023-01-06"]).tz_localize("America/New_York"))
        stock = Stock("AAPL")
        stock.set_dividend_history(divs)
        assert stock.dividends.index.tz is None


class TestPrice:

    def test_price_on_observation_date(self, aapl):
        assert aapl.price == 130.73

    def test_price_is_float(self, aapl):
        assert isinstance(aapl.price, float)

    def test_price_on_weekend_uses_last_close(self, aapl):
        aapl.set_evaluation_date("2023-01-07")  # Saturday
        assert aapl.price == 129.62  # Friday 2023-01-06 close

    def test_price_after_last_observation_uses_last_close(self, aapl):
        aapl.set_evaluation_date("2023-06-30")
        assert aapl.price == 130.73

    def test_price_before_first_observation_raises(self, aapl):
        aapl.set_evaluation_date("2022-12-30")
        with pytest.raises(ValueError, match="precedes the first"):
            aapl.price

    def test_price_without_history_raises(self):
        stock = Stock("AAPL")
        stock.set_evaluation_date("2023-01-10")
        with pytest.raises(ValueError, match="set_historical_prices"):
            stock.price

    def test_price_without_evaluation_date_raises(self, history):
        stock = Stock("AAPL")
        stock.set_historical_prices(history)
        with pytest.raises(ValueError, match="set_evaluation_date"):
            stock.price


class TestMarketValue:

    def test_market_value_is_price_times_quantity(self, aapl):
        assert aapl.market_value == pytest.approx(aapl.price * aapl.quantity)

    def test_market_value_scales_with_quantity(self, history):
        stock = Stock("AAPL", quantity=1)
        stock.set_evaluation_date("2023-01-10")
        stock.set_historical_prices(history)
        assert stock.market_value == pytest.approx(stock.price)

    def test_market_value_without_history_raises(self):
        stock = Stock("AAPL")
        stock.set_evaluation_date("2023-01-10")
        with pytest.raises(ValueError, match="set_historical_prices"):
            stock.market_value

    def test_market_value_without_evaluation_date_raises(self, history):
        stock = Stock("AAPL")
        stock.set_historical_prices(history)
        with pytest.raises(ValueError, match="set_evaluation_date"):
            stock.market_value


class TestPriceReturns:

    def test_log_returns_match_manual_calc(self, aapl, history):
        expected = np.log(history / history.shift(1)).dropna()
        pd.testing.assert_series_equal(aapl.log_returns, expected)

    def test_simple_returns_match_manual_calc(self, aapl, history):
        expected = history.pct_change().dropna()
        pd.testing.assert_series_equal(aapl.simple_returns, expected)

    def test_simple_returns_aggregate_exactly_across_positions(self, history):
        """The whole reason 'simple_returns' exists alongside 'log_returns':
        a weighted sum of SIMPLE returns (weighted by start-of-period value)
        reproduces the level-based return of a combined position exactly; log
        returns do not (ln doesn't distribute over a sum), which is why
        portfolio-level aggregation must use these, not log_returns."""
        prices_a = history
        prices_b = history * 2 - 10  # a different price path
        combined = prices_a + prices_b
        combined_simple_return = combined.pct_change().dropna()

        start_of_period_weight_a = (prices_a / combined).shift(1)
        start_of_period_weight_b = (prices_b / combined).shift(1)
        weighted_simple = (start_of_period_weight_a * prices_a.pct_change()
                          + start_of_period_weight_b * prices_b.pct_change()).dropna()
        assert np.allclose(weighted_simple.to_numpy(), combined_simple_return.to_numpy())

    def test_log_returns_is_float_series(self, aapl):
        assert aapl.log_returns.dtype == float

    def test_without_history_raises(self):
        stock = Stock("AAPL")
        stock.set_evaluation_date("2023-01-10")
        with pytest.raises(ValueError, match="set_historical_prices"):
            stock.log_returns
        with pytest.raises(ValueError, match="set_historical_prices"):
            stock.simple_returns


class TestTotalReturns:

    @pytest.fixture
    def aapl_with_dividend(self, aapl, history):
        divs = pd.Series(0.0, index=history.index)
        divs.loc["2023-01-06"] = 0.50
        aapl.set_dividend_history(divs)
        return aapl

    def test_log_total_returns_matches_manual_calc_on_ex_div_date(self, aapl_with_dividend):
        expected = np.log((129.62 + 0.50) / 125.02)
        assert aapl_with_dividend.log_total_returns.loc["2023-01-06"] == pytest.approx(expected)

    def test_simple_total_returns_matches_manual_calc_on_ex_div_date(self, aapl_with_dividend):
        expected = (129.62 + 0.50) / 125.02 - 1
        assert aapl_with_dividend.simple_total_returns.loc["2023-01-06"] == pytest.approx(expected)

    def test_log_total_returns_exceeds_log_returns_on_ex_div_date(self, aapl_with_dividend):
        assert aapl_with_dividend.log_total_returns.loc["2023-01-06"] > aapl_with_dividend.log_returns.loc["2023-01-06"]

    def test_log_total_returns_equals_log_returns_on_non_ex_div_dates(self, aapl_with_dividend):
        pd.testing.assert_series_equal(
            aapl_with_dividend.log_total_returns.drop("2023-01-06"),
            aapl_with_dividend.log_returns.drop("2023-01-06"),
        )

    def test_off_calendar_dividend_is_ignored(self, history):
        stock = Stock("AAPL")
        stock.set_evaluation_date("2023-01-10")
        stock.set_historical_prices(history)
        divs = pd.Series([1.0], index=pd.to_datetime(["2023-01-07"]))  # Saturday, not a price date
        stock.set_dividend_history(divs)
        pd.testing.assert_series_equal(stock.log_total_returns, stock.log_returns)

    def test_respects_evaluation_date_no_lookahead(self, history):
        stock = Stock("AAPL")
        stock.set_evaluation_date("2023-01-06")
        stock.set_historical_prices(history)
        divs = pd.Series(0.0, index=history.index)
        divs.loc["2023-01-09"] = 100.0  # large dividend after the evaluation date
        stock.set_dividend_history(divs)
        pd.testing.assert_series_equal(stock.log_total_returns, stock.log_returns.loc[:"2023-01-06"])

    def test_without_dividends_raises(self, aapl):
        with pytest.raises(ValueError, match="set_dividend_history"):
            aapl.log_total_returns
        with pytest.raises(ValueError, match="set_dividend_history"):
            aapl.simple_total_returns


class TestRealizedVolatility:

    def test_full_history_matches_log_return_std(self, aapl, history):
        expected = np.log(history / history.shift(1)).std(ddof=1)
        assert aapl.realized_volatility() == pytest.approx(expected)

    def test_total_return_flag_uses_log_total_returns(self, history):
        stock = Stock("AAPL")
        stock.set_evaluation_date("2023-01-10")
        stock.set_historical_prices(history)
        divs = pd.Series(0.0, index=history.index)
        divs.loc["2023-01-06"] = 0.50
        stock.set_dividend_history(divs)
        expected = stock.log_total_returns.std(ddof=1)
        assert stock.realized_volatility(total_return=True) == pytest.approx(expected)
        assert stock.realized_volatility(total_return=True) != pytest.approx(stock.realized_volatility())

    def test_total_return_flag_without_dividends_raises(self, aapl):
        with pytest.raises(ValueError, match="set_dividend_history"):
            aapl.realized_volatility(total_return=True)

    def test_n_obs_window_matches_trailing_log_return_std(self, aapl, history):
        expected = np.log(history / history.shift(1)).iloc[-3:].std(ddof=1)
        assert aapl.realized_volatility(n_obs=3) == pytest.approx(expected)

    def test_respects_evaluation_date_no_lookahead(self, history):
        stock = Stock("AAPL")
        stock.set_evaluation_date("2023-01-06")
        stock.set_historical_prices(history)
        past = history.loc[:"2023-01-06"]
        expected = np.log(past / past.shift(1)).std(ddof=1)
        assert stock.realized_volatility() == pytest.approx(expected)

    def test_n_obs_larger_than_available_history_raises(self, aapl):
        with pytest.raises(ValueError, match="obs are available"):
            aapl.realized_volatility(n_obs=100)

    @pytest.mark.parametrize("n_obs", [0, 1])
    def test_n_obs_below_minimum_raises(self, aapl, n_obs):
        with pytest.raises(ValueError, match="at least 2"):
            aapl.realized_volatility(n_obs=n_obs)

    def test_without_historical_prices_raises(self):
        stock = Stock("AAPL")
        stock.set_evaluation_date("2023-01-10")
        with pytest.raises(ValueError, match="set_historical_prices"):
            stock.realized_volatility()

    def test_without_evaluation_date_raises(self, history):
        stock = Stock("AAPL")
        stock.set_historical_prices(history)
        with pytest.raises(ValueError, match="set_evaluation_date"):
            stock.realized_volatility()


class TestLoadYahooHistoryMocked:
    """Offline tests of the Yahoo adapter: yfinance is replaced by a fake module."""

    @staticmethod
    def _install_fake_yfinance(monkeypatch, frame, capture=None):
        import sys
        import types

        class FakeTicker:
            def __init__(self, symbol):
                if capture is not None:
                    capture["symbol"] = symbol

            def history(self, start=None, end=None, auto_adjust=None):
                if capture is not None:
                    capture["start"] = start
                    capture["end"] = end
                    capture["auto_adjust"] = auto_adjust
                return frame

        monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=FakeTicker))

    @pytest.fixture
    def yahoo_frame(self):
        # yfinance returns a tz-aware index and OHLC + Dividends/Stock Splits columns
        index = pd.date_range("2023-01-02", periods=5, freq="B", tz="Europe/Rome")
        return pd.DataFrame({"Open": [13.1, 13.3, 13.2, 13.6, 13.5],
                             "Close": [13.2, 13.4, 13.1, 13.7, 13.6],
                             "Dividends": [0.0, 0.0, 0.30, 0.0, 0.0]}, index=index)

    def test_requires_evaluation_date_first(self, monkeypatch, yahoo_frame):
        self._install_fake_yfinance(monkeypatch, yahoo_frame)
        with pytest.raises(ValueError, match="set_evaluation_date"):
            Stock("UCG.MI").load_yahoo_history()

    def test_history_stored_from_close_column(self, monkeypatch, yahoo_frame):
        self._install_fake_yfinance(monkeypatch, yahoo_frame)
        stock = Stock("UCG.MI")
        stock.set_evaluation_date("2023-01-10")
        stock.load_yahoo_history()
        assert stock.historical_prices.tolist() == [13.2, 13.4, 13.1, 13.7, 13.6]
        assert stock.historical_prices.index.tz is None

    def test_dividends_stored_from_dividends_column(self, monkeypatch, yahoo_frame):
        self._install_fake_yfinance(monkeypatch, yahoo_frame)
        stock = Stock("UCG.MI")
        stock.set_evaluation_date("2023-01-10")
        stock.load_yahoo_history()
        assert stock.dividends.tolist() == [0.0, 0.0, 0.30, 0.0, 0.0]
        assert stock.dividends.index.tz is None

    def test_requests_window_anchored_on_evaluation_date(self, monkeypatch, yahoo_frame):
        capture = {}
        self._install_fake_yfinance(monkeypatch, yahoo_frame, capture)
        stock = Stock("UCG.MI")
        stock.set_evaluation_date("2023-01-10")
        stock.load_yahoo_history(lookback=pd.DateOffset(months=1))
        assert capture == {
            "symbol": "UCG.MI",
            "start": pd.Timestamp("2023-01-10") - pd.DateOffset(months=1),
            "end": pd.Timestamp("2023-01-11"),  # evaluation date + 1 day: yfinance's 'end' is exclusive
            "auto_adjust": False,
        }

    def test_empty_download_raises(self, monkeypatch):
        self._install_fake_yfinance(monkeypatch, pd.DataFrame())
        stock = Stock("UCG.MI")
        stock.set_evaluation_date("2023-01-10")
        with pytest.raises(ValueError, match="UCG.MI"):
            stock.load_yahoo_history()


@pytest.mark.network
class TestLoadYahooHistoryLive:
    """Live download for UCG.MI (UniCredit, Borsa Italiana). Requires internet;
    deselect with: pytest -m "not network"."""

    def test_load_ucg_mi_history(self):
        pytest.importorskip("yfinance")
        stock = Stock("UCG.MI", currency="EUR")
        stock.set_evaluation_date(pd.Timestamp.today())
        stock.load_yahoo_history(lookback=pd.DateOffset(months=1))
        history = stock.historical_prices
        assert len(history) > 5
        assert history.index.tz is None
        assert history.index.is_monotonic_increasing
        assert (history > 0).all()

    def test_price_at_today_equals_last_close(self):
        pytest.importorskip("yfinance")
        stock = Stock("UCG.MI")
        stock.set_evaluation_date(pd.Timestamp.today())
        stock.load_yahoo_history(lookback=pd.DateOffset(months=1))
        assert stock.price == pytest.approx(stock.historical_prices.iloc[-1])

    def test_unknown_ticker_raises(self):
        pytest.importorskip("yfinance")
        stock = Stock("THISTICKERDOESNOTEXIST")
        stock.set_evaluation_date(pd.Timestamp.today())
        with pytest.raises(ValueError, match="THISTICKERDOESNOTEXIST"):
            stock.load_yahoo_history(lookback=pd.DateOffset(months=1))
