"""
Tests for EquityPortfolio.

All tests are offline: prices are synthetic series over ten business days, so
neither yfinance nor network access is required.
"""
import numpy as np
import pandas as pd
import pytest

from QuantGYMM.equity.instruments import Stock
from QuantGYMM.equity.portfolio import EquityPortfolio
from QuantGYMM.fx import FXRate

DATES = pd.date_range("2023-01-02", periods=10, freq="B")
EVAL_DATE = DATES[-1]

A_CLOSES = [10.0, 10.5, 10.2, 10.8, 10.6, 10.9, 10.3, 10.7, 11.0, 11.2]
B_CLOSES = [100.0, 99.0, 101.0, 98.5, 97.0, 99.5, 96.0, 98.0, 97.5, 99.0]


@pytest.fixture
def stock_a():
    a = Stock("A", quantity=10, currency="EUR")
    a.set_evaluation_date(EVAL_DATE)
    a.set_historical_prices(pd.Series(A_CLOSES, index=DATES))
    return a


@pytest.fixture
def stock_b():
    b = Stock("B", quantity=5, currency="EUR")
    b.set_evaluation_date(EVAL_DATE)
    b.set_historical_prices(pd.Series(B_CLOSES, index=DATES))
    return b


@pytest.fixture
def portfolio(stock_a, stock_b):
    port = EquityPortfolio([stock_a, stock_b])
    port.set_evaluation_date(EVAL_DATE)
    return port


@pytest.fixture
def expected_value():
    return pd.Series(A_CLOSES, index=DATES) * 10 + pd.Series(B_CLOSES, index=DATES) * 5


class TestConstruction:

    def test_positions_stored(self, portfolio, stock_a, stock_b):
        assert portfolio.positions == [stock_a, stock_b]

    def test_default_base_currency_is_eur(self, portfolio):
        assert portfolio.base_currency == "EUR"

    def test_custom_base_currency(self, stock_a, stock_b):
        port = EquityPortfolio([stock_a, stock_b], base_currency="USD")
        assert port.base_currency == "USD"

    def test_empty_positions_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            EquityPortfolio([])

    def test_non_stock_positions_rejected(self, stock_a):
        with pytest.raises(TypeError, match="Stock"):
            EquityPortfolio([stock_a, "not a stock"])

    def test_currencies_excludes_base(self, stock_a, stock_b):
        port = EquityPortfolio([stock_a, stock_b])
        assert port.currencies == []

    def test_currencies_includes_foreign(self, stock_a):
        usd_stock = Stock("C", currency="USD")
        port = EquityPortfolio([stock_a, usd_stock])
        assert port.currencies == ["USD"]


class TestEvaluationDate:

    def test_unset_raises(self, stock_a, stock_b):
        port = EquityPortfolio([stock_a, stock_b])
        with pytest.raises(ValueError, match="set_evaluation_date"):
            port.evaluation_date

    def test_propagates_to_positions(self, stock_a, stock_b):
        port = EquityPortfolio([stock_a, stock_b])
        port.set_evaluation_date("2023-01-10")
        assert stock_a.evaluation_date == pd.Timestamp("2023-01-10")
        assert stock_b.evaluation_date == pd.Timestamp("2023-01-10")

    def test_propagates_to_registered_fx_rates(self, stock_a):
        usd_stock = Stock("C", currency="USD")
        port = EquityPortfolio([stock_a, usd_stock])
        fx = FXRate("USD", "EUR")
        fx.set_historical_rates(pd.Series([0.9] * len(DATES), index=DATES))
        port.set_fx_rate(fx)
        port.set_evaluation_date("2023-01-10")
        assert fx.evaluation_date == pd.Timestamp("2023-01-10")

    def test_unparsable_date_raises(self, stock_a, stock_b):
        port = EquityPortfolio([stock_a, stock_b])
        with pytest.raises(ValueError, match="Can't convert"):
            port.set_evaluation_date("not a date")


class TestMarketValueAndWeights:

    def test_market_value_matches_manual_calc(self, portfolio):
        assert portfolio.market_value == pytest.approx(10 * 11.2 + 5 * 99.0)

    def test_market_value_without_evaluation_date_raises(self, stock_a, stock_b):
        port = EquityPortfolio([stock_a, stock_b])
        with pytest.raises(ValueError, match="set_evaluation_date"):
            port.market_value

    def test_weights_sum_to_one(self, portfolio):
        assert sum(portfolio.weights.values()) == pytest.approx(1.0)

    def test_weights_proportional_to_value(self, portfolio, stock_a, stock_b):
        expected_a = (10 * 11.2) / (10 * 11.2 + 5 * 99.0)
        assert portfolio.weights["A"] == pytest.approx(expected_a)


class TestForeignCurrency:

    @pytest.fixture
    def usd_stock(self):
        c = Stock("C", quantity=100, currency="USD")
        c.set_evaluation_date(EVAL_DATE)
        c.set_historical_prices(pd.Series([5.0] * len(DATES), index=DATES))
        return c

    def test_market_value_without_fx_rate_raises(self, stock_a, usd_stock):
        port = EquityPortfolio([stock_a, usd_stock])
        port.set_evaluation_date(EVAL_DATE)
        with pytest.raises(ValueError, match="No FXRate registered"):
            port.market_value

    def test_set_fx_rate_wrong_quote_currency_rejected(self, stock_a, usd_stock):
        port = EquityPortfolio([stock_a, usd_stock])
        fx = FXRate("USD", "GBP")
        fx.set_historical_rates(pd.Series([0.8] * len(DATES), index=DATES))
        with pytest.raises(ValueError, match="base currency"):
            port.set_fx_rate(fx)

    def test_market_value_converts_via_fx_rate(self, stock_a, usd_stock):
        port = EquityPortfolio([stock_a, usd_stock])
        fx = FXRate("USD", "EUR")
        fx.set_historical_rates(pd.Series([0.9] * len(DATES), index=DATES))
        port.set_fx_rate(fx)
        port.set_evaluation_date(EVAL_DATE)
        expected = 10 * 11.2 + 100 * 5.0 * 0.9
        assert port.market_value == pytest.approx(expected)


class TestReturns:

    def test_matches_manual_pct_change(self, portfolio, expected_value):
        expected = expected_value.pct_change().dropna()
        pd.testing.assert_series_equal(portfolio.returns, expected)

    def test_respects_evaluation_date_no_lookahead(self, stock_a, stock_b):
        port = EquityPortfolio([stock_a, stock_b])
        port.set_evaluation_date(DATES[4])
        past = pd.Series(A_CLOSES, index=DATES).loc[:DATES[4]] * 10 \
            + pd.Series(B_CLOSES, index=DATES).loc[:DATES[4]] * 5
        pd.testing.assert_series_equal(port.returns, past.pct_change().dropna())

    def test_mismatched_calendars_are_forward_filled(self, stock_b):
        sparse_dates = DATES[[0, 2, 5, 9]]
        sparse = Stock("D", quantity=1, currency="EUR")
        sparse.set_evaluation_date(EVAL_DATE)
        sparse.set_historical_prices(pd.Series([50.0, 52.0, 51.0, 55.0], index=sparse_dates))
        port = EquityPortfolio([stock_b, sparse])
        port.set_evaluation_date(EVAL_DATE)
        # every date from the denser series should survive (forward-filled), not just the 4 common ones
        assert len(port.returns) == len(DATES) - 1

    def test_no_overlapping_dates_raises(self):
        a = Stock("A", currency="EUR")
        a.set_evaluation_date("2023-06-01")
        a.set_historical_prices(pd.Series([10.0, 11.0], index=pd.to_datetime(["2023-01-01", "2023-01-02"])))
        b = Stock("B", currency="EUR")
        b.set_evaluation_date("2023-06-01")
        b.set_historical_prices(pd.Series([20.0, 21.0], index=pd.to_datetime(["2024-01-01", "2024-01-02"])))
        port = EquityPortfolio([a, b])
        port.set_evaluation_date("2023-06-01")
        with pytest.raises(ValueError, match="No overlapping"):
            port.returns


class TestHistoricalVaR:

    def test_var_and_cvar_non_negative(self, portfolio):
        assert portfolio.historical_var(0.95) >= 0.0
        assert portfolio.historical_cvar(0.95) >= 0.0

    def test_cvar_at_least_var(self, portfolio):
        assert portfolio.historical_cvar(0.95) >= portfolio.historical_var(0.95)

    def test_matches_manual_quantile_calc(self, portfolio, expected_value):
        returns = expected_value.pct_change().dropna()
        quantile = returns.quantile(0.05)
        expected_var = max(-quantile, 0.0) * portfolio.market_value
        assert portfolio.historical_var(0.95) == pytest.approx(expected_var)

    def test_higher_confidence_gives_larger_or_equal_var(self, portfolio):
        assert portfolio.historical_var(0.99) >= portfolio.historical_var(0.95)

    def test_n_obs_window(self, portfolio):
        expected = portfolio.returns.iloc[-3:].quantile(0.05)
        expected_var = max(-expected, 0.0) * portfolio.market_value
        assert portfolio.historical_var(0.95, n_obs=3) == pytest.approx(expected_var)

    def test_n_obs_larger_than_available_raises(self, portfolio):
        with pytest.raises(ValueError, match="obs are available"):
            portfolio.historical_var(0.95, n_obs=100)

    @pytest.mark.parametrize("n_obs", [0, 1])
    def test_n_obs_below_minimum_raises(self, portfolio, n_obs):
        with pytest.raises(ValueError, match="at least 2"):
            portfolio.historical_var(0.95, n_obs=n_obs)

    def test_var_without_evaluation_date_raises(self, stock_a, stock_b):
        port = EquityPortfolio([stock_a, stock_b])
        with pytest.raises(ValueError, match="set_evaluation_date"):
            port.historical_var(0.95)
