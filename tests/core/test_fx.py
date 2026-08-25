"""
Tests for the FXRate shared market-data object.

All tests are offline: rates are a synthetic EURUSD-style series (first trading
days of 2023), so neither yfinance nor network access is required.
"""
import numpy as np
import pandas as pd
import pytest

from QuantGYMM.fx import FXRate

# Synthetic USD-per-EUR rates, first six trading days of 2023.
FX_DATES = pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05",
                           "2023-01-06", "2023-01-09", "2023-01-10"])
FX_RATES = [1.0555, 1.0602, 1.0498, 1.0623, 1.0644, 1.0752]


@pytest.fixture
def rates():
    return pd.Series(FX_RATES, index=FX_DATES)


@pytest.fixture
def eurusd(rates):
    fx = FXRate("EUR", "USD")
    fx.set_evaluation_date("2023-01-10")
    fx.set_historical_rates(rates)
    return fx


class TestConstruction:

    def test_attributes(self, eurusd):
        assert eurusd.base_currency == "EUR"
        assert eurusd.quote_currency == "USD"

    def test_ticker(self, eurusd):
        assert eurusd.ticker == "EURUSD=X"

    def test_exported_at_top_level(self):
        from QuantGYMM import FXRate as TopLevelFXRate
        assert TopLevelFXRate is FXRate

    @pytest.mark.parametrize("currency", [1, None, True, ["EUR"]])
    def test_invalid_base_currency(self, currency):
        with pytest.raises(TypeError):
            FXRate(currency, "USD")

    @pytest.mark.parametrize("currency", [1, None, True, ["USD"]])
    def test_invalid_quote_currency(self, currency):
        with pytest.raises(TypeError):
            FXRate("EUR", currency)


class TestEvaluationDate:

    @pytest.mark.parametrize("date", ["2023-01-10", pd.Timestamp("2023-01-10")])
    def test_set_and_read(self, date):
        fx = FXRate("EUR", "USD")
        fx.set_evaluation_date(date)
        assert fx.evaluation_date == pd.Timestamp("2023-01-10")

    def test_unset_raises(self):
        with pytest.raises(ValueError, match="set_evaluation_date"):
            FXRate("EUR", "USD").evaluation_date

    def test_unparsable_date_raises(self):
        with pytest.raises(ValueError, match="Can't convert"):
            FXRate("EUR", "USD").set_evaluation_date("not a date")


class TestHistoricalRates:

    def test_series_round_trip(self, eurusd, rates):
        pd.testing.assert_series_equal(eurusd.historical_rates, rates)

    def test_unset_raises(self):
        with pytest.raises(ValueError, match="set_historical_rates"):
            FXRate("EUR", "USD").historical_rates

    def test_non_datetime_index_rejected(self):
        with pytest.raises(ValueError, match="DatetimeIndex"):
            FXRate("EUR", "USD").set_historical_rates(pd.Series(FX_RATES))

    @pytest.mark.parametrize("bad_rate", [0.0, -1.0])
    def test_non_positive_rates_rejected(self, rates, bad_rate):
        corrupted = rates.copy()
        corrupted.iloc[2] = bad_rate
        with pytest.raises(ValueError, match="positive"):
            FXRate("EUR", "USD").set_historical_rates(corrupted)

    def test_all_nan_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            FXRate("EUR", "USD").set_historical_rates(pd.Series(np.nan, index=FX_DATES))

    def test_timezone_aware_index_normalized(self, rates):
        aware = rates.copy()
        aware.index = aware.index.tz_localize("America/New_York")
        fx = FXRate("EUR", "USD")
        fx.set_historical_rates(aware)
        assert fx.historical_rates.index.tz is None


class TestRate:

    def test_rate_on_observation_date(self, eurusd):
        assert eurusd.rate == 1.0752

    def test_rate_is_float(self, eurusd):
        assert isinstance(eurusd.rate, float)

    def test_rate_on_weekend_uses_last_close(self, eurusd):
        eurusd.set_evaluation_date("2023-01-07")  # Saturday
        assert eurusd.rate == 1.0623  # Friday 2023-01-06 rate

    def test_rate_before_first_observation_raises(self, eurusd):
        eurusd.set_evaluation_date("2022-12-30")
        with pytest.raises(ValueError, match="precedes the first"):
            eurusd.rate

    def test_rate_without_history_raises(self):
        fx = FXRate("EUR", "USD")
        fx.set_evaluation_date("2023-01-10")
        with pytest.raises(ValueError, match="set_historical_rates"):
            fx.rate

    def test_rate_without_evaluation_date_raises(self, rates):
        fx = FXRate("EUR", "USD")
        fx.set_historical_rates(rates)
        with pytest.raises(ValueError, match="set_evaluation_date"):
            fx.rate


class TestConvert:

    def test_convert_multiplies_by_rate(self, eurusd):
        assert eurusd.convert(1000) == pytest.approx(1000 * 1.0752)

    def test_convert_zero_is_zero(self, eurusd):
        assert eurusd.convert(0) == 0.0

    def test_inverse_pair_round_trips(self, rates):
        eur_usd = FXRate("EUR", "USD")
        eur_usd.set_evaluation_date("2023-01-10")
        eur_usd.set_historical_rates(rates)

        usd_eur = FXRate("USD", "EUR")
        usd_eur.set_evaluation_date("2023-01-10")
        usd_eur.set_historical_rates(1 / rates)

        amount_eur = 1000
        amount_usd = eur_usd.convert(amount_eur)
        assert usd_eur.convert(amount_usd) == pytest.approx(amount_eur)


class TestReturns:

    def test_full_history_matches_log_return(self, eurusd, rates):
        expected = np.log(rates / rates.shift(1)).dropna()
        pd.testing.assert_series_equal(eurusd.returns, expected)

    def test_respects_evaluation_date_no_lookahead(self, rates):
        fx = FXRate("EUR", "USD")
        fx.set_evaluation_date("2023-01-06")
        fx.set_historical_rates(rates)
        past = rates.loc[:"2023-01-06"]
        expected = np.log(past / past.shift(1)).dropna()
        pd.testing.assert_series_equal(fx.returns, expected)


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
        index = pd.date_range("2023-01-02", periods=5, freq="B", tz="UTC")
        return pd.DataFrame({"Open": [1.05, 1.06, 1.05, 1.06, 1.07],
                             "Close": [1.06, 1.05, 1.06, 1.07, 1.08]}, index=index)

    def test_requires_evaluation_date_first(self, monkeypatch, yahoo_frame):
        self._install_fake_yfinance(monkeypatch, yahoo_frame)
        with pytest.raises(ValueError, match="set_evaluation_date"):
            FXRate("EUR", "USD").load_yahoo_history()

    def test_rates_stored_from_close_column(self, monkeypatch, yahoo_frame):
        self._install_fake_yfinance(monkeypatch, yahoo_frame)
        fx = FXRate("EUR", "USD")
        fx.set_evaluation_date("2023-01-10")
        fx.load_yahoo_history()
        assert fx.historical_rates.tolist() == [1.06, 1.05, 1.06, 1.07, 1.08]
        assert fx.historical_rates.index.tz is None

    def test_requests_window_anchored_on_evaluation_date(self, monkeypatch, yahoo_frame):
        capture = {}
        self._install_fake_yfinance(monkeypatch, yahoo_frame, capture)
        fx = FXRate("EUR", "USD")
        fx.set_evaluation_date("2023-01-10")
        fx.load_yahoo_history(lookback=pd.DateOffset(months=1))
        assert capture == {
            "symbol": "EURUSD=X",
            "start": pd.Timestamp("2023-01-10") - pd.DateOffset(months=1),
            "end": pd.Timestamp("2023-01-11"),  # evaluation date + 1 day: yfinance's 'end' is exclusive
            "auto_adjust": False,
        }

    def test_empty_download_raises(self, monkeypatch):
        self._install_fake_yfinance(monkeypatch, pd.DataFrame())
        fx = FXRate("EUR", "USD")
        fx.set_evaluation_date("2023-01-10")
        with pytest.raises(ValueError, match="EURUSD=X"):
            fx.load_yahoo_history()


@pytest.mark.network
class TestLoadYahooHistoryLive:
    """Live download for EURUSD=X. Requires internet; deselect with: pytest -m "not network"."""

    def test_load_eurusd_history(self):
        pytest.importorskip("yfinance")
        fx = FXRate("EUR", "USD")
        fx.set_evaluation_date(pd.Timestamp.today())
        fx.load_yahoo_history(lookback=pd.DateOffset(months=1))
        rates = fx.historical_rates
        assert len(rates) > 5
        assert rates.index.tz is None
        assert rates.index.is_monotonic_increasing
        assert (rates > 0).all()

    def test_same_currency_pair_rate_is_one(self):
        pytest.importorskip("yfinance")
        fx = FXRate("EUR", "EUR")
        fx.set_evaluation_date(pd.Timestamp.today())
        fx.load_yahoo_history(lookback=pd.DateOffset(months=1))
        assert fx.rate == pytest.approx(1.0)

    def test_inverse_pair_round_trips_live(self):
        pytest.importorskip("yfinance")
        eur_usd = FXRate("EUR", "USD")
        eur_usd.set_evaluation_date(pd.Timestamp.today())
        eur_usd.load_yahoo_history(lookback=pd.DateOffset(months=1))

        usd_eur = FXRate("USD", "EUR")
        usd_eur.set_evaluation_date(pd.Timestamp.today())
        usd_eur.load_yahoo_history(lookback=pd.DateOffset(months=1))

        assert eur_usd.rate * usd_eur.rate == pytest.approx(1.0, abs=1e-3)
