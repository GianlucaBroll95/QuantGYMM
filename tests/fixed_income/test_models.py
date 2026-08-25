"""
Tests for models.py: Hull-White helpers (hw_b, hw_zcb_option) and MertonSimulator.
"""
import numpy as np
import pandas as pd
import pytest
from pandas.tseries.offsets import DateOffset

from QuantGYMM.fixed_income.models import MertonSimulator, hw_b, hw_zcb_option
from QuantGYMM.utils import accrual_factor


# ---------------------------------------------------------------------------
# hw_b
# ---------------------------------------------------------------------------

class TestHwB:

    def test_closed_form(self):
        a, T = 0.03, 5.0
        assert abs(hw_b(0.0, T, a) - (1 - np.exp(-a * T)) / a) < 1e-12

    def test_zero_interval(self):
        assert hw_b(2.0, 2.0, 0.05) == 0.0

    def test_small_mean_reversion_limit(self):
        # a → 0: B(t, T) → T − t
        assert abs(hw_b(1.0, 4.0, 1e-8) - 3.0) < 1e-4

    def test_vectorized_target(self):
        out = hw_b(1.0, np.array([2.0, 3.0, 5.0]), 0.05)
        assert out.shape == (3,)
        assert np.all(np.diff(out) > 0)   # increasing in T


# ---------------------------------------------------------------------------
# hw_zcb_option
# ---------------------------------------------------------------------------

class TestHwZcbOption:

    A, SIGMA = 0.03, 0.01
    T, S = 1.0, 5.0
    R = 0.03

    @property
    def df_T(self):
        return np.exp(-self.R * self.T)

    @property
    def df_S(self):
        return np.exp(-self.R * self.S)

    def test_put_call_parity(self):
        K = 0.9
        call = hw_zcb_option(self.df_T, self.df_S, K, self.T, self.S, self.A, self.SIGMA, kind="call")
        put  = hw_zcb_option(self.df_T, self.df_S, K, self.T, self.S, self.A, self.SIGMA, kind="put")
        assert abs((call - put) - (self.df_S - K * self.df_T)) < 1e-12

    def test_zero_volatility_gives_intrinsic(self):
        K = 0.8
        call = hw_zcb_option(self.df_T, self.df_S, K, self.T, self.S, self.A, 0.0, kind="call")
        assert abs(call - max(self.df_S - K * self.df_T, 0.0)) < 1e-12
        put = hw_zcb_option(self.df_T, self.df_S, 1.1, self.T, self.S, self.A, 0.0, kind="put")
        assert abs(put - max(1.1 * self.df_T - self.df_S, 0.0)) < 1e-12

    def test_option_value_positive_and_bounded(self):
        K = 0.88
        call = hw_zcb_option(self.df_T, self.df_S, K, self.T, self.S, self.A, self.SIGMA, kind="call")
        assert 0.0 < call < self.df_S

    def test_call_decreasing_in_strike(self):
        prices = [hw_zcb_option(self.df_T, self.df_S, k, self.T, self.S, self.A, self.SIGMA)
                  for k in (0.80, 0.85, 0.90, 0.95)]
        assert all(prices[i] > prices[i + 1] for i in range(len(prices) - 1))

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError):
            hw_zcb_option(self.df_T, self.df_S, 0.9, self.T, self.S, self.A, self.SIGMA, kind="straddle")


# ---------------------------------------------------------------------------
# MertonSimulator
# ---------------------------------------------------------------------------

START = pd.Timestamp("2023-01-02")
NODES = pd.DatetimeIndex([START + DateOffset(years=y) for y in range(1, 9)])

R_TRUE, MU_TRUE, S_TRUE = 0.02, 0.002, 0.0001


def _merton_spot_rates() -> pd.DataFrame:
    """Exact Merton term structure R(t) = r + mu·t/2 − s·t²/6 on NODES."""
    ttm = accrual_factor("ACT/365", START, NODES)
    rates = R_TRUE + MU_TRUE * ttm / 2 - S_TRUE * ttm ** 2 / 6
    return pd.DataFrame({"spotRate": rates}, index=NODES)


class TestMertonCalibration:

    def test_risk_neutral_recovers_parameters(self):
        sim = MertonSimulator(start_date=START, nsim=100, seed=42)
        sim.risk_neutral_calibration(_merton_spot_rates())
        assert abs(sim.r - R_TRUE) < 1e-10
        assert abs(sim.mu - MU_TRUE) < 1e-10
        assert abs(sim.s - S_TRUE) < 1e-10

    def test_negative_sigma_calibration_fails(self):
        # Convex-up term structure implies a negative fitted variance → error.
        ttm = accrual_factor("ACT/365", START, NODES)
        rates = R_TRUE + MU_TRUE * ttm / 2 + S_TRUE * ttm ** 2 / 6
        bad = pd.DataFrame({"spotRate": rates}, index=NODES)
        sim = MertonSimulator(start_date=START, nsim=10, seed=1)
        with pytest.raises(ValueError):
            sim.risk_neutral_calibration(bad)

    def test_properties_raise_before_calibration(self):
        sim = MertonSimulator(start_date=START, nsim=10)
        for attr in ("ttm", "r", "mu", "s"):
            with pytest.raises(ValueError):
                getattr(sim, attr)

    def test_real_world_calibration(self):
        # The _spot_rates descriptor requires a DataFrame (a bare Series is rejected).
        idx = pd.date_range("2023-01-02", periods=500, freq="D")
        rng = np.random.default_rng(7)
        proxy = pd.DataFrame(
            {"shortRate": 0.02 + np.cumsum(rng.normal(0, 1e-4, len(idx)))}, index=idx
        )
        sim = MertonSimulator(start_date=idx[-1], nsim=10, seed=0)
        sim.real_world_calibration(proxy, ttm=list(NODES))
        assert sim.r == proxy.iloc[-1].item()
        assert abs(sim.mu - proxy.diff().mean().item() * 252) < 1e-12
        assert abs(sim.s - proxy.diff().var().item() * 252) < 1e-12

    def test_real_world_non_daily_warns(self):
        idx = pd.date_range("2023-01-02", periods=50, freq="W")
        proxy = pd.DataFrame({"shortRate": np.linspace(0.02, 0.025, len(idx))}, index=idx)
        sim = MertonSimulator(start_date=idx[-1], nsim=10)
        with pytest.warns(UserWarning):
            sim.real_world_calibration(proxy, ttm=list(NODES))


class TestMertonSimulation:

    def _calibrated(self, nsim=200, seed=42):
        sim = MertonSimulator(start_date=START, nsim=nsim, seed=seed)
        sim.risk_neutral_calibration(_merton_spot_rates())
        return sim

    def test_short_rate_shapes(self):
        sim = self._calibrated()
        short = sim.simulated_short_rates
        assert short.shape == (len(NODES) + 1, 200)
        assert np.allclose(short[0, :], sim.r)

    def test_discount_factor_and_spot_shapes(self):
        sim = self._calibrated()
        assert sim.simulated_discount_factors.shape == (len(NODES), 200)
        assert sim.simulated_spot_rates.shape == (len(NODES), 200)
        # Positive short rates and horizon → discount factors in (0, 1)
        assert np.all(sim.simulated_discount_factors > 0)

    def test_seed_reproducibility(self):
        a = self._calibrated(seed=123).simulated_short_rates
        b = self._calibrated(seed=123).simulated_short_rates
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a = self._calibrated(seed=1).simulated_short_rates
        b = self._calibrated(seed=2).simulated_short_rates
        assert not np.array_equal(a, b)

    def test_mean_spot_rate_close_to_curve(self):
        # With many paths the average simulated curve should sit near the input curve.
        sim = self._calibrated(nsim=5000, seed=42)
        target = _merton_spot_rates()["spotRate"].to_numpy()
        mean_sim = sim.simulated_spot_rates.mean(axis=1)
        assert np.max(np.abs(mean_sim - target)) < 0.005
