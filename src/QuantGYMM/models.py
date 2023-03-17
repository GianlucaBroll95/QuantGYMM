import warnings

from .descriptors import *
import numpy as np
import pandas as pd
from .utils import *

__all__ = ["MertonSimulator"]


class MertonSimulator:
    """
    Class for risk neutral and real world calibration and simulation of the Merton model.
    """
    start_date = Date(sterilize_attr=["_simulated_short_rates", "_simulated_spot_rates",
                                      "_simulated_discount_factors", "_ttm"])
    nsim = PositiveInteger(sterilize_attr=["_simulated_short_rates", "_simulated_spot_rates",
                                           "_simulated_discount_factors"])
    seed = NonNegativeInteger(sterilize_attr=["_simulated_short_rates", "_simulated_spot_rates",
                                              "_simulated_discount_factors"], none_accepted=True)
    _spot_rates = DataFrame(index_type=pd.DatetimeIndex, sterilize_attr=["_simulated_short_rates",
                                                                         "_simulated_spot_rates",
                                                                         "_simulated_discount_factors", "_ttm"],
                            none_accepted=True)

    def __init__(self, start_date, nsim, seed=None):
        """
        Args:
            start_date (str | pandas.Timestamp): starting date for the simulation
            nsim (int): number of simulation
            seed (int): [optional] seed for reproducibility
        """

        self._af = None
        self.start_date = start_date
        self.nsim = nsim
        self.seed = seed
        self._simulated_short_rates = None
        self._simulated_spot_rates = None
        self._simulated_discount_factors = None
        self._ttm = None
        self._dt = None
        self._s = None
        self._mu = None
        self._r = None

    @property
    def ttm(self):
        if self._ttm is None:
            raise ValueError(
                "Calibration not performed. Use 'calibrate_risk_neutral' method or 'calibrate_real_world'.")
        return self._ttm

    @ttm.setter
    def ttm(self, value):
        raise ValueError("Cannot set ttm. Use Use 'calibrate_risk_neutral' method or 'calibrate_real_world'.")

    @property
    def spot_rates(self):
        if self._spot_rates is None:
            raise ValueError("Spot rates have not been set. Use 'calibrate_risk_neutral' method or "
                             "'calibrate_real_world' method to set spot rates.")
        return self._spot_rates

    @spot_rates.setter
    def spot_rates(self, value):
        raise ValueError("Cannot set spot rates. Use 'calibrate_risk_neutral' method or 'calibrate_real_world' "
                         "method to set spot rates.")

    @property
    def s(self):
        if self._s is None:
            raise ValueError(
                "Calibration not performed. Use 'calibrate_risk_neutral' method or 'calibrate_real_world'.")
        return self._s

    @s.setter
    def s(self, sigma_squared):
        if isinstance(sigma_squared, float) and sigma_squared >= 0:
            self._s = sigma_squared
        else:
            raise ValueError("Sigma squared must be non negative.")

    @property
    def mu(self):
        if self._mu is None:
            raise ValueError(
                "Calibration not performed. Use 'calibrate_risk_neutral' method or 'calibrate_real_world'.")
        return self._mu

    @mu.setter
    def mu(self, mean):
        if isinstance(mean, float):
            self._mu = mean
        else:
            raise ValueError("Mean must be a float number.")

    @property
    def r(self):
        if self._r is None:
            raise ValueError(
                "Calibration not performed. Use 'calibrate_risk_neutral' method or 'calibrate_real_world'.")
        return self._r

    @r.setter
    def r(self, r):
        if isinstance(r, float):
            self._r = r
        else:
            raise ValueError("r must be a float number.")

    @property
    def simulated_short_rates(self):
        if self._simulated_short_rates is None:
            self._simulate_short_rates()
        return self._simulated_short_rates

    @property
    def simulated_spot_rates(self):
        if self._simulated_spot_rates is None:
            self._simulate_spot_rates()
        return self._simulated_spot_rates

    @property
    def simulated_discount_factors(self):
        if self._simulated_discount_factors is None:
            self._simulate_discount_factors()
        return self._simulated_discount_factors

    def risk_neutral_calibration(self, spot_rates, ttm=None):
        """
        Args:
            spot_rates (pandas.Series | pandas.DataFrame): spot rates to be used for the risk-neutral calibration
            ttm (list): [optional] list of term structure nodes at which simulate the term structure.
                        It should be a list of pandas.Timestamp. If None, it uses the same used for calibration.
        """
        self._spot_rates = spot_rates
        self._ttm = accrual_factor("ACT/365", self.start_date, self.spot_rates.index if ttm is None else ttm)
        self._af = accrual_factor("ACT/365", [self.start_date] +
                                  (self.spot_rates.index.tolist() if ttm is None else ttm)).reshape(-1, 1)
        self._dt = np.diff(self._ttm, prepend=0)
        self._param_risk_free_calibration()

    def _param_risk_free_calibration(self):
        X = np.array([np.ones(len(self._ttm)), self._ttm / 2, -(self._ttm ** 2) / 6]).T
        self._r, self._mu, self._s = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(self.spot_rates.to_numpy()).squeeze()
        if self.s < 0:
            raise ValueError("Calibration failed. Sigma is negative.")

    def real_world_calibration(self, short_rates_proxy, ttm):
        """
        Args:
            short_rates_proxy (pandas.Series | pandas.DataFrame): proxy for short rate to be used for the real-world
                                                            calibration, daily observation.
            ttm (list): list of term structure nodes at which simulate the term structure. It should be a list of dates.

        """
        self._spot_rates = short_rates_proxy
        self._ttm = accrual_factor("ACT/365", self.start_date, ttm)
        self._dt = np.diff(self._ttm, prepend=0)
        self._param_real_world_calibration()

    def _param_real_world_calibration(self):
        if pd.to_timedelta(np.diff(self._spot_rates.index).min()).days != 1:
            warnings.warn("It seems that the spot rates frequency is not daily.")
        self.r = self._spot_rates.iloc[-1].item()
        self.mu = self._spot_rates.diff().mean().item() * 252
        self.s = self._spot_rates.diff().var().item() * 252

    def _simulate_discount_factors(self):
        # af = accrual_factor("ACT/365", [self.start_date] + self.spot_rates.index.to_list()).reshape(-1, 1)
        ds = self.simulated_short_rates[1:, :] + self.simulated_short_rates[:-1, :]
        self._simulated_discount_factors = 1 / (1 * np.exp(np.cumsum(ds * self._af / 2, axis=0)))

    def _simulate_spot_rates(self):
        self._simulated_spot_rates = (-np.log(self.simulated_discount_factors.T) / self._ttm).T

    def _simulate_short_rates(self):
        np.random.seed(self.seed)
        epsilon = np.random.randn(len(self._ttm), self.nsim)
        short_rate = np.empty((len(self._ttm) + 1, self.nsim))
        short_rate[0, :] = self.r
        dt = self._dt.reshape(-1, 1)
        self._simulated_short_rates = np.insert(
            self.r + np.cumsum(self.mu * dt + (self.s * self._dt.reshape(-1, 1)) ** 0.5 * epsilon, axis=0),
            0, self.r, axis=0
        )
