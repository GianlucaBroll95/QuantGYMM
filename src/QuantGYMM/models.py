from .descriptors import *
import numpy as np
import pandas as pd
from .utils import *

__all__ = ["MertonSimulator"]


class MertonSimulator:
    """
    Class for risk neutral calibration and simulation of the Merton model.
    """
    start_date = Date(sterilize_attr=["_simulated_short_rates", "_simulated_spot_rates",
                                      "_simulated_discount_factors", "_ttm"])
    nsim = PositiveInteger(sterilize_attr=["_simulated_short_rates", "_simulated_spot_rates",
                                           "_simulated_discount_factors", "_ttm"])
    seed = NonNegativeInteger(sterilize_attr=["_simulated_short_rates", "_simulated_spot_rates",
                                              "_simulated_discount_factors", "_ttm"], none_accepted=True)
    spot_rates = DataFrame(index_type=pd.DatetimeIndex, sterilize_attr=["_simulated_short_rates",
                                                                        "_simulated_spot_rates",
                                                                        "_simulated_discount_factors", "_ttm"])

    def __init__(self, spot_rates, start_date, nsim, seed=None):
        """
        Args:
            spot_rates (pandas.Series | pandas.DataFrame): spot rates to be used for the risk-neutral calibration
            start_date (str | pandas.Timestamp): starting date for the simulation
            nsim (int): number of simulation
            seed (int): seed for reproducibility
        """
        self.spot_rates = spot_rates
        self.start_date = start_date
        self.nsim = nsim
        self.seed = seed
        self._simulated_short_rates = None
        self._simulated_spot_rates = None
        self._simulated_discount_factors = None
        self._ttm = None
        self._s = None
        self._mu = None
        self._r = None

    @property
    def ttm(self):
        if self._ttm is None:
            self._ttm = accrual_factor("ACT/365", self.start_date, self.spot_rates.index)
        return self._ttm

    @property
    def s(self):
        if self._s is None:
            self._param()
        return self._s

    @property
    def mu(self):
        if self._mu is None:
            self._param()
        return self._mu

    @property
    def r(self):
        if self._r is None:
            self._param()
        return self._r

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

    def _simulate_discount_factors(self):
        A = - self.mu / 2 * self.ttm ** 2 + self.s / 6 * self.ttm ** 3
        self._simulated_discount_factors = np.exp(-self.ttm * self.simulated_short_rates.T + A).T

    def _simulate_spot_rates(self):
        self._simulated_spot_rates = (-np.log(self.simulated_discount_factors.T) / self.ttm).T

    def _simulate_short_rates(self):
        dt = np.diff(self.ttm, prepend=0)
        np.random.seed(self.seed)
        epsilon = np.random.randn(len(self.ttm), self.nsim)
        short_rate = np.empty((len(self.ttm), self.nsim))
        short_rate[0, :] = self.r
        for i in range(1, len(self.ttm + 1)):
            short_rate[i, :] = (short_rate[i - 1, :] + self.mu * dt[i - 1]
                                + (self.s * dt[i - 1]) ** 0.5 * epsilon[i - 1, :])
        self._simulated_short_rates = short_rate

    def _param(self):
        X = np.array([np.ones(len(self.ttm)), self.ttm / 2, -(self.ttm ** 2) / 6]).T
        self._r, self._mu, self._s = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(self.spot_rates.to_numpy()).squeeze()
        if self.s < 0:
            raise ValueError("Calibration failed. Sigma is negative.")
