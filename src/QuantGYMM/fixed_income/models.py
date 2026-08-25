import warnings

from ..descriptors import *
import numpy as np
import pandas as pd
from scipy.stats import norm
from ..utils import *

__all__ = ["MertonSimulator", "hw_b", "hw_zcb_option", "hw_trinomial_tree",
           "hw_bermudan_bond_value"]


def hw_b(t, T, a):
    """
    Hull-White B(t,T) = (1 - exp(-a (T - t))) / a.
    Args:
        t (float): valuation time (year fraction).
        T (float): target time (year fraction).
        a (float): mean reversion speed.
    Returns:
        float
    """
    return (1.0 - np.exp(-a * (T - t))) / a


def hw_zcb_option(df_T, df_S, strike, T, S, a, sigma, kind="call"):
    """
    Closed-form price at time 0 of a European option expiring at T on a
    zero-coupon bond maturing at S (T < S), under one-factor Hull-White
    fitted to today's curve.

    Args:
        df_T (float): today's discount factor P(0, T)
        df_S (float): today's discount factor P(0, S)
        strike (float): option strike (price of the ZCB, per unit notional)
        T (float): option expiry in years
        S (float): ZCB maturity in years
        a (float): mean reversion speed
        sigma (float): short rate volatility
        kind (str): 'call' or 'put'
    Returns:
        float
    """
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'.")

    sigma_p = sigma * np.sqrt((1.0 - np.exp(-2.0 * a * T)) / (2.0 * a)) * hw_b(T, S, a)

    if sigma_p == 0:
        if kind == "call":
            return max(df_S - strike * df_T, 0.0)
        return max(strike * df_T - df_S, 0.0)

    h = np.log(df_S / (df_T * strike)) / sigma_p + sigma_p / 2.0
    if kind == "call":
        return df_S * norm.cdf(h) - strike * df_T * norm.cdf(h - sigma_p)
    return strike * df_T * norm.cdf(-h + sigma_p) - df_S * norm.cdf(-h)


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
        self._af = accrual_factor("ACT/365", [self.start_date] + list(ttm)).reshape(-1, 1)
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


def hw_trinomial_tree(dt, n_steps, a, sigma, discount_factors):
    """
    Build a Hull-White trinomial tree calibrated to an observed discount curve.

    Two-stage construction of Hull & White (1994, 1996). Stage one builds a
    symmetric tree for the zero-mean process dx = -a x dt + sigma dW with
    space step dx = sigma sqrt(3 dt), switching the branching at the top and
    bottom of the lattice so that all probabilities stay positive. Stage two
    displaces each time slice by alpha(t), solved from the Arrow-Debreu prices,
    so that the tree reprices the input discount factors exactly. The short
    rate at node (i, j) is then r = alpha[i] + j dx.

    Args:
        dt (float): time step in years (uniform).
        n_steps (int): number of time steps.
        a (float): mean reversion speed, strictly positive.
        sigma (float): short rate volatility, strictly positive.
        discount_factors (numpy.ndarray): P(0, t_i) for i = 0 ... n_steps, on
            the grid t_i = i dt, so length n_steps + 1 with P(0, 0) = 1.

    Returns:
        dict with keys:
            'dx'      (float) space step,
            'j_max'   (int) index at which branching switches,
            'j'       (list of numpy.ndarray) node indices alive at each slice,
            'alpha'   (numpy.ndarray) displacement per slice, length n_steps,
            'probs'   (list of numpy.ndarray) (3, n_nodes) probabilities
                      ordered (up, middle, down) for each slice,
            'k'       (list of numpy.ndarray) index of the middle destination
                      node reached from each node of the slice.
    """
    if a <= 0 or sigma <= 0:
        raise ValueError("'a' and 'sigma' must be strictly positive.")
    discount_factors = np.asarray(discount_factors, dtype=float)
    if len(discount_factors) != n_steps + 1:
        raise ValueError(f"'discount_factors' must have {n_steps + 1} entries, got {len(discount_factors)}.")

    # dx = sigma sqrt(3 dt) is not arbitrary. Any multiple matches the first two
    # moments, given suitable probabilities; this one also matches the FOURTH
    # moment of the normal exactly (pu = pd = 1/6, pm = 2/3 give 3 sigma^4 dt^2,
    # the normal's kurtosis), so each step is more accurate and the tree
    # converges faster. With sqrt(2 dt) the fourth moment would come out a third
    # too small.
    dx = sigma * np.sqrt(3.0 * dt)

    # Threshold at which the branching has to switch: above it the normal
    # branching would need a negative probability, so the top of the tree
    # branches downwards and the bottom upwards.
    #
    # The number is a consequence of dx, not a fitted constant. With downward
    # branching pm = -1/3 - eta^2 + 2 eta, whose roots are 1 -/+ sqrt(2/3);
    # below the lower one no valid set of probabilities exists. The 2/3 is pm at
    # the central node, which comes from the sqrt(3 dt) choice above - change one
    # and the other has to change with it (sqrt(2 dt) would give 1 - sqrt(1/2)).
    branching_threshold = 1.0 - np.sqrt(2.0 / 3.0)   # 0.1835...
    j_max = int(np.ceil(branching_threshold / (a * dt)))

    j_slices, prob_slices, k_slices = [], [], []
    for i in range(n_steps):
        width = min(i, j_max)
        j = np.arange(width, -width - 1, -1, dtype=int)
        adt = a * j * dt
        adt2 = adt ** 2

        # Normal branching: the middle child sits on the same index.
        k = j.copy()
        pu = 1.0 / 6.0 + (adt2 - adt) / 2.0
        pm = 2.0 / 3.0 - adt2
        pd = 1.0 / 6.0 + (adt2 + adt) / 2.0

        top, bottom = j == j_max, j == -j_max
        if top.any():
            k[top] = j[top] - 1
            pu[top] = 7.0 / 6.0 + (adt2[top] - 3.0 * adt[top]) / 2.0
            pm[top] = -1.0 / 3.0 - adt2[top] + 2.0 * adt[top]
            pd[top] = 1.0 / 6.0 + (adt2[top] - adt[top]) / 2.0
        if bottom.any():
            k[bottom] = j[bottom] + 1
            pu[bottom] = 1.0 / 6.0 + (adt2[bottom] + adt[bottom]) / 2.0
            pm[bottom] = -1.0 / 3.0 - adt2[bottom] - 2.0 * adt[bottom]
            pd[bottom] = 7.0 / 6.0 + (adt2[bottom] + 3.0 * adt[bottom]) / 2.0

        j_slices.append(j)
        k_slices.append(k)
        prob_slices.append(np.vstack([pu, pm, pd]))

    # Stage two: alpha from the Arrow-Debreu prices, one slice at a time.
    alpha = np.empty(n_steps, dtype=float)
    q = np.array([1.0])
    for i in range(n_steps):
        j = j_slices[i]
        disc_unit = np.exp(-j * dx * dt)
        alpha[i] = (np.log(q.dot(disc_unit)) - np.log(discount_factors[i + 1])) / dt

        if i + 1 < n_steps:
            width_next = min(i + 1, j_max)
            q_next = np.zeros(2 * width_next + 1)
            # Slices are stored from the highest index down, so position of
            # index n inside a slice of half width w is w - n.
            discounted = q * np.exp(-(alpha[i] + j * dx) * dt)
            probs = prob_slices[i]
            for offset, row in zip((1, 0, -1), (0, 1, 2)):
                targets = width_next - (k_slices[i] + offset)
                np.add.at(q_next, targets, discounted * probs[row])
            q = q_next

    return {"dx": dx, "j_max": j_max, "j": j_slices, "alpha": alpha,
            "probs": prob_slices, "k": k_slices}


def hw_bermudan_bond_value(tree, dt, cash_flows, call_prices=None):
    """
    Value a bond, optionally callable on any subset of the grid, by backward
    induction on a Hull-White trinomial tree.

    The issuer holds the option, so at a call date the holder's value is capped:
    the bond is worth min(continuation value, call price), and the coupon due on
    that date is paid on top. Applying that cap at several dates is what a
    closed form cannot reproduce, and is the source of the negative convexity.

    Args:
        tree (dict): output of hw_trinomial_tree.
        dt (float): time step in years, the same used to build the tree.
        cash_flows (numpy.ndarray): amount paid at each grid point, length
            n_steps + 1, zero where nothing is paid. The redemption belongs to
            the last non-zero entry.
        call_prices (numpy.ndarray | None): call price payable at each grid
            point, NaN where the bond is not callable. Length n_steps + 1.
            Prices are in the same unit as cash_flows (i.e. per face amount).

    Returns:
        float: present value at the root of the tree.
    """
    n_steps = len(tree["alpha"])
    cash_flows = np.asarray(cash_flows, dtype=float)
    if len(cash_flows) != n_steps + 1:
        raise ValueError(f"'cash_flows' must have {n_steps + 1} entries, got {len(cash_flows)}.")
    if call_prices is None:
        call_prices = np.full(n_steps + 1, np.nan)
    call_prices = np.asarray(call_prices, dtype=float)

    dx, j_max = tree["dx"], tree["j_max"]

    # Terminal slice: only what is paid at the horizon.
    width = min(n_steps, j_max)
    value = np.full(2 * width + 1, cash_flows[-1], dtype=float)

    for i in range(n_steps - 1, -1, -1):
        j, k, probs = tree["j"][i], tree["k"][i], tree["probs"][i]
        width_next = min(i + 1, j_max)

        positions = width_next - k
        continuation = (probs[0] * value[positions - 1]
                        + probs[1] * value[positions]
                        + probs[2] * value[positions + 1])
        value = np.exp(-(tree["alpha"][i] + j * dx) * dt) * continuation

        # Il richiamo avviene PRIMA di incassare la cedola del giorno: chi
        # richiama paga lo strike piu' la cedola dovuta.
        if np.isfinite(call_prices[i]):
            value = np.minimum(value, call_prices[i])
        value = value + cash_flows[i]

    return float(value[0])
