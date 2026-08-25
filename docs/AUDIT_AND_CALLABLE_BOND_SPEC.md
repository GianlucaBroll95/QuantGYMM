# QuantGYMM — Full Audit & CallableBond Implementation Spec

Date: 2026-07-02. **Updated 2026-07-03: all Part 1 findings (A1–A9) have been fixed and
verified — 415/415 tests pass. Part 1 is kept for historical context; the implementing
model should start directly at Part 2, skipping §2.0 items 1–3 (already done).**
This document is self-contained: an implementing model should not need any other context
beyond the repository itself.

---

# PART 1 — AUDIT

## 1.1 Project map

| Module | Lines | Responsibility |
|---|---|---|
| `src/QuantGYMM/calendar.py` | 72 | `Schedule` — coupon date generation (backward from end date, EOM rule, business adjustment) |
| `src/QuantGYMM/descriptors.py` | 317 | Validating data descriptors (`Date`, `FloatNumber`, `DataFrame`, ...) with cache-sterilization side effects |
| `src/QuantGYMM/utils.py` | 388 | Day count conventions (ACT/360, ACT/365, ACT/ACT ISDA & ICMA, 30/360, 30E/360, NL/365), TARGET calendar, business adjustment |
| `src/QuantGYMM/term_structures.py` | 710 | `SwapRateCurve`, `SpotRateCurve` (incl. `from_ecb`), `EuriborCurve`, `DiscountCurve`. Fast path: `rate_at` / `discount_factor_at` bypass the daily grid |
| `src/QuantGYMM/pricers.py` | 397 | `Pricer` (deterministic forward projection), `BlackPricer`, `BachelierPricer`, `DisplacedBlackPricer` (cap/floor coupons) |
| `src/QuantGYMM/instruments.py` | 1220 | `FloatingRateBond`, `FixedRateBond`, `VanillaSwap`. Both bonds have `prices`, `sensitivity`, `key_rate_dv01`, `duration`, `modified_duration`, CDS-adjusted valuation |
| `src/QuantGYMM/models.py` | 187 | `MertonSimulator` (short-rate simulation) |
| `src/QuantGYMM/__init__.py` | 0 | Empty — no top-level API |

Architecture conventions worth preserving in new code:
- Bonds are dumb data holders; `FloatingRateBond` delegates valuation to a `Pricer` bound
  via `set_pricer` / `transfer_bond_features`. `FixedRateBond` prices itself off a
  `DiscountCurve` directly.
- Private `_attr` is the cache / raw storage; the public property **raises** if unset.
  Any code checking "is it set?" must test the private attribute (e.g. `self._cds_spread`),
  never the property.
- Sensitivities are computed by mutating `SpotRateCurve.spot_rates_data` node-by-node
  (assigning to it invalidates `_interpolator` via the `DataFrame` descriptor's
  `sterilize_attr`), repricing through the `rate_at` fast path, and restoring the original
  data in a `finally` block.

## 1.2 Findings — bugs (ordered by severity)

**A1. `DiscountCurve.discount_factor_at` crashes for SwapRateCurve-based curves.**
`term_structures.py:188` calls `self.rate_curve.rate_at(dates)`, but `SwapRateCurve` has
no `rate_at` method (only `SpotRateCurve` and `EuriborCurve` do). Since
`Pricer._get_forward_rates` (pricers.py:83), `Pricer.present_value` (pricers.py:150) and
`FixedRateBond.prices` (instruments.py:~927) all route through `discount_factor_at`, **any
bond priced off a swap-rate-bootstrapped DiscountCurve raises AttributeError**. Fix: add a
`rate_at(dates)` method to `DiscountCurve` itself that interpolates `self.sr` (the
bootstrapped node table `{maturity, term, spotRate}` built by `_get_spot`), caches the
scipy interpolator, and use it in `discount_factor_at` when
`not hasattr(self.rate_curve, "rate_at")`. Mirror the implementation of
`SpotRateCurve.rate_at` (term_structures.py:647-668).

**A2. Legacy shift mechanism is silently dead for bonds.**
`apply_parallel_shift` / `apply_slope_shift` / `apply_curvature_shift`
(term_structures.py:258-299) mutate the daily `_spot_rates` grid. The pricing fast path
(`discount_factor_at` → `rate_curve.rate_at`) never reads that grid, so after
`dc.apply_parallel_shift()`, `bond.prices()` returns the **unshifted** price with no error
(`shift_flag` still triggers cache invalidation in `Pricer`, but recomputation reads
unshifted node data). VanillaSwap still works because it reads the `discount_factors` grid.
Fix options: (a) deprecate the `apply_*_shift` API and migrate `VanillaSwap.sensitivity`
to the node-mutation pattern used by both bonds; or (b) make `rate_at` add `self._shift`.
Recommended: (a) — one mechanism, not two.

**A3. `EuriborCurve.discount_factors` returns `None` on second access.**
term_structures.py:443-447 — the `return` is inside `if self._discount_factors is None:`.
Second access skips the branch and falls through returning `None`. Move the return out.

**A4. `pyproject.toml` declares `requires-python = ">=3.9"` but the code needs 3.12.**
`match` statements everywhere (3.10+) and nested same-quote f-strings in
`SpotRateCurve.from_ecb` (term_structures.py:586, PEP 701, 3.12+). Set
`requires-python = ">=3.12"` and update the README's suggested `python=3.10` env.

**A5. Pre-existing test failure (Python-version dependent).**
`test_descriptors.py::test_invalid_date[20230101]` — on Python ≥3.11,
`datetime.date.fromisoformat("20230101")` accepts the compact form, so the `Date`
descriptor no longer raises. Decide: either accept compact ISO (drop the parametrize
case) or reject it explicitly in `Date.__set__`. Not caused by recent work.

**A6. `MertonSimulator.real_world_calibration` never sets `self._af`.**
models.py:150-161 sets `_ttm`/`_dt` but not `_af` (only `risk_neutral_calibration` does,
line 139). `_simulate_discount_factors` (line 172-173) then does `ds * self._af / 2` with
`_af = None` → TypeError. Set `_af` in `real_world_calibration` too.

**A7. `VanillaSwap._get_calendar` passes float `periods` to `pd.date_range`.**
instruments.py:~718-735 — `np.ceil` returns float; pandas FutureWarning today, will raise
in a future release (26 warnings in the suite). Wrap with `int(...)`.

**A8. Empty-past-reset edge case.**
`Pricer._get_current_coupon` (pricers.py:97), `BlackPricer._get_current_coupon`
(pricers.py:258) and `FloatingRateBond.key_rate_dv01` (instruments.py:~390) all do
`reset_dates[past_mask][-1]` → IndexError if `evaluation_date` precedes the first reset
date. Low priority (unusual configuration) but a clear `ValueError("evaluation date
precedes first reset date")` guard would help.

**A9. Typo in `SpotRateCurve.__add__` error message** — "SportRateCurve"
(term_structures.py:705).

## 1.3 Findings — design notes (no action required, context for new code)

- `__init__.py` is empty: users must import from submodules
  (`from QuantGYMM.instruments import ...`). Tests do the same via relative imports.
  Any new module must be imported explicitly; nothing to register.
- `calendar.py` shadows the stdlib module name — harmless because the package only uses
  relative imports, but don't add `import calendar` anywhere inside the package.
- Accrued interest in `Pricer.present_value` and `FixedRateBond.prices` uses a hardcoded
  T+2 settlement (`evaluation_date + BDay(2)`) and a calendar-day ratio, ignoring the
  bond dcc. Convention choice — keep consistent in new code.
- `expected_coupons` DataFrames use `.replace(np.nan, "-")`, so non-`coupon` columns may
  contain strings. Only ever consume the `coupon`, `couponStart`, `couponEnd` columns
  numerically (the `coupon` column is always numeric via `np.nansum`).
- `key_rate_dv01` legitimately shows non-zero DV01 at curve nodes beyond bond maturity for
  floaters (forward-rate `df2` dates interpolate between surrounding nodes). Don't "fix".
- Duplicated `_interpolate_spot_rates` / `_get_discount_factors` across the three curve
  classes — refactor candidate, out of scope here.

---

# PART 2 — CallableBond IMPLEMENTATION SPEC (hand to Sonnet)

Goal: a `CallableBond` wrapping an existing `FixedRateBond` **or** `FloatingRateBond`
(composition, not inheritance), supporting a Bermudan-style call schedule, with two
pricing methods:

- `method="worst"` — **price-to-worst** (deterministic, no vol model). Works for both
  fixed and floating underlyings. This is the baseline and must be implemented first.
- `method="hw"` — **Hull-White one-factor**, European call (single call date) via
  Jamshidian decomposition, closed form. **Fixed-rate underlying only**; raise
  `NotImplementedError` for floaters and for multiple call dates (Bermudan tree is a
  later stage, out of scope now).

## 2.0 Prerequisite fixes — **ALL DONE (2026-07-03), skip to §2.1**

All Part 1 findings A1–A9 are fixed. Relevant behavior changes to be aware of:
- `DiscountCurve` now has its own `rate_at` (fallback for SwapRateCurve-based curves);
  `discount_factor_at` works for every curve type.
- `discount_factor_at` reads the shifted daily grid when `shift_flag` is set, so
  `apply_*_shift` affects bond prices correctly (A2 fixed via option (b), not (a) —
  the legacy shift API was kept, not deprecated, because VanillaSwap on a
  SwapRateCurve-based curve has no `spot_rates_data` to node-mutate).
- `Pricer._get_current_coupon`, `BlackPricer._get_current_coupon` and
  `FloatingRateBond.key_rate_dv01` raise a clear `ValueError` when the evaluation date
  precedes the first reset date.
- The date descriptor test now treats compact ISO "20230101" as valid (Python 3.11+
  `date.fromisoformat` behavior); baseline is **415/415 passing**.

Before starting §2.1, run `python -m pytest src/QuantGYMM/test/ -q` — expect 415 passed.

## 2.1 Files

| File | Action |
|---|---|
| `src/QuantGYMM/models.py` | Add Hull-White helper functions (§2.2). Extend `__all__`. |
| `src/QuantGYMM/callables.py` | **New module** — `CallableBond` (§2.3). `__all__ = ["CallableBond"]`. |
| `src/QuantGYMM/test/test_callable_bond.py` | **New test file** (§2.5). |

Do not modify `instruments.py` except where §2.0 says so.

## 2.2 Hull-White helpers (`models.py`)

Add module-level functions (plain functions, not a class — matches the utils style):

```python
def hw_b(t, T, a):
    """Hull-White B(t,T) = (1 - exp(-a (T - t))) / a."""
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
    """
    sigma_p = sigma * np.sqrt((1.0 - np.exp(-2.0 * a * T)) / (2.0 * a)) * hw_b(T, S, a)
    h = np.log(df_S / (df_T * strike)) / sigma_p + sigma_p / 2.0
    if kind == "call":
        return df_S * norm.cdf(h) - strike * df_T * norm.cdf(h - sigma_p)
    if kind == "put":
        return strike * df_T * norm.cdf(-h + sigma_p) - df_S * norm.cdf(-h)
    raise ValueError("kind must be 'call' or 'put'.")
```

Import `from scipy.stats import norm` at the top of models.py. Guard `sigma_p == 0`
(sigma=0): return the intrinsic value `max(df_S - strike * df_T, 0)` for a call
(symmetric for put) to avoid division by zero.

## 2.3 `CallableBond` (`callables.py`)

### Imports

```python
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from .instruments import FixedRateBond, FloatingRateBond
from .models import hw_b, hw_zcb_option
from .utils import accrual_factor
```

### Constructor and state

```python
class CallableBond:
    def __init__(self, bond, call_schedule, mean_reversion=0.03, volatility=0.01):
```

- `bond`: must be `FixedRateBond` or `FloatingRateBond`, else
  `ValueError("'bond' must be a FixedRateBond or FloatingRateBond object.")`.
- `call_schedule`: `pd.Series` with `DatetimeIndex` (call dates) and float values =
  **clean call price as % of face** (e.g. 100.0 = at par). Validate type and index type;
  sort by index; store as `self._call_schedule`.
- `mean_reversion` / `volatility`: positive floats (HW `a` and `sigma`). Store privately;
  expose read-only properties. Add `set_hw_params(mean_reversion=None, volatility=None)`
  to update them.
- Convenience delegation (read-only properties forwarding to the wrapped bond):
  `face_amount`, `evaluation_date`, `schedule`. Also expose `self.bond` and a
  `call_schedule` property that filters to **future call dates only**
  (`index > self.bond.evaluation_date`) — raise
  `ValueError("All call dates are in the past.")` if that filter leaves nothing.

Discount curve access helper (used everywhere below):

```python
def _discount_curve(self):
    if isinstance(self.bond, FixedRateBond):
        return self.bond.discount_curve
    return self.bond.pricer.discount_curve
```

Cash flow table helper — normalize both bond types to one shape
(`couponEnd` timestamps, `cashFlowNoRedemption` floats; face redemption handled
separately):

- Fixed: `cf = self.bond.cash_flows`; coupons only =
  `cf.cashFlow.to_numpy().copy()`; subtract
  `self.bond.redemption / 100 * self.bond.face_amount` from the last element;
  dates = `cf.couponEnd`.
- Floating: `ec = self.bond.pricer.expected_coupons`; coupons =
  `ec.coupon.to_numpy().astype(float)`; dates = `ec.couponEnd`. (Face amount is not in
  `coupon` for floaters — it's added at maturity, matching `Pricer.present_value`.)

### `prices(method="worst") -> dict`

Return shape (mirrors the bond classes' `prices()` style):

```python
{
    "straightValue": {"dirtyPrice": ..., "cleanPrice": ..., "accruedInterest": ...},
    "optionValue": <float>,
    "callableValue": {"dirtyPrice": ..., "cleanPrice": ..., "accruedInterest": ...},
}
```

`straightValue` = `self.bond.prices()["riskFreeValue"]` (ignore CDS in v1 — if
`bond._cds_spread` is set, raise
`NotImplementedError("CallableBond does not support CDS-adjusted valuation yet.")`).
`callableValue.dirtyPrice = straight.dirtyPrice - optionValue`;
`accruedInterest` copied from straight; clean = dirty − accrued.

#### method="worst" (both underlying types)

For each future call date `T_j` with clean call price `K_j` (% of face):

1. Build the "called at T_j" cash flow stream: all coupons with
   `couponEnd <= T_j`, plus a redemption payment of `K_j / 100 * face_amount` at `T_j`.
   (Assume call dates coincide with coupon payment dates — **validate this**: every call
   date must be in `schedule.schedule["paymentDate"]`, else
   `ValueError("Call dates must coincide with coupon payment dates.")`. This keeps
   accrued-at-call zero and is market-standard for callables.)
2. PV the stream with `dc.discount_factor_at(dates)` (dc = `self._discount_curve()`).
3. Also compute the to-maturity PV = straight dirty price.
4. `worst = min(all called PVs, straight)`;
   `optionValue = straight_dirty - worst` (≥ 0 by construction).

#### method="hw" (FixedRateBond only, exactly one future call date)

If the underlying is a `FloatingRateBond` → `NotImplementedError("Hull-White valuation
is only available for fixed rate underlyings.")`. If more than one future call date →
`NotImplementedError("Bermudan callables not supported yet; use method='worst'.")`.

Let `T` = the call date (years from trade date via
`accrual_factor(dc.dcc, dc.trade_date, call_date).item()`), `K` = dirty strike per unit
face = `K_clean / 100` (accrued is zero because call dates are coupon dates). Remaining
cash flows after `T`: coupons `c_i` at times `t_i > T` **plus redemption at maturity**
(use the fixed bond's `cash_flows.cashFlow`, which already contains redemption in the
last row — take rows with `couponEnd > call_date`, keep amounts per unit face by dividing
by `face_amount`).

Jamshidian decomposition:

1. Today's discount factors: `P0 = dc.discount_factor_at(dates)` for `T` and each `t_i`.
2. Instantaneous forward at `T` (numerical):
   `eps = 1/365`; `f0T = -(ln P0(T+eps) - ln P0(T-eps)) / (2 eps)` using
   `discount_factor_at` on `call_date ± pd.Timedelta(days=1)` and year-fraction eps
   computed with the same dcc.
3. HW bond price function at `T` for short rate `r`:
   `B_i = hw_b(T, t_i, a)`;
   `lnA_i = ln(P0(t_i)/P0(T)) + B_i * f0T - (sigma**2 / (4*a)) * B_i**2 * (1 - exp(-2*a*T))`;
   `P(T, t_i; r) = exp(lnA_i - B_i * r)`.
4. Solve `sum_i c_i * P(T, t_i; r*) = K` for `r*` with
   `brentq(f, -1.0, 1.0)` (widen bracket ×2 up to |r|=10 if `f` has the same sign at
   both ends; raise ValueError if still unbracketed).
5. Per-cash-flow strikes: `K_i = P(T, t_i; r*)`.
6. Issuer's call option value per unit face =
   `sum_i c_i * hw_zcb_option(P0(T), P0(t_i), K_i, T, t_i, a, sigma, kind="call")`.
7. `optionValue = that * face_amount`.

### `option_value(method="worst") -> float`

`return self.prices(method)["optionValue"]`.

### `sensitivity(shift_type="parallel", shift_size=0.01, kind="symmetric", method="worst") -> float`

Copy the node-mutation pattern from `FixedRateBond.sensitivity`
(instruments.py:1069-1128) **verbatim in structure**: validate args, snapshot
`sr_curve.spot_rates_data`, apply `_node_shift`, reprice, restore in `finally`.
Differences:

- `sr_curve = self._discount_curve().rate_curve` — but note after fix A1 the curve may be
  a `SwapRateCurve` without `spot_rates_data`; in that case raise
  `ValueError("sensitivity requires a SpotRateCurve-based discount curve.")`
  (same limitation as the existing bonds' `key_rate_dv01`).
- `_reprice()` must clear the underlying bond's caches before repricing, then return the
  **callable dirty price**:
  ```python
  def _reprice() -> float:
      if isinstance(self.bond, FloatingRateBond):
          self.bond.pricer._forward_rates = None
          self.bond.pricer._expected_coupons = None
      return self.prices(method)["callableValue"]["dirtyPrice"]
  ```
  (`FixedRateBond` has no pricer caches; `cash_flows` doesn't depend on the curve.)
- The `finally` block restores `spot_rates_data` and clears the same caches again.

### `key_rate_dv01(shift_size=0.0001, kind="symmetric", method="worst") -> dict`

Same node-loop as `FixedRateBond.key_rate_dv01` (instruments.py:1035-1067) with the
`_reprice` above. Returns `{tenor_date: krd}`.

### `effective_duration(method="worst") -> float`

`-sensitivity(parallel) * 10000 / callable_dirty_price` — same formula as the bonds'
`modified_duration`, but on the callable price. (Name it `effective_duration`; that is
the standard term for option-adjusted duration.)

## 2.4 Behavioral requirements (encode these as tests)

1. `callableValue.dirtyPrice <= straightValue.dirtyPrice` always (both methods).
2. `optionValue >= 0` always.
3. With absurdly high call prices (e.g. 999.0), worst-method callable == straight
   (option value 0, exact equality).
4. HW: option value is monotonically increasing in `sigma`.
5. HW with `sigma → 1e-9`: option ≈ intrinsic = `max(0, PV(remaining CFs) − K·P0(T))`
   (tolerance 1e-6 of face).
6. HW callable ≤ straight and HW option ≥ worst-method option is **not** guaranteed in
   general — do not test cross-method inequalities other than (1) and (2).
7. After `sensitivity` / `key_rate_dv01`, `sr_curve.spot_rates_data` equals the original
   (use `pd.testing.assert_frame_equal`).
8. Callable |parallel DV01| ≤ straight |parallel DV01| (call caps upside → shorter
   effective duration). Use the worst method with an in-the-money call for a robust test.

## 2.5 Test file: `src/QuantGYMM/test/test_callable_bond.py`

Reuse the fixture style of `test_floating_rate_bond.py` (module head, lines 1-71):
same `TRADE_DATE = pd.Timestamp("2020-01-02")`, `EVAL_DATE = pd.Timestamp("2022-06-15")`,
flat 2% curve `SpotRateCurve` → `DiscountCurve` (both `continuous`, `ACT/365`, linear),
annual `Schedule` 2020-01-02 → 2024-01-02.

Fixtures:

- `dc`, `schedule_annual` — copy from test_floating_rate_bond.py.
- `fixed_bond`: `FixedRateBond(schedule_annual, "30/360", 1_000_000.0, coupon_rate=0.05)`
  (5% coupon on a 2% curve → trades above par → call at 100 is in the money → good for
  tests 3/5/8), `set_evaluation_date(EVAL_DATE)`, `set_discount_curve(dc)`.
- `floating_bond`: copy the `bond_annual` fixture (incl. historical euribor) from
  test_floating_rate_bond.py, plus `set_pricer(Pricer(dc))`.
- `call_sched`: `pd.Series([100.0], index=[<the 2023 payment date>])` — read the actual
  date from `schedule_annual` (`schedule.schedule["paymentDate"]`, the one in 2023) so
  business adjustment is respected.
- `callable_fixed`, `callable_floating` fixtures wrapping the above.

Test classes and cases (~24 tests):

**TestConstruction** — wrong bond type raises; call_schedule not a Series raises;
non-DatetimeIndex raises; call date not on a payment date raises; all-past call dates →
ValueError on `call_schedule` access; past dates filtered out when some are future;
delegation: `face_amount`, `evaluation_date` match the wrapped bond.

**TestPriceToWorstFixed** — returns dict with the three keys; requirement 1;
requirement 2; requirement 3 (high strike); straight value matches
`fixed_bond.prices()["riskFreeValue"]` exactly; dirty = clean + accrued for callableValue.

**TestPriceToWorstFloating** — same requirements 1-3 on the floating underlying;
works without historical euribor when `set_current_coupon_rate` is used (reuse the
inception-style setup from `TestCurrentCouponRatePath` in test_floating_rate_bond.py).

**TestHullWhite** — requirement 4 (sigma 0.005 < 0.01 < 0.02); requirement 5;
requirements 1-2; floater underlying raises NotImplementedError; two future call dates
raises NotImplementedError; CDS-set underlying raises NotImplementedError.

**TestGreeks** — `sensitivity` returns negative float; requirement 8; requirement 7
(curve restored, both for sensitivity and key_rate_dv01); `key_rate_dv01` keys ==
`spot_rates_data.index`; symmetric vs oneside same sign; `effective_duration` positive
and less than straight `modified_duration`.

## 2.6 Verification

```
python -m pytest src/QuantGYMM/test/ -q
```

Expected: **all new tests pass** and the prior 415 still pass (0 failures).

Style rules for the implementing model:
- Match the existing docstring format (Args/Returns, no type hints in signatures except
  return annotations like `-> dict`).
- Use the private-attribute pattern for anything optional (raise-on-unset properties).
- Never leave the curve in a shocked state — always `try/finally` restore.
- Do not touch the legacy `apply_*_shift` mechanism.
- Do not commit; leave changes in the working tree.
