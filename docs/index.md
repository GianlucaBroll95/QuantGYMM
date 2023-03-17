# QuantGYMM Documentation

- [What is QuantGYMM](#what_is)
- [Getting Started](#getting_started)
- [Instruments and Usages](#usages)

## [What is QuantGYMM](#what_is)

QuantGYMM was born within the group project for the course in Fixed Income and Credit Risk, offered in the Master in Finance Insurance and Risk Management at the Collegio Carlo Alberto. The code was therefore devised to tackle the project task, and later become a structured package. The library is mainly based on ```Pandas```, ```Scipy``` and ```NumPy```. ```QuantGYMM``` is still at embrional stage, and any suggestion or contribution is well accepted. 

## [Getting started](#getting_started)

Before installing 'QuantGYMM', be sure that you have a Python version >= 3.9 installed in you computer/local enviroment. If you are using a conda, I suggest to create a virtual enviroment and install a suitable Python version. 

### Suggested Set Up:

1) Create virtual enviroment and activate it:
```
conda create --name [your_env_name_here] python=3.10
conda activate [your_env_name_here]
```
2) Install Jupyter (optional):
```
conda install jupyter
```
3) Install QuantGYMM:
```
pip install QuantGYMM
```
or (better):
```
python3 -m pip install QuantGYMM
```
### Alternative:
1) You could clone the repository:
    
```
git clone https://github.com/GianlucaBroll95/QuantGYMM.git
```
and then install running from within the directory:
```
pip install .
```

## [Instruments and Usages](#usages)

`QuantGYMM` comprises six main modules: `term_structures`, `calendar`, `instruments`, `pricers`, `models` and `utils`. 

### Term Structures
There are four object in this module: `SwapRateCurve`, `SpotRateCurve`, `EuriborCurve` and `DiscountCurve`. The former three curves are functional to the construction of the latter. Each curves can be added to the others.

#### `SwapRateCurve`
```
SwapRateCurve(swap_rate, trade_date, frequency, business_convention, dcc, interpolation)
```
- swap_rate: swap rate market data (swap rate curve);
- trade_date: date of the swap market curve observation;
- frequency: tenor of the swap curve;
- business_convention: business adjustment convention, available conventions are "preceding", "following", "modified_following", "modified_following_bimonthly";
- dcc: day count convention, available conventions are "ACT/365", "ACT/360" and "30/360";
- interpolation: type of interpolation to perform on swap rates.
##### Methods and Properties
- `interpolate()`: interpolates swap rates;
- `interpolated_rates`: returns interpolated swap rates.

#### `SpotRateCurve`
```
SpotRateCurve(spot_rates, trade_date, dcc, business_convention, interpolation, compounding)
```
- spot_rates: spot rates market data;
- trade_date: date of the swap market curve observation;
- dcc: day count convention, available conventions are "ACT/365", "ACT/360" and "30/360";
- business_convention: business adjustment convention, available conventions are "preceding", "following", "modified_following", "modified_following_bimonthly";
- interpolation: type of interpolation to perform on spot rates (see `scipy.interpolate.interp1d`).
- compounding: kind of compounding for the market rates, available are "annually_compounded", "continuous" and "simple".
##### Methods & Properties
- `spot_rates`: interpolated spot rates;
- `discount_factors`: discount factors implicit in the spot rates;

#### `EuriborCurve`
```
EuriborCurve(euribor, trade_date, interpolation)
```
- euribor: euribor market data
- trade_date: date of the swap market curve observation;
- interpolation: type of interpolation to perform on spot rates (see `scipy.interpolate.interp1d`).
##### Methods & Properties
- `spot_rates`: interpolated spot rates;
- `discount_factors`: discount factors implicit in the spot rates;

#### `DiscountCurve`
```
DiscountCurve(rate_curve, compounding, dcc, interpolation)
```
- rate_curve: either a `SwapRateCurve`, a `SpotRateCurve` or an `EuriborCurve` instance.
- compounding: kind of compounding for the market rates, available are "annually_compounded", "continuous" and "simple".
- dcc: day count convention, available conventions are "ACT/365", "ACT/360" and "30/360";
- interpolation: type of interpolation to perform on spot rates (see `scipy.interpolate.interp1d`).
##### Methods & Properties
- `spot_rates`: interpolated spot rates;
- `discount_factors`: discount factors implicit in the spot rates;
- `apply_parallel_shift(shift)`: applies a parallel shift of size 'shift';
- `apply_slope_shift(shift)`: applies a slope shift of size 'shift';
- `apply_curvature_shift(shift)`: applies a curvature shift of size 'shift';
- `reset_shift()`: resets shifts;
- `set_discount_factors(df)`: overrides internal discount factors and set 'df' as discount factors.


### Calendar
The object in this module is `Schedule`. The `Schedule` class handles the coupons bond schedule. 

#### `Schedule`
```
Schedule(start_date, end_date, frequency, convention, eom)
```
- start_date: starting date of the coupon schedule;
- end_date: ending date of the coupon schedule;
- frequency: payment frequency;
- convention: business adjustment convention, available conventions are "preceding", "following", "modified_following", "modified_following_bimonthly";
- eom: end of month convention.

##### Methods & Properties:
- `schedule`: coupons schedule


### Instruments
At the moment, there are two instruments available: `FloatingRateBond` and `VanillaSwap`. 

#### `FloatingRateBond`
```
FloatingRateBond(schedule, dcc, face_amount, fixing_days, spread, floor, cap)
```
- schedule: an instance of `Schedule` for the coupons;
- dcc: day count convention, available conventions are "ACT/365", "ACT/360" and "30/360";
- face_amount: bond face amount;
- fixing_days: how many days, before the coupon accrual start, the reference rate is fixed;
- spread: spread on the floating rate;
- floor: floor for the floating rate + spread;
- cap: cap for the floating rate + spread.

##### Methods & Properties:
- `set_evaluation_date(date)`: sets evaluation date for the bond price;
- `set_pricer(pricer)`: sets the pricing engine;
- `expected_coupons()`: returns the expected coupon;
- `set_historical_euribor(euribor_data)`: sets past euribor for past coupon history;
- `get_coupon_history()`: returns historical coupons;
- `prices()`: returns market prices;
- `sensitivity(shift_type, shift_size, kind)`: calculates bond sensivitiy to term structure shift;
- `set_hedging_instruments(instruments)`: passes the hedging instruments;
- `hedging_ratio(hedge)`: calculates hedge ratio against the "hedge" risk factors (parallel, slope, curvature);
- `set_cds_spread(spread)`: sets the CDS spread for the bond issuer;
- `set_recovery_rate(recovery_rate)`: set the recovery rate for the bond in case of default;
- `survival_probabilities`: returns estimated survival probabilities.

#### `VanillaSwap`
```
VanillaSwap(discount_curve, fixed_leg_frequency, floating_leg_frequency, maturity, start)
```
- discount_curve: an instance of `DiscountCurve`;
- fixed_leg_frequency: frequency of fixed rate payments;
- floating_leg_frequency: frequency of floating rate payments;
- maturity: swap contract maturity;
- start: swap contract start.

##### Methods & Properties
- `market_price()`: swap mark-to-market value (fixed_leg - floating_leg);
- `swap_rate`: fair swap rate of the contract;
- `sensitivity(shift_type, shift_size, kind)`: calculates bond sensivitiy to term structure shift;


### Pricers
The prices modules contains four pricing engines: `Pricer`, `BlackPricer`, `BachelierPricer` and `DisplacedBlackPricer`. A pricer object is not used independently, and must be bound to a bond object.

#### `Pricer`
Basic pricer, works if no cap or floor are embedded in coupons.
```
Pricer(discount_curve)
```
- discount_curve: an instance of `DiscountCurve`;

#### `BlackPricer`
```
BlackPricer(discount_curve, volatility_surface)
```
- discount_curve: an instance of `DiscountCurve`;
- volatility_surface: Black volatility surface

#### `BachelierPricer`
```
BachelierkPricer(discount_curve, volatility_surface)
```
- discount_curve: an instance of `DiscountCurve`;
- volatility_surface: Bachelier volatility surface

#### `DisplacedBlackPricer`
```
BlackPricer(discount_curve, volatility_surface)
```
- discount_curve: an instance of `DiscountCurve`;
- volatility_surface: DisplacedBlack volatility surface


### Models
At the moment only a Merton model simulator is implemented. `MertonSimulator` allows to simulate the term structure by using Merton model and allowing for risk-neutral and real-world calibration.

#### `MertonSimulator`
```
MertonSimulator(start_date, nsim, seed)
```
- start_date: start date for the simulation of the term struture (usually, evaluation date);
- nsim: number of simulations;
- seed: seed for reproducibility purposes.

##### Methods & Properties
- `risk_neutral_calbration(spot_rates, ttm)`: calibrates the model using 'spot_rates'. If 'ttm' is passed, it simulated rates at nodes indicated by the 'ttm' list of dates;
- `real_world_calbration(short_rate_proxy, ttm)`: calibrates the model using the historical data 'short_rate_proxy' and simulates at noted indicated by the 'ttm' list of dates;
- `mu`: calibrated mu for the short rate process;
- `s`: calibrated variance for the short rate process;
- `r`: calibrated r0 for the short rate process;
- `simulated_short_rates`: simulated short rates
- `simulated_spot_rates`: simulated sport rated, extracted from the simulated discount factors;
- `simulated_discount_factors`: simulated discount factors, obtained by the simulation of the money market account.

### Utils
Utils module contains uself functions used in the classes above, but can be usefull also standalone. Main functions are:
- `is_target_holiday(date)`: checks if 'date' is a bank holiday accoridng to TARGET calendar;
- `is_bd(date)`: checks if 'date' is a business day;
- `business_adjustment(convention, date)`: adjusts 'date' according to 'convention' ("preceding", "following", "modified_following", "modified_following_bimonthly")
- `accrual_factor(dcc,*dates)`: determines the accrual factors of 'dates' according to 'dcc' ('ACT/365', 'ACT/360', '30/360')
