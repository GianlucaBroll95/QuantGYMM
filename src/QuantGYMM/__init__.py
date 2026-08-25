"""
QuantGYMM — multi-asset pricing and hedging toolkit.

The package is organised as a shared core (calendars, descriptors, term
structures, FX rates, utilities) plus one subpackage per asset class:

    QuantGYMM.fixed_income  — bonds, swaps, callables, pricers, rate models
    QuantGYMM.equity        — stocks (and, in the future, equity derivatives)

The public API of every submodule is re-exported here so that consumers can
simply do:

    from QuantGYMM import FixedRateBond, Stock, Schedule, DiscountCurve, FXRate
"""
from .utils import *
from .descriptors import *
from .calendar import *
from .term_structures import *
from .fx import *
from .fixed_income import *
from .equity import *

from . import calendar as calendar
from . import descriptors as descriptors
from . import equity as equity
from . import fixed_income as fixed_income
from . import fx as fx
from . import term_structures as term_structures
from . import utils as utils

__version__ = "0.4"

__all__ = (
    utils.__all__
    + descriptors.__all__
    + calendar.__all__
    + term_structures.__all__
    + fx.__all__
    + fixed_income.__all__
    + equity.__all__
    + ["__version__"]
)
