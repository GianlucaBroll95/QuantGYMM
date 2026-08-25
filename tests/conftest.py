"""
Make the src/ layout importable when the package is not pip-installed,
so `pytest` works straight from a source checkout.
"""
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
