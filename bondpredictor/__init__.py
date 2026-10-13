"""Forecasting government bond excess returns for the US, Germany and Japan.

Conventions used throughout the package:

* Data are monthly and indexed by a ``pandas.PeriodIndex`` (freq ``"M"``).
  A value at month ``t`` is what was known at the *end* of month ``t``.
* Maturities and horizons are in years (``1/12`` is one month).
* Yields are continuously compounded, annualised decimals (``0.05`` = 5%).
* A return stored at month ``t`` is the return from the end of ``t`` to the
  end of ``t + h``.  It is not known until ``t + h``; the backtest masks it
  until then.
"""

__version__ = "0.1.0"
