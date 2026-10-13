"""Helpers for putting raw series of any frequency onto a monthly grid."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Months covered by one observation, keyed by inferred frequency.
_PERIOD_MONTHS = {"M": 1, "Q": 3, "A": 12}


def infer_frequency(index: pd.DatetimeIndex) -> str:
    """Classify a date index as daily/weekly ("D"), monthly, quarterly or annual."""
    if len(index) < 3:
        return "M"
    gaps = np.diff(index.values).astype("timedelta64[D]").astype(float)
    med = float(np.median(gaps))
    if med <= 10:
        return "D"
    if med <= 45:
        return "M"
    if med <= 135:
        return "Q"
    return "A"


def to_monthly(
    series: pd.Series,
    how: str = "last",
    lag: int = 0,
    ffill: int | None = None,
) -> pd.Series:
    """Convert a dated series to a contiguous monthly PeriodIndex.

    Parameters
    ----------
    series:
        Values indexed by a DatetimeIndex (or PeriodIndex).  Low-frequency
        observations are assumed dated at the start of their reference period
        (the FRED convention), so a quarterly value dated 2024-04-01 refers to
        Q2 and is placed in June 2024.
    how:
        Aggregation for daily/weekly data: ``"last"`` (end of month) or
        ``"mean"``.
    lag:
        Publication lag in months.  The value for reference month ``m`` is
        treated as known at the end of ``m + lag``.
    ffill:
        Months to carry a value forward.  Defaults to one period minus one
        month for quarterly/annual data and zero otherwise.
    """
    s = series.dropna()
    if s.empty:
        return pd.Series(dtype=float, index=pd.PeriodIndex([], freq="M"), name=series.name)
    if isinstance(s.index, pd.PeriodIndex):
        s = s.copy()
        s.index = s.index.to_timestamp()
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    freq = infer_frequency(pd.DatetimeIndex(s.index))
    if freq == "D":
        grouped = s.groupby(s.index.to_period("M"))
        out = grouped.last() if how == "last" else grouped.mean()
    else:
        out = s.copy()
        out.index = s.index.to_period("M") + (_PERIOD_MONTHS[freq] - 1)
        out = out[~out.index.duplicated(keep="last")]
    out = out.astype(float)
    if lag:
        out.index = out.index + lag
    if ffill is None:
        ffill = _PERIOD_MONTHS.get(freq, 1) - 1
    # Extend past the last observation so a quarterly value stays live until
    # the next release would be due.
    full = pd.period_range(out.index.min(), out.index.max() + ffill, freq="M")
    out = out.reindex(full)
    if ffill:
        out = out.ffill(limit=ffill)
    out.name = series.name
    return out


def monthly_index(start, end) -> pd.PeriodIndex:
    return pd.period_range(pd.Period(start, "M"), pd.Period(end, "M"), freq="M")


def align(frames: dict[str, pd.Series | pd.DataFrame], index: pd.PeriodIndex) -> pd.DataFrame:
    """Reindex several monthly series/frames onto a common index and concat."""
    parts = []
    for name, obj in frames.items():
        if obj is None:
            continue
        if isinstance(obj, pd.Series):
            obj = obj.rename(name).to_frame()
        parts.append(obj.reindex(index))
    if not parts:
        return pd.DataFrame(index=index)
    return pd.concat(parts, axis=1)


def yoy_log_change(level: pd.Series, periods: int = 12) -> pd.Series:
    """12-month log change (decimal), e.g. year-on-year inflation from a price index."""
    return np.log(level).diff(periods)


def forward_sum(series: pd.Series, horizon: int) -> pd.Series:
    """Sum of ``series[t] ... series[t + horizon - 1]``, stored at ``t``."""
    return series.rolling(horizon, min_periods=horizon).sum().shift(-(horizon - 1))
