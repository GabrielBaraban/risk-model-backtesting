"""Data loading for risk-model-backtesting.

Conventions used throughout this project:

- All returns are 1-day **log** returns: ``r_t = log(P_t / P_{t-1})``.
  Log returns can be negative or positive; a loss is a negative return.
- ``alpha`` is the tail probability of a VaR/ES model (e.g. 0.05 for a 95%
  VaR). VaR and ES are always reported as **positive** loss magnitudes:

    VaR = -quantile_alpha(returns)
    ES  = -E[return | return <= quantile_alpha(returns)]

  i.e. a VaR of 0.023 means "the model expects a 2.3% loss to be exceeded
  with probability alpha on the next day." Never a negative number.
- Each model in ``src/models.py`` takes a ``window`` array of past returns
  and forecasts the *next* day's (VaR, ES) — no lookahead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_returns(
    ticker: str,
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.Series:
    """Load daily 1-day log returns for a ticker between start and end.

    Downloads adjusted-close prices via yfinance (or reads a local cache
    at ``data/<ticker>.csv`` if present and ``use_cache`` is True), then
    converts to log returns.

    Args:
        ticker: e.g. "SPY".
        start: ISO date string, e.g. "2015-01-01".
        end: ISO date string, e.g. "2024-01-01".
        use_cache: if True and a cache file exists, read from it instead
            of hitting the network. The cache is always written after a
            fresh download.

    Returns:
        A pandas Series of log returns, indexed by date, named "log_return".
        NaN-free (the first observation, which has no prior price, is
        dropped).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / f"{ticker}.csv"

    if use_cache and cache_path.exists():
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)["adj_close"]
    else:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            timeout=30,
        )
        if raw.empty:
            raise ValueError(
                f"yfinance returned no data for ticker={ticker!r}, "
                f"start={start!r}, end={end!r}"
            )

        # yfinance sometimes returns a MultiIndex column frame even for a
        # single ticker (columns like ("Close", "SPY")); flatten it.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        close_col = "Close" if "Close" in raw.columns else "Adj Close"
        prices = raw[close_col].rename("adj_close")
        prices.to_csv(cache_path, index_label="Date")

    prices = prices.sort_index()
    log_returns = np.log(prices / prices.shift(1)).dropna()
    log_returns.name = "log_return"

    mask = (log_returns.index >= pd.Timestamp(start)) & (
        log_returns.index <= pd.Timestamp(end)
    )
    return log_returns.loc[mask]
