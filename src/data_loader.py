"""
Market data loading utilities.
"""
from __future__ import annotations

from typing import Iterable
import pandas as pd
import yfinance as yf


def clean_ticker_list(tickers: str | Iterable[str]) -> list[str]:
    """Convert a comma-separated string or iterable into a clean ticker list."""
    if isinstance(tickers, str):
        items = tickers.replace(";", ",").split(",")
    else:
        items = list(tickers)
    return [str(t).strip().upper() for t in items if str(t).strip()]


def download_adjusted_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Download adjusted close prices from Yahoo Finance.

    Parameters
    ----------
    tickers:
        List of tickers.
    start:
        Start date in YYYY-MM-DD format.
    end:
        Optional end date.

    Returns
    -------
    pd.DataFrame
        Adjusted close prices indexed by date.
    """
    if not tickers:
        raise ValueError("At least one ticker is required.")

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    if data.empty:
        raise ValueError("No market data returned. Check tickers and date range.")

    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            prices = data["Adj Close"].copy()
        elif "Close" in data.columns.get_level_values(0):
            prices = data["Close"].copy()
        else:
            raise ValueError("Could not find adjusted close or close prices.")
    else:
        col = "Adj Close" if "Adj Close" in data.columns else "Close"
        prices = data[[col]].copy()
        prices.columns = tickers[:1]

    prices = prices.dropna(axis=1, how="all").ffill().dropna()
    if prices.shape[1] == 0:
        raise ValueError("All tickers returned empty price series.")
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily returns from prices."""
    return prices.pct_change().dropna(how="all")
