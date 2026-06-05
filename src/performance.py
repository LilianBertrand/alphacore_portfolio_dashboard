"""
Performance analytics for portfolios and assets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from config import TRADING_DAYS


def annualized_return(returns: pd.Series | pd.DataFrame) -> pd.Series | float:
    """Annualized arithmetic return estimate."""
    result = returns.mean() * TRADING_DAYS
    return result


def annualized_volatility(returns: pd.Series | pd.DataFrame) -> pd.Series | float:
    """Annualized volatility estimate."""
    return returns.std() * np.sqrt(TRADING_DAYS)


def cumulative_returns(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Cumulative wealth index starting at 1."""
    return (1 + returns).cumprod()


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio."""
    excess_daily = returns - risk_free_rate / TRADING_DAYS
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return float(excess_daily.mean() * TRADING_DAYS / vol)


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sortino ratio using downside volatility."""
    excess_daily = returns - risk_free_rate / TRADING_DAYS
    downside = excess_daily[excess_daily < 0]
    downside_vol = downside.std() * np.sqrt(TRADING_DAYS)
    if downside_vol == 0 or np.isnan(downside_vol):
        return np.nan
    return float(excess_daily.mean() * TRADING_DAYS / downside_vol)


def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Compute portfolio returns from asset returns and weights."""
    weights = np.asarray(weights, dtype=float)
    return returns @ weights


def performance_table(returns: pd.DataFrame, risk_free_rate: float = 0.0) -> pd.DataFrame:
    """Build an asset-level performance table."""
    rows = []
    for col in returns.columns:
        r = returns[col].dropna()
        rows.append({
            "Asset": col,
            "Annual Return": annualized_return(r),
            "Annual Volatility": annualized_volatility(r),
            "Sharpe Ratio": sharpe_ratio(r, risk_free_rate),
            "Sortino Ratio": sortino_ratio(r, risk_free_rate),
        })
    return pd.DataFrame(rows).set_index("Asset")
