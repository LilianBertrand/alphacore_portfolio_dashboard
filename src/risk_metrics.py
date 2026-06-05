"""
Risk analytics for investment portfolios.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from config import TRADING_DAYS


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown from a return series."""
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1
    return float(drawdown.min())


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Full drawdown series."""
    wealth = (1 + returns).cumprod()
    return wealth / wealth.cummax() - 1


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Value at Risk as a negative return threshold."""
    return float(np.percentile(returns.dropna(), (1 - confidence) * 100))


def historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Conditional VaR: average return below the VaR threshold."""
    var = historical_var(returns, confidence)
    tail = returns.dropna()[returns.dropna() <= var]
    return float(tail.mean()) if len(tail) else np.nan


def rolling_volatility(returns: pd.Series, window: int = 63) -> pd.Series:
    """Rolling annualized volatility."""
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def rolling_sharpe(returns: pd.Series, risk_free_rate: float = 0.0, window: int = 63) -> pd.Series:
    """Rolling annualized Sharpe ratio."""
    excess = returns - risk_free_rate / TRADING_DAYS
    roll_ret = excess.rolling(window).mean() * TRADING_DAYS
    roll_vol = returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
    return roll_ret / roll_vol


def beta_alpha(portfolio_returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.0) -> tuple[float, float]:
    """Compute annualized CAPM beta and alpha versus benchmark."""
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ["portfolio", "benchmark"]
    if aligned.empty:
        return np.nan, np.nan
    cov = aligned.cov().iloc[0, 1]
    var = aligned["benchmark"].var()
    beta = cov / var if var != 0 else np.nan
    p_ret = aligned["portfolio"].mean() * TRADING_DAYS
    b_ret = aligned["benchmark"].mean() * TRADING_DAYS
    alpha = p_ret - (risk_free_rate + beta * (b_ret - risk_free_rate))
    return float(beta), float(alpha)


def tracking_error(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Annualized tracking error."""
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return float(active.std() * np.sqrt(TRADING_DAYS))


def information_ratio(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Annualized information ratio."""
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    te = active.std() * np.sqrt(TRADING_DAYS)
    if te == 0 or np.isnan(te):
        return np.nan
    return float(active.mean() * TRADING_DAYS / te)


def risk_contribution(weights: np.ndarray, covariance_matrix: pd.DataFrame) -> pd.Series:
    """Percentage contribution of each asset to portfolio variance."""
    w = np.asarray(weights, dtype=float)
    cov = covariance_matrix.values
    portfolio_var = float(w.T @ cov @ w)
    if portfolio_var <= 0:
        return pd.Series(np.nan, index=covariance_matrix.index)
    marginal = cov @ w
    contribution = w * marginal / portfolio_var
    return pd.Series(contribution, index=covariance_matrix.index)
