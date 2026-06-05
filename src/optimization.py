"""
Portfolio optimization routines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from config import TRADING_DAYS


def annualized_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.cov() * TRADING_DAYS


def portfolio_return(weights: np.ndarray, expected_returns: pd.Series) -> float:
    return float(np.dot(weights, expected_returns))


def portfolio_volatility(weights: np.ndarray, covariance_matrix: pd.DataFrame) -> float:
    cov = covariance_matrix.values
    return float(np.sqrt(weights.T @ cov @ weights))


def _bounds(n: int, long_only: bool = True):
    return tuple((0.0, 1.0) for _ in range(n)) if long_only else tuple((-1.0, 1.0) for _ in range(n))


def _constraints():
    return ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)


def equal_weight(n: int) -> np.ndarray:
    return np.repeat(1 / n, n)


def min_volatility_weights(returns: pd.DataFrame, long_only: bool = True) -> np.ndarray:
    n = returns.shape[1]
    cov = annualized_covariance(returns)
    x0 = equal_weight(n)
    result = minimize(
        lambda w: portfolio_volatility(w, cov),
        x0=x0,
        method="SLSQP",
        bounds=_bounds(n, long_only),
        constraints=_constraints(),
        options={"maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Minimum volatility optimization failed: {result.message}")
    return result.x


def max_sharpe_weights(returns: pd.DataFrame, risk_free_rate: float = 0.0, long_only: bool = True) -> np.ndarray:
    n = returns.shape[1]
    exp_ret = returns.mean() * TRADING_DAYS
    cov = annualized_covariance(returns)
    x0 = equal_weight(n)

    def negative_sharpe(w):
        vol = portfolio_volatility(w, cov)
        if vol == 0:
            return 1e6
        return -((portfolio_return(w, exp_ret) - risk_free_rate) / vol)

    result = minimize(
        negative_sharpe,
        x0=x0,
        method="SLSQP",
        bounds=_bounds(n, long_only),
        constraints=_constraints(),
        options={"maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Max Sharpe optimization failed: {result.message}")
    return result.x


def risk_parity_weights(returns: pd.DataFrame, long_only: bool = True) -> np.ndarray:
    n = returns.shape[1]
    cov = annualized_covariance(returns)
    x0 = equal_weight(n)

    def objective(w):
        cov_values = cov.values
        port_var = w.T @ cov_values @ w
        if port_var <= 0:
            return 1e6
        marginal = cov_values @ w
        rc = w * marginal / port_var
        target = np.repeat(1 / n, n)
        return np.sum((rc - target) ** 2)

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=_bounds(n, long_only),
        constraints=_constraints(),
        options={"maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Risk parity optimization failed: {result.message}")
    return result.x


def efficient_frontier_simulation(returns: pd.DataFrame, n_portfolios: int = 5000, risk_free_rate: float = 0.0) -> pd.DataFrame:
    n = returns.shape[1]
    exp_ret = returns.mean() * TRADING_DAYS
    cov = annualized_covariance(returns)
    rows = []
    rng = np.random.default_rng(42)
    for _ in range(n_portfolios):
        weights = rng.random(n)
        weights /= weights.sum()
        ret = portfolio_return(weights, exp_ret)
        vol = portfolio_volatility(weights, cov)
        sharpe = (ret - risk_free_rate) / vol if vol != 0 else np.nan
        row = {"Return": ret, "Volatility": vol, "Sharpe": sharpe}
        row.update({returns.columns[i]: weights[i] for i in range(n)})
        rows.append(row)
    return pd.DataFrame(rows)


def black_litterman_weights(
    returns: pd.DataFrame,
    market_weights: np.ndarray | None = None,
    views: dict[str, float] | None = None,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
) -> np.ndarray:
    """
    Simplified Black-Litterman allocation.

    views maps ticker -> expected annual excess return view.
    This is intentionally simplified for educational and interview purposes.
    """
    tickers = list(returns.columns)
    n = len(tickers)
    cov = annualized_covariance(returns).values
    if market_weights is None:
        market_weights = equal_weight(n)
    market_weights = np.asarray(market_weights)

    implied_returns = risk_aversion * cov @ market_weights

    if not views:
        posterior = implied_returns
    else:
        P = []
        Q = []
        for ticker, view_return in views.items():
            if ticker in tickers:
                row = np.zeros(n)
                row[tickers.index(ticker)] = 1.0
                P.append(row)
                Q.append(float(view_return))
        if not P:
            posterior = implied_returns
        else:
            P = np.vstack(P)
            Q = np.asarray(Q)
            omega = np.diag(np.diag(P @ (tau * cov) @ P.T))
            inv_tau_cov = np.linalg.inv(tau * cov)
            middle = inv_tau_cov + P.T @ np.linalg.inv(omega) @ P
            rhs = inv_tau_cov @ implied_returns + P.T @ np.linalg.inv(omega) @ Q
            posterior = np.linalg.solve(middle, rhs)

    inv_cov = np.linalg.pinv(cov)
    raw = inv_cov @ posterior / risk_aversion
    raw = np.clip(raw, 0, None)
    if raw.sum() == 0:
        return equal_weight(n)
    return raw / raw.sum()


def esg_constrained_weights(
    returns: pd.DataFrame,
    esg_scores: pd.Series,
    min_esg_score: float,
    risk_free_rate: float = 0.0,
) -> np.ndarray:
    """Max Sharpe optimization with a minimum weighted ESG score constraint."""
    n = returns.shape[1]
    exp_ret = returns.mean() * TRADING_DAYS
    cov = annualized_covariance(returns)
    scores = esg_scores.reindex(returns.columns).fillna(esg_scores.mean()).values
    x0 = equal_weight(n)

    def negative_sharpe(w):
        vol = portfolio_volatility(w, cov)
        if vol == 0:
            return 1e6
        return -((portfolio_return(w, exp_ret) - risk_free_rate) / vol)

    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: np.dot(w, scores) - min_esg_score},
    )
    result = minimize(
        negative_sharpe,
        x0=x0,
        method="SLSQP",
        bounds=_bounds(n, True),
        constraints=constraints,
        options={"maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"ESG constrained optimization failed: {result.message}")
    return result.x
