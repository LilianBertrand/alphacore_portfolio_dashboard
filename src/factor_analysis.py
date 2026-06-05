"""
Light factor analysis module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from config import TRADING_DAYS


def run_factor_regression(portfolio_returns: pd.Series, factor_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Regress portfolio returns against selected factor proxies.
    Factor proxies can be ETFs such as SPY, IWM, QQQ, VLUE, MTUM, USMV.
    """
    data = pd.concat([portfolio_returns.rename("Portfolio"), factor_returns], axis=1).dropna()
    if data.empty or data.shape[1] < 2:
        return pd.DataFrame()

    y = data["Portfolio"].values
    X = data.drop(columns="Portfolio").values
    model = LinearRegression().fit(X, y)
    annual_alpha = model.intercept_ * TRADING_DAYS
    r2 = model.score(X, y)

    rows = [{"Factor": "Alpha", "Exposure": annual_alpha}]
    rows.extend({"Factor": col, "Exposure": coef} for col, coef in zip(data.drop(columns="Portfolio").columns, model.coef_))
    rows.append({"Factor": "R-squared", "Exposure": r2})
    return pd.DataFrame(rows).set_index("Factor")


def default_factor_proxies() -> dict[str, str]:
    return {
        "Market": "SPY",
        "Growth/Tech": "QQQ",
        "Small Cap": "IWM",
        "Value": "VLUE",
        "Momentum": "MTUM",
        "Low Volatility": "USMV",
    }
