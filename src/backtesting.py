"""
Backtesting utilities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rebalance_backtest(returns: pd.DataFrame, target_weights: np.ndarray, rebalance_frequency: str = "M") -> pd.Series:
    """
    Simple calendar rebalanced backtest.

    The strategy resets to target weights at each period start according to the chosen frequency.
    """
    target_weights = np.asarray(target_weights, dtype=float)
    rebal_dates = returns.resample(rebalance_frequency).first().index
    weights = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    current_weights = target_weights.copy()

    for date in returns.index:
        if date in rebal_dates:
            current_weights = target_weights.copy()
        weights.loc[date] = current_weights
        day_ret = returns.loc[date].fillna(0).values
        current_weights = current_weights * (1 + day_ret)
        if current_weights.sum() != 0:
            current_weights = current_weights / current_weights.sum()

    return (weights.shift(1).fillna(pd.Series(target_weights, index=returns.columns)) * returns).sum(axis=1)
