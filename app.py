from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config import DEFAULT_TICKERS, DEFAULT_BENCHMARK, DEFAULT_START_DATE, DEFAULT_RISK_FREE_RATE, TRADING_DAYS
from src.data_loader import clean_ticker_list, download_adjusted_prices, compute_returns
from src.performance import (
    annualized_return,
    annualized_volatility,
    cumulative_returns,
    sharpe_ratio,
    sortino_ratio,
    portfolio_returns,
    performance_table,
)
from src.risk_metrics import (
    max_drawdown,
    drawdown_series,
    historical_var,
    historical_cvar,
    rolling_volatility,
    rolling_sharpe,
    beta_alpha,
    tracking_error,
    information_ratio,
    risk_contribution,
)
from src.optimization import (
    equal_weight,
    min_volatility_weights,
    max_sharpe_weights,
    risk_parity_weights,
    efficient_frontier_simulation,
    black_litterman_weights,
    esg_constrained_weights,
    annualized_covariance,
    portfolio_return,
    portfolio_volatility,
)
from src.factor_analysis import run_factor_regression, default_factor_proxies
from src.visualization import line_chart, heatmap, weights_pie, efficient_frontier_plot

st.set_page_config(
    page_title="AlphaCore Portfolio Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("AlphaCore — Portfolio Optimization & Risk Analytics")
st.caption("A candidature-ready Python dashboard for asset management, portfolio construction and risk monitoring.")

with st.sidebar:
    st.header("Portfolio Universe")
    ticker_input = st.text_area(
        "Tickers",
        value=", ".join(DEFAULT_TICKERS),
        help="Use Yahoo Finance tickers separated by commas.",
    )
    benchmark = st.text_input("Benchmark", value=DEFAULT_BENCHMARK)
    start_date = st.date_input("Start date", value=pd.to_datetime(DEFAULT_START_DATE))
    end_date = st.date_input("End date", value=pd.Timestamp.today())
    risk_free_rate = st.number_input("Risk-free rate", value=DEFAULT_RISK_FREE_RATE, step=0.005, format="%.3f")
    optimization_method = st.selectbox(
        "Allocation method",
        [
            "Equal Weight",
            "Minimum Volatility",
            "Maximum Sharpe",
            "Risk Parity",
            "Black-Litterman Simplified",
            "ESG-Constrained Max Sharpe",
        ],
    )
    confidence = st.slider("VaR / CVaR confidence", 0.90, 0.99, 0.95, 0.01)
    n_frontier = st.slider("Efficient frontier portfolios", 500, 10000, 3000, 500)

all_tickers = clean_ticker_list(ticker_input)
if benchmark.upper() not in all_tickers:
    download_tickers = all_tickers + [benchmark.upper()]
else:
    download_tickers = all_tickers

@st.cache_data(show_spinner=False)
def load_data(tickers, start, end):
    prices = download_adjusted_prices(tickers, str(start), str(end))
    returns = compute_returns(prices)
    return prices, returns

try:
    prices_all, returns_all = load_data(download_tickers, start_date, end_date)
except Exception as exc:
    st.error(f"Data loading failed: {exc}")
    st.stop()

available_assets = [t for t in all_tickers if t in returns_all.columns]
if len(available_assets) < 2:
    st.error("At least two valid assets are required for portfolio optimization.")
    st.stop()

returns = returns_all[available_assets].dropna()
prices = prices_all[available_assets]
benchmark_returns = returns_all[benchmark.upper()].dropna() if benchmark.upper() in returns_all.columns else returns.mean(axis=1)

# ESG demo scores. In a professional setup these would come from MSCI/Sustainalytics/refinitiv/etc.
def make_demo_esg_scores(cols):
    base = np.linspace(55, 85, len(cols))
    rng = np.random.default_rng(7)
    scores = base + rng.normal(0, 4, len(cols))
    return pd.Series(np.clip(scores, 0, 100), index=cols, name="ESG Score")

esg_scores = make_demo_esg_scores(returns.columns)

try:
    if optimization_method == "Equal Weight":
        weights = equal_weight(returns.shape[1])
    elif optimization_method == "Minimum Volatility":
        weights = min_volatility_weights(returns)
    elif optimization_method == "Maximum Sharpe":
        weights = max_sharpe_weights(returns, risk_free_rate)
    elif optimization_method == "Risk Parity":
        weights = risk_parity_weights(returns)
    elif optimization_method == "Black-Litterman Simplified":
        views = {returns.columns[0]: 0.09}
        weights = black_litterman_weights(returns, views=views)
    else:
        min_esg = st.sidebar.slider("Minimum weighted ESG score", 50, 90, 70)
        weights = esg_constrained_weights(returns, esg_scores, min_esg, risk_free_rate)
except Exception as exc:
    st.warning(f"Optimization failed, falling back to Equal Weight. Details: {exc}")
    weights = equal_weight(returns.shape[1])

weights_series = pd.Series(weights, index=returns.columns, name="Weight")
p_ret = portfolio_returns(returns, weights)
p_cum = cumulative_returns(p_ret)
b_cum = cumulative_returns(benchmark_returns.reindex(p_ret.index).dropna())

aligned = pd.concat([p_ret.rename("Portfolio"), benchmark_returns.rename("Benchmark")], axis=1).dropna()
metric_cols = st.columns(6)
metric_cols[0].metric("Annual Return", f"{annualized_return(p_ret):.2%}")
metric_cols[1].metric("Annual Volatility", f"{annualized_volatility(p_ret):.2%}")
metric_cols[2].metric("Sharpe", f"{sharpe_ratio(p_ret, risk_free_rate):.2f}")
metric_cols[3].metric("Sortino", f"{sortino_ratio(p_ret, risk_free_rate):.2f}")
metric_cols[4].metric("Max Drawdown", f"{max_drawdown(p_ret):.2%}")
metric_cols[5].metric("VaR", f"{historical_var(p_ret, confidence):.2%}")

tabs = st.tabs([
    "Market Overview",
    "Portfolio Builder",
    "Efficient Frontier",
    "Risk Dashboard",
    "Benchmark & Factors",
    "ESG Module",
])

with tabs[0]:
    st.subheader("Market Data Overview")
    st.plotly_chart(line_chart(cumulative_returns(returns), "Asset Cumulative Performance"), use_container_width=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(performance_table(returns, risk_free_rate).style.format("{:.2%}"), use_container_width=True)
    with c2:
        st.plotly_chart(heatmap(returns.corr(), "Asset Correlation Matrix"), use_container_width=True)

with tabs[1]:
    st.subheader("Portfolio Construction")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(weights_pie(weights_series, f"{optimization_method} Allocation"), use_container_width=True)
        st.dataframe(weights_series.to_frame().style.format("{:.2%}"), use_container_width=True)
    with c2:
        st.plotly_chart(line_chart(pd.DataFrame({"Portfolio": p_cum}), "Portfolio Cumulative Performance"), use_container_width=True)
        summary = pd.DataFrame({
            "Metric": ["Annual Return", "Annual Volatility", "Sharpe", "Sortino", "Max Drawdown", "VaR", "CVaR"],
            "Value": [
                annualized_return(p_ret),
                annualized_volatility(p_ret),
                sharpe_ratio(p_ret, risk_free_rate),
                sortino_ratio(p_ret, risk_free_rate),
                max_drawdown(p_ret),
                historical_var(p_ret, confidence),
                historical_cvar(p_ret, confidence),
            ],
        })
        st.dataframe(summary, use_container_width=True)

with tabs[2]:
    st.subheader("Efficient Frontier")
    frontier = efficient_frontier_simulation(returns, n_frontier, risk_free_rate)
    cov = annualized_covariance(returns)
    exp_ret = returns.mean() * TRADING_DAYS
    selected = {
        optimization_method: (
            portfolio_volatility(weights, cov),
            portfolio_return(weights, exp_ret),
        )
    }
    st.plotly_chart(efficient_frontier_plot(frontier, selected), use_container_width=True)
    st.caption("The frontier is simulated with long-only random portfolios. It is useful for intuition and communication, not as a guarantee of future performance.")

with tabs[3]:
    st.subheader("Risk Dashboard")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(line_chart(pd.DataFrame({"Drawdown": drawdown_series(p_ret)}), "Portfolio Drawdown"), use_container_width=True)
        st.plotly_chart(line_chart(pd.DataFrame({"Rolling Volatility": rolling_volatility(p_ret)}), "Rolling Annualized Volatility"), use_container_width=True)
    with c2:
        st.plotly_chart(line_chart(pd.DataFrame({"Rolling Sharpe": rolling_sharpe(p_ret, risk_free_rate)}), "Rolling Sharpe Ratio"), use_container_width=True)
        rc = risk_contribution(weights, cov).rename("Risk Contribution")
        st.dataframe(rc.to_frame().style.format("{:.2%}"), use_container_width=True)

with tabs[4]:
    st.subheader("Benchmark Comparison & Factor Analysis")
    beta, alpha = beta_alpha(aligned["Portfolio"], aligned["Benchmark"], risk_free_rate) if not aligned.empty else (np.nan, np.nan)
    comp = pd.DataFrame({
        "Metric": ["Beta", "Annual Alpha", "Tracking Error", "Information Ratio"],
        "Value": [beta, alpha, tracking_error(aligned["Portfolio"], aligned["Benchmark"]), information_ratio(aligned["Portfolio"], aligned["Benchmark"])] if not aligned.empty else [np.nan]*4,
    })
    st.dataframe(comp, use_container_width=True)
    comparison = pd.concat([
        cumulative_returns(aligned["Portfolio"]).rename("Portfolio"),
        cumulative_returns(aligned["Benchmark"]).rename(benchmark.upper()),
    ], axis=1)
    st.plotly_chart(line_chart(comparison, "Portfolio vs Benchmark"), use_container_width=True)

    with st.expander("Run light factor analysis with ETF proxies"):
        proxy_map = default_factor_proxies()
        st.write(proxy_map)
        factor_tickers = list(proxy_map.values())
        try:
            factor_prices, factor_returns = load_data(factor_tickers, start_date, end_date)
            factor_returns.columns = list(proxy_map.keys())
            factor_table = run_factor_regression(p_ret, factor_returns)
            st.dataframe(factor_table, use_container_width=True)
        except Exception as exc:
            st.warning(f"Factor analysis unavailable: {exc}")

with tabs[5]:
    st.subheader("ESG-Constrained Allocation Module")
    st.caption("Demo ESG scores are synthetic. In a professional version, connect a licensed ESG data provider.")
    esg_table = pd.concat([weights_series, esg_scores], axis=1)
    esg_table["Weighted ESG Contribution"] = esg_table["Weight"] * esg_table["ESG Score"]
    weighted_score = float(esg_table["Weighted ESG Contribution"].sum())
    st.metric("Portfolio ESG Score", f"{weighted_score:.1f} / 100")
    st.dataframe(esg_table.style.format({"Weight": "{:.2%}", "ESG Score": "{:.1f}", "Weighted ESG Contribution": "{:.2f}"}), use_container_width=True)

st.divider()
st.caption("Educational project. Historical data and optimization outputs are not investment advice and do not predict future returns.")
