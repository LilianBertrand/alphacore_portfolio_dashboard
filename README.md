# AlphaCore — Portfolio Optimization & Risk Analytics Dashboard

AlphaCore is a Python-based multi-asset portfolio analytics dashboard designed for asset management, portfolio construction and risk monitoring.

The project demonstrates how a buy-side analyst or portfolio manager can use Python to transform market data into a complete allocation and risk framework. It combines performance analysis, risk metrics, optimization methods, benchmark comparison, ESG constraints and factor analysis in a clean Streamlit dashboard.

## Why this project is relevant for Asset Management

This project is designed to show practical skills expected in asset management and investment analysis:

- Python for financial data analysis
- Portfolio construction and allocation methods
- Risk-adjusted performance measurement
- Benchmark comparison
- Risk monitoring and drawdown analysis
- Optimization under constraints
- Factor exposure analysis
- ESG-aware portfolio construction
- Interactive dashboard development

## Main Features

### Market Data

- Yahoo Finance data import through `yfinance`
- Multi-asset universe support
- Daily returns calculation
- Cumulative performance analysis
- Correlation matrix
- Asset-level performance table

### Portfolio Construction

The dashboard includes several allocation methodologies:

1. **Equal Weight**  
   Simple diversification baseline.

2. **Minimum Volatility**  
   Allocates capital to minimize total portfolio volatility.

3. **Maximum Sharpe Ratio**  
   Maximizes expected excess return per unit of risk.

4. **Risk Parity**  
   Balances the contribution of each asset to total portfolio risk.

5. **Simplified Black-Litterman**  
   Combines market-implied expected returns with manager views.

6. **ESG-Constrained Max Sharpe**  
   Optimizes the portfolio while respecting a minimum weighted ESG score.

### Risk Analytics

- Annualized volatility
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Historical Value at Risk
- Historical Conditional Value at Risk
- Rolling volatility
- Rolling Sharpe ratio
- Risk contribution by asset

### Benchmark Analytics

- Portfolio versus benchmark cumulative performance
- Beta
- Annualized alpha
- Tracking error
- Information ratio

### Factor Analysis

A light factor regression module estimates exposure to ETF proxies such as:

- Market
- Growth / Technology
- Small Cap
- Value
- Momentum
- Low Volatility

## Project Architecture

```text
alphacore-portfolio-dashboard/
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── performance.py
│   ├── risk_metrics.py
│   ├── optimization.py
│   ├── factor_analysis.py
│   ├── backtesting.py
│   └── visualization.py
│
├── notebooks/
├── reports/
└── assets/
```

## Suggested Universe

The default portfolio universe is:

```text
SPY, QQQ, EFA, EEM, TLT, IEF, LQD, GLD, VNQ
```

This creates a diversified multi-asset portfolio with exposure to:

- US equities
- Technology / growth equities
- Developed markets
- Emerging markets
- Long-term Treasuries
- Intermediate Treasuries
- Corporate bonds
- Gold
- Real estate


## Technical Skills Demonstrated

- Python
- pandas
- numpy
- scipy optimization
- yfinance
- plotly
- Streamlit
- scikit-learn regression
- Financial performance analytics
- Portfolio optimization
- Risk management

## Live DEMO 
https://alphacoreportfoliodashboard-qzzgmhzpwdtqbwvku7r8ec.streamlit.app/

