#  Financial Market Analytics Project
## Data Analytics Portfolio Project

---

##  Project Overview

**"Financial Market Risk Analytics & Algorithmic Signal Generation System"**

A production-grade, end-to-end quantitative analytics project that mirrors the exact
work done by .' FAST (Franchise Analytics Strategy & Technology) team
and Quantitative Strategists.

---

##  Project Structure

```
goldman_sachs_project/
│
├── main.py               ←RUN THIS — Full pipeline orchestrator
├── data_generator.py     ← Geometric Brownian Motion market simulation
├── analytics_engine.py   ← Risk metrics, portfolio optimization, charts
├── ml_models.py          ← ML models, feature engineering, backtest
├── sql_analytics.py      ← SQL-style financial queries (CTEs, windows)
├── requirements.txt      ← Python dependencies
├── README.md             ← This file
│
├── data/                 ← Generated CSV files
└── charts/               ← Generated PNG visualizations
```

---

##  Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python main.py
```

---

##  Technologies & Methods

| Category | Technology / Method |
|---|---|
| Language | Python 3.10+ |
| Data Manipulation | Pandas, NumPy |
| Statistical Analysis | SciPy (normality tests, z-scores) |
| Machine Learning | Scikit-learn (Random Forest, Isolation Forest) |
| Visualization | Matplotlib, Seaborn |
| Stochastic Modeling | Geometric Brownian Motion (GBM) |
| Risk Analytics | VaR, CVaR, Drawdown, Beta, Alpha |
| Portfolio Theory | Markowitz Mean-Variance, Efficient Frontier |
| Technical Analysis | RSI, MACD, Bollinger Bands, ATR |
| Validation | Walk-Forward Cross-Validation (Time-Series safe) |
| SQL Concepts | CTEs, Window Functions, GROUP BY, CASE WHEN |

---

##  Key Analyses

### 1. Risk Analytics
- **Value at Risk (VaR)** at 95% and 99% confidence — daily and portfolio
- **Conditional VaR / Expected Shortfall** — Basel III standard
- **Maximum Drawdown** — peak-to-trough analysis
- **Volatility Regime Detection** — high/low vol regimes

### 2. Performance Metrics
- **Sharpe Ratio** — return per unit of risk
- **Sortino Ratio** — downside-adjusted return
- **Calmar Ratio** — return per unit of drawdown
- **Jensen's Alpha** — outperformance vs CAPM prediction
- **Beta** — systematic market risk exposure

### 3. Portfolio Optimization
- **Markowitz Efficient Frontier** — 60-point curve
- **Maximum Sharpe Portfolio** — optimal risk-adjusted allocation
- **Minimum Variance Portfolio** — lowest risk allocation
- Weight constraints: 2%–40% per stock

### 4. Machine Learning
- **Feature Engineering**: 20+ technical indicators as features
- **Random Forest Classifier**: 5-day return direction prediction
- **Walk-Forward Validation**: No future data leakage (critical!)
- **ROC-AUC evaluation**: Industry standard for signal quality
- **Isolation Forest**: Anomaly detection in price/volume
- **Strategy Backtest**: ML signal vs Buy-and-Hold

### 5. SQL Analytics
- Monthly returns aggregation (GROUP BY equivalent)
- Rolling Sharpe ranking (Window Functions)
- Volatility regime classification (CASE WHEN)
- Pair correlation CTEs
- P&L attribution (LAG + cumulative SUM)
- Z-score outlier detection (3-sigma events)

### 6. Monte Carlo Simulation
- 500 paths, 60-day horizon for GS stock
- Confidence interval bands (5th / 50th / 95th percentile)
- Same GBM model used in Black-Scholes pricing



##  Output Files

| File | Description |
|---|---|
| `charts/01_price_dashboard.png` | 5-panel: prices, distributions, correlation, Sharpe, drawdowns |
| `charts/02_risk_dashboard.png` | VaR/CVaR bars, Monte Carlo, volatility regimes, stress test |
| `charts/03_portfolio_optimization.png` | Efficient frontier with optimal portfolios |
| `charts/03_ml_signals.png` | Feature importance, ROC curve, RSI, MACD, anomalies |
| `charts/04_backtest.png` | ML strategy vs Buy-and-Hold |
| `data/risk_report.csv` | Full risk metrics for all tickers |
| `data/sql_monthly_summary.csv` | Monthly aggregated returns |
| `data/sql_pnl_attribution.csv` | P&L attribution per stock |
| `data/sql_zscore_outliers.csv` | 3-sigma market anomaly events |
| `reports/final_report.txt` | Complete analytical report |

---

