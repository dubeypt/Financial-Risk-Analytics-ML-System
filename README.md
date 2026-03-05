# 📈 Financial Market Risk Analytics & ML Signal System

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-Data-lightblue?style=for-the-badge&logo=pandas)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

> Production-grade, end-to-end quantitative analytics system — VaR, CVaR, Monte Carlo Simulation, ML Signal Generation & Portfolio Optimization across 8 financial stocks (504 trading days).

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| 📦 Stocks Analyzed | 8 (504 trading days) |
| 📈 Max Sharpe Ratio | **0.499** |
| 🤖 ML AUC Score | **0.585** (Walk-Forward CV) |
| 🎲 Monte Carlo Paths | 500 (60-day horizon) |
| 💼 Portfolio Return | **+37.5%** vs SPY +9.5% |
| 🛡️ VaR Confidence | 95% & 99% |
| 🔢 Efficient Frontier | 60 portfolios |

---

## 🗂️ Project Structure

```
Financial-Risk-Analytics-ML-System/
│
├── main.py               ← RUN THIS — Full pipeline orchestrator
├── data_generator.py     ← Geometric Brownian Motion market simulation
├── analytics_engine.py   ← Risk metrics, portfolio optimization, charts
├── ml_models.py          ← ML models, feature engineering, backtest
├── sql_analytics.py      ← SQL-style financial queries (CTEs, windows)
├── streamlit_app.py      ← Interactive 6-tab Streamlit dashboard
├── requirements.txt      ← Python dependencies
├── README.md             ← This file
│
├── data/                 ← Generated CSV files
└── charts/               ← Generated PNG visualizations
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/dubeypt/Financial-Risk-Analytics-ML-System.git
cd Financial-Risk-Analytics-ML-System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full pipeline
python main.py

# 4. Launch interactive dashboard
streamlit run streamlit_app.py
```

---

## 🛠️ Technologies & Methods

| Category | Technology / Method |
|----------|-------------------|
| Language | Python 3.10+ |
| Data Manipulation | Pandas, NumPy |
| Statistical Analysis | SciPy (normality tests, z-scores) |
| Machine Learning | Scikit-learn (Random Forest, Isolation Forest) |
| Visualization | Matplotlib, Seaborn, Streamlit |
| Stochastic Modeling | Geometric Brownian Motion (GBM) |
| Risk Analytics | VaR, CVaR, Drawdown, Beta, Alpha |
| Portfolio Theory | Markowitz Mean-Variance, Efficient Frontier |
| Technical Analysis | RSI, MACD, Bollinger Bands, ATR |
| Validation | Walk-Forward Cross-Validation (Time-Series safe) |
| SQL Concepts | CTEs, Window Functions, GROUP BY, CASE WHEN |

---

## 🔍 Key Analyses

### 1. 🛡️ Risk Analytics
- **Value at Risk (VaR)** at 95% and 99% confidence — daily and portfolio
- **Conditional VaR / Expected Shortfall** — Basel III standard
- **Maximum Drawdown** — peak-to-trough analysis
- **Volatility Regime Detection** — high/low vol regimes

### 2. 📊 Performance Metrics
- **Sharpe Ratio** — return per unit of risk
- **Sortino Ratio** — downside-adjusted return
- **Calmar Ratio** — return per unit of drawdown
- **Jensen's Alpha** — outperformance vs CAPM prediction
- **Beta** — systematic market risk exposure

### 3. 💼 Portfolio Optimization
- **Markowitz Efficient Frontier** — 60-point curve
- **Maximum Sharpe Portfolio** — optimal risk-adjusted allocation
- **Minimum Variance Portfolio** — lowest risk allocation
- Weight constraints: 2%–40% per stock

### 4. 🤖 Machine Learning
- **Feature Engineering**: 20+ technical indicators as features
- **Random Forest Classifier**: 5-day return direction prediction
- **Walk-Forward Validation**: No future data leakage (critical!)
- **ROC-AUC evaluation**: Industry standard for signal quality
- **Isolation Forest**: Anomaly detection in price/volume
- **Strategy Backtest**: ML signal vs Buy-and-Hold

### 5. 🗄️ SQL Analytics
- Monthly returns aggregation (GROUP BY equivalent)
- Rolling Sharpe ranking (Window Functions)
- Volatility regime classification (CASE WHEN)
- Pair correlation CTEs
- P&L attribution (LAG + cumulative SUM)
- Z-score outlier detection (3-sigma events)

### 6. 🎲 Monte Carlo Simulation
- 500 paths, 60-day horizon for GS stock
- Confidence interval bands (5th / 50th / 95th percentile)
- Same GBM model used in Black-Scholes pricing

---

## 📁 Output Files

| File | Description |
|------|-------------|
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

## 👤 Author

**Aditya Dubey** — Data Analyst | Bengaluru, India

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/pt-adityadubey)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/dubeypt)
[![Email](https://img.shields.io/badge/Email-Contact-red?style=flat&logo=gmail)](mailto:ptaddubey@gmail.com)
