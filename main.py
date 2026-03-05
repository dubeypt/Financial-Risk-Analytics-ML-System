

import sys, os, time, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
os.makedirs("data", exist_ok=True)
os.makedirs("charts", exist_ok=True)
os.makedirs("reports", exist_ok=True)


def banner(msg, color_code="\033[94m"):
    RESET = "\033[0m"
    BOLD  = "\033[1m"
    w = 64
    print(f"\n{color_code}{BOLD}{'═'*w}{RESET}")
    print(f"{color_code}{BOLD}  {msg}{RESET}")
    print(f"{color_code}{BOLD}{'═'*w}{RESET}")


def step(msg):
    print(f"\n  \033[93m▸\033[0m {msg}")


def ok(msg):
    print(f"    \033[92m✓\033[0m {msg}")


# ══════════════════════════════════════════════════════════════════
# STEP 1 — DATA GENERATION
# ══════════════════════════════════════════════════════════════════
banner(". — Financial Analytics Pipeline", "\033[93m")
banner("STEP 1: Generating Market Data (GBM Simulation)", "\033[94m")

from data_generator import main as generate_data
generate_data()

# ══════════════════════════════════════════════════════════════════
# STEP 2 — LOAD DATA
# ══════════════════════════════════════════════════════════════════
banner("STEP 2: Loading & Validating Data", "\033[94m")

close_prices = pd.read_csv("data/close_prices.csv",
                            index_col="date", parse_dates=True)

tickers = close_prices.columns.tolist()
ok(f"Loaded {len(close_prices)} trading days × {len(tickers)} tickers")
ok(f"Date range: {close_prices.index[0].date()} → {close_prices.index[-1].date()}")
ok(f"Tickers: {', '.join(tickers)}")
ok(f"Missing values: {close_prices.isnull().sum().sum()}")

# Load OHLCV for GS (primary stock)
gs_ohlcv = pd.read_csv("data/GS.csv", index_col="date", parse_dates=True)

# ══════════════════════════════════════════════════════════════════
# STEP 3 — ANALYTICS ENGINE
# ══════════════════════════════════════════════════════════════════
banner("STEP 3: Quantitative Analytics & Risk Metrics", "\033[94m")

from analytics_engine import (
    compute_returns, compute_cumulative_returns,
    compute_var, compute_cvar, compute_volatility,
    compute_max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio,
    beta_alpha, optimize_portfolio,
    plot_price_dashboard, plot_risk_dashboard,
    plot_portfolio_optimization, generate_risk_report
)

step("Computing log returns...")
returns = compute_returns(close_prices)
ok(f"Returns shape: {returns.shape}")

step("Computing risk metrics for each ticker...")
risk_report = generate_risk_report(close_prices, returns)
print("\n" + "─" * 70)
print(risk_report.to_string())
print("─" * 70)
ok("Risk report saved → data/risk_report.csv")

step("Running Portfolio Optimization (Markowitz)...")
opt = optimize_portfolio(returns)
ms_r, ms_v, ms_s = opt["max_sharpe_perf"]
mv_r, mv_v, mv_s = opt["min_var_perf"]

print(f"\n  Max Sharpe Portfolio:")
for t, w in opt["max_sharpe_weights"].items():
    print(f"    {t:5s}: {w*100:5.1f}%")
print(f"  → Return={ms_r*100:.2f}%  Vol={ms_v*100:.2f}%  Sharpe={ms_s:.3f}")

print(f"\n  Min Variance Portfolio:")
for t, w in opt["min_var_weights"].items():
    print(f"    {t:5s}: {w*100:5.1f}%")
print(f"  → Return={mv_r*100:.2f}%  Vol={mv_v*100:.2f}%  Sharpe={mv_s:.3f}")

step("Generating charts (1/3): Price Dashboard...")
plot_price_dashboard(close_prices, returns)

step("Generating charts (2/3): Risk Dashboard...")
plot_risk_dashboard(close_prices, returns)

step("Generating charts (3/3): Portfolio Optimization...")
plot_portfolio_optimization(opt)

# ══════════════════════════════════════════════════════════════════
# STEP 4 — MACHINE LEARNING
# ══════════════════════════════════════════════════════════════════
banner("STEP 4: Machine Learning Models", "\033[94m")

from ml_models import (build_features, walk_forward_train,
                       plot_ml_results, backtest_strategy, plot_backtest)

step("Building feature matrix (technical indicators)...")
features = build_features(gs_ohlcv)
ok(f"Features: {features.shape[1]-1} features × {len(features)} samples")

step("Training Random Forest with Walk-Forward Validation...")
model_results = walk_forward_train(features, n_splits=5)

folds = model_results["fold_results"]
print(f"\n  Walk-Forward Results:")
for _, row in folds.iterrows():
    status = "✓" if row["auc"] > 0.55 else "~"
    print(f"    Fold {int(row['fold'])}: AUC={row['auc']:.4f}  "
          f"Accuracy={row['accuracy']*100:.1f}%  {status}")
print(f"\n  Final AUC: {model_results['final_auc']:.4f}")
ok("Model trained successfully")

step("Backtesting ML trading strategy...")
bt = backtest_strategy(features, close_prices["GS"], model_results)
print(f"\n  Strategy vs Buy-and-Hold:")
print(f"    ML Strategy Final: ${bt['final_strategy_value']:.4f}")
print(f"    Buy & Hold Final:  ${bt['final_bah_value']:.4f}")
print(f"    Win Rate:          {bt['win_rate']*100:.1f}%")
print(f"    Sharpe Ratio:      {bt['strategy_sharpe']:.4f}")
print(f"    Total Trades:      {int(bt['total_trades'])}")

step("Generating ML charts...")
plot_ml_results(features, model_results, close_prices)
plot_backtest(bt)

# ══════════════════════════════════════════════════════════════════
# STEP 5 — FINAL REPORT
# ══════════════════════════════════════════════════════════════════
banner("STEP 5: Generating Final Report", "\033[94m")

from sql_analytics import run_sql_analytics
sql_results = run_sql_analytics(close_prices, returns)

# Save comprehensive report
report_lines = [
    "=" * 70,
    "  . — DATA ANALYTICS PROJECT",
    "  Financial Market Intelligence Report",
    "=" * 70,
    "",
    "PROJECT SUMMARY",
    "-" * 40,
    "Author:    [Your Name Here]",
    "Role:      Data Analytics Intern / Analyst Candidate",
    "Date:      " + pd.Timestamp.now().strftime("%Y-%m-%d"),
    "",
    "TECHNOLOGIES USED",
    "-" * 40,
    "  Python, Pandas, NumPy, Scikit-learn",
    "  Matplotlib, Seaborn (visualization)",
    "  SciPy (statistical analysis)",
    "  SQL (via pandasql / in-memory simulation)",
    "  Algorithms: Random Forest, Isolation Forest,",
    "              GBM Simulation, Markowitz Optimization,",
    "              CAPM, Walk-Forward Cross-Validation",
    "",
    "DATASET",
    "-" * 40,
    f"  Stocks:      {', '.join([t for t in tickers if t != 'SPY'])}",
    f"  Benchmark:   S&P 500 (SPY)",
    f"  Period:      {close_prices.index[0].date()} to {close_prices.index[-1].date()}",
    f"  Data Points: {len(close_prices):,} trading days",
    "",
    "KEY FINDINGS",
    "-" * 40,
    "",
]

bench_ret = returns["SPY"].mean() * 252
for t in [t for t in tickers if t != "SPY"]:
    ann_ret  = returns[t].mean() * 252
    ann_vol  = returns[t].std()  * np.sqrt(252)
    sr       = sharpe_ratio(returns[t])
    var95    = compute_var(returns[t], 0.95)
    max_dd   = compute_max_drawdown(close_prices[t])["max_drawdown"]
    ba       = beta_alpha(returns[t], returns["SPY"])
    alpha_str = f"{ba['alpha']*100:+.2f}%"
    report_lines.append(
        f"  {t:5s}: Ret={ann_ret*100:+6.2f}%  Vol={ann_vol*100:5.2f}%  "
        f"SR={sr:6.3f}  VaR95={var95*100:6.3f}%  MaxDD={max_dd*100:6.2f}%  "
        f"Beta={ba['beta']:.3f}  Alpha={alpha_str}"
    )

report_lines += [
    "",
    "ML MODEL PERFORMANCE",
    "-" * 40,
    f"  Model:        Random Forest (200 trees, Walk-Forward CV)",
    f"  Task:         5-day return direction classification",
    f"  Final AUC:    {model_results['final_auc']:.4f}",
    f"  Mean CV AUC:  {folds['auc'].mean():.4f}",
    f"  Win Rate:     {bt['win_rate']*100:.1f}%",
    f"  Strategy SR:  {bt['strategy_sharpe']:.4f}",
    "",
    "PORTFOLIO OPTIMIZATION",
    "-" * 40,
    f"  Max Sharpe:   Ret={ms_r*100:.2f}%  Vol={ms_v*100:.2f}%  SR={ms_s:.3f}",
    f"  Min Variance: Ret={mv_r*100:.2f}%  Vol={mv_v*100:.2f}%  SR={mv_s:.3f}",
    "",
    "OUTPUT FILES",
    "-" * 40,
    "  charts/01_price_dashboard.png      ← Price, returns, correlations",
    "  charts/02_risk_dashboard.png       ← VaR, CVaR, Monte Carlo",
    "  charts/03_portfolio_optimization.png ← Efficient Frontier",
    "  charts/03_ml_signals.png           ← ML signals & anomaly detection",
    "  charts/04_backtest.png             ← Strategy vs Buy-and-Hold",
    "  data/risk_report.csv              ← Full risk metrics table",
    "  data/close_prices.csv             ← Price data",
    "",
    "=" * 70,
]

report_text = "\n".join(report_lines)
with open("reports/final_report.txt","w", encoding="utf-8") as f:
    f.write(report_text)

print(report_text)

banner("PIPELINE COMPLETE! All outputs saved.", "\033[92m")
print("""
     Open these files to see your analysis:
     → charts/01_price_dashboard.png
     → charts/02_risk_dashboard.png
     → charts/03_portfolio_optimization.png
     → charts/03_ml_signals.png
     → charts/04_backtest.png
     → data/risk_report.csv
     → reports/final_report.txt

""")
