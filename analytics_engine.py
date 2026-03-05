

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from scipy.stats import norm, kurtosis, skew
from scipy.optimize import minimize
import warnings, os
warnings.filterwarnings("ignore")

# ── . inspired color palette ─────────────────────────
GS_BLUE   = "#00B5CC"
GS_GOLD   = "#B5A642"
GS_RED    = "#CC0000"
GS_DARK   = "#0A1628"
GS_GREY   = "#2E4057"
GS_GREEN  = "#00CC66"
GS_WHITE  = "#E8F0F5"

RISK_FREE_RATE = 0.053      # Fed Funds Rate ~5.3% (2024)
TRADING_DAYS   = 252

plt.rcParams.update({
    "figure.facecolor":  GS_DARK,
    "axes.facecolor":    GS_DARK,
    "axes.edgecolor":    "#1E3A5F",
    "axes.labelcolor":   GS_WHITE,
    "axes.titlecolor":   GS_WHITE,
    "text.color":        GS_WHITE,
    "xtick.color":       GS_WHITE,
    "ytick.color":       GS_WHITE,
    "grid.color":        "#1E3A5F",
    "grid.alpha":        0.5,
    "font.family":       "monospace",
    "font.size":         9,
})

os.makedirs("charts", exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# 1. RETURN CALCULATIONS
# ══════════════════════════════════════════════════════════════════

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns — preferred in quant finance (time-additive)."""
    return np.log(prices / prices.shift(1)).dropna()


def compute_cumulative_returns(returns: pd.DataFrame) -> pd.DataFrame:
    return (1 + returns).cumprod() - 1


# ══════════════════════════════════════════════════════════════════
# 2. RISK METRICS
# ══════════════════════════════════════════════════════════════════

def compute_var(returns: pd.Series, confidence: float = 0.95,
                method: str = "historical") -> float:
    """
    Value at Risk:  maximum expected loss at given confidence level.
    GS uses both Historical Simulation and Parametric (Normal).
    """
    if method == "historical":
        return float(np.percentile(returns, (1 - confidence) * 100))
    else:  # parametric
        mu, sigma = returns.mean(), returns.std()
        return float(norm.ppf(1 - confidence, mu, sigma))


def compute_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Conditional VaR / Expected Shortfall:
    Average loss BEYOND the VaR threshold. More conservative than VaR.
    Preferred by Basel III and used extensively at GS risk management.
    """
    var = compute_var(returns, confidence)
    return float(returns[returns <= var].mean())


def compute_volatility(returns: pd.Series, annualize: bool = True) -> float:
    vol = returns.std()
    return float(vol * np.sqrt(TRADING_DAYS)) if annualize else float(vol)


def compute_max_drawdown(prices: pd.Series) -> dict:
    """Peak-to-trough maximum loss percentage."""
    roll_max   = prices.cummax()
    drawdown   = (prices - roll_max) / roll_max
    max_dd     = drawdown.min()
    end_idx    = drawdown.idxmin()
    start_idx  = prices[:end_idx].idxmax()
    return {
        "max_drawdown":   float(max_dd),
        "peak_date":      start_idx,
        "trough_date":    end_idx,
        "drawdown_series": drawdown
    }


# ══════════════════════════════════════════════════════════════════
# 3. PERFORMANCE RATIOS
# ══════════════════════════════════════════════════════════════════

def sharpe_ratio(returns: pd.Series) -> float:
    """Risk-adjusted return per unit of total risk."""
    excess     = returns.mean() * TRADING_DAYS - RISK_FREE_RATE
    annual_vol = returns.std() * np.sqrt(TRADING_DAYS)
    return round(float(excess / annual_vol), 4)


def sortino_ratio(returns: pd.Series) -> float:
    """Like Sharpe but penalizes only downside volatility."""
    excess          = returns.mean() * TRADING_DAYS - RISK_FREE_RATE
    downside_vol    = returns[returns < 0].std() * np.sqrt(TRADING_DAYS)
    return round(float(excess / downside_vol), 4) if downside_vol > 0 else np.nan


def calmar_ratio(returns: pd.Series, prices: pd.Series) -> float:
    """Annual return / Max Drawdown — used for hedge fund evaluation."""
    annual_ret = returns.mean() * TRADING_DAYS
    max_dd     = abs(compute_max_drawdown(prices)["max_drawdown"])
    return round(float(annual_ret / max_dd), 4) if max_dd > 0 else np.nan


def beta_alpha(asset_returns: pd.Series,
               benchmark_returns: pd.Series) -> dict:
    """
    CAPM Beta & Jensen's Alpha:
    Beta  = systematic risk relative to market
    Alpha = excess return beyond what CAPM predicts
    """
    cov  = np.cov(asset_returns, benchmark_returns)
    beta = cov[0, 1] / cov[1, 1]
    alpha = (asset_returns.mean() - RISK_FREE_RATE / TRADING_DAYS
             - beta * (benchmark_returns.mean()
                       - RISK_FREE_RATE / TRADING_DAYS)) * TRADING_DAYS
    return {"beta": round(float(beta), 4), "alpha": round(float(alpha), 4)}


# ══════════════════════════════════════════════════════════════════
# 4. PORTFOLIO OPTIMIZATION (Markowitz Mean-Variance)
# ══════════════════════════════════════════════════════════════════

def portfolio_performance(weights, mean_returns, cov_matrix):
    ret = np.sum(mean_returns * weights) * TRADING_DAYS
    vol = np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(TRADING_DAYS)
    sr  = (ret - RISK_FREE_RATE) / vol
    return ret, vol, sr


def optimize_portfolio(returns: pd.DataFrame) -> dict:
    """
    Efficient Frontier Optimization.
    Finds: Max Sharpe Ratio Portfolio & Min Variance Portfolio.
    """
    tickers      = [c for c in returns.columns if c != "SPY"]
    ret_no_bench = returns[tickers]
    mean_ret     = ret_no_bench.mean()
    cov_mat      = ret_no_bench.cov()
    n            = len(tickers)
    bounds       = tuple((0.02, 0.40) for _ in range(n))   # max 40% any stock
    constraints  = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    w0           = np.array([1 / n] * n)

    # ── Max Sharpe ────────────────────────────────────────────────
    def neg_sharpe(w):
        r, v, s = portfolio_performance(w, mean_ret, cov_mat)
        return -s

    res_sharpe = minimize(neg_sharpe, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)

    # ── Min Variance ──────────────────────────────────────────────
    def port_vol(w):
        return portfolio_performance(w, mean_ret, cov_mat)[1]

    res_minvar = minimize(port_vol, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)

    # ── Efficient Frontier Points ─────────────────────────────────
    target_rets = np.linspace(mean_ret.min() * 252,
                              mean_ret.max() * 252, 60)
    ef_vols     = []
    for tr in target_rets:
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w, r=tr:
                 np.sum(w * mean_ret) * TRADING_DAYS - r}]
        r    = minimize(port_vol, w0, method="SLSQP",
                        bounds=bounds, constraints=cons)
        ef_vols.append(r.fun if r.success else np.nan)

    return {
        "tickers":           tickers,
        "mean_returns":      mean_ret,
        "cov_matrix":        cov_mat,
        "max_sharpe_weights": dict(zip(tickers, res_sharpe.x)),
        "max_sharpe_perf":    portfolio_performance(res_sharpe.x, mean_ret, cov_mat),
        "min_var_weights":    dict(zip(tickers, res_minvar.x)),
        "min_var_perf":       portfolio_performance(res_minvar.x, mean_ret, cov_mat),
        "ef_returns":         target_rets,
        "ef_vols":            ef_vols,
    }


# ══════════════════════════════════════════════════════════════════
# 5. VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════

def pct_fmt(x, _): return f"{x*100:.0f}%"
def dollar_fmt(x, _): return f"${x:,.0f}"


def plot_price_dashboard(prices, returns):
    """Chart 1: Price performance + volume + returns distribution."""
    tickers    = [c for c in prices.columns if c != "SPY"]
    base_prices = prices[tickers].div(prices[tickers].iloc[0])   # normalized

    fig = plt.figure(figsize=(18, 10), facecolor=GS_DARK)
    fig.suptitle("  . — Financial Market Analytics Dashboard",
                 fontsize=14, fontweight="bold", color=GS_GOLD,
                 x=0.02, ha="left")

    gs_layout = gridspec.GridSpec(2, 3, figure=fig,
                                  hspace=0.45, wspace=0.35,
                                  left=0.07, right=0.97,
                                  top=0.90, bottom=0.08)

    colors = [GS_BLUE, GS_GOLD, GS_GREEN, "#FF6B6B",
              "#A78BFA", "#34D399", "#FBBF24"]

    # ── Panel 1: Normalized prices ────────────────────────────────
    ax1 = fig.add_subplot(gs_layout[0, :2])
    for i, t in enumerate(tickers):
        ax1.plot(base_prices.index, base_prices[t],
                 color=colors[i % len(colors)], linewidth=1.2,
                 label=t, alpha=0.9)
    ax1.axhline(1, color="#555", linestyle="--", linewidth=0.8)
    ax1.set_title("Normalized Price Performance (Base = $1.00)",
                  fontsize=10, pad=8)
    ax1.yaxis.set_major_formatter(FuncFormatter(dollar_fmt))
    ax1.legend(loc="upper left", fontsize=7,
               ncol=4, framealpha=0.2)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Returns distribution (GS) ───────────────────────
    ax2 = fig.add_subplot(gs_layout[0, 2])
    gs_ret = returns["GS"].dropna()
    ax2.hist(gs_ret, bins=60, color=GS_BLUE,
             alpha=0.7, density=True, label="Empirical")
    xmin, xmax = ax2.get_xlim()
    x_norm = np.linspace(xmin, xmax, 200)
    ax2.plot(x_norm,
             norm.pdf(x_norm, gs_ret.mean(), gs_ret.std()),
             color=GS_GOLD, linewidth=1.8, label="Normal Fit")
    var_95 = compute_var(gs_ret, 0.95)
    ax2.axvline(var_95, color=GS_RED, linestyle="--",
                linewidth=1.5, label=f"VaR 95%: {var_95*100:.2f}%")
    ax2.set_title("GS — Daily Returns Distribution", fontsize=10, pad=8)
    ax2.legend(fontsize=7, framealpha=0.2)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Correlation heatmap ──────────────────────────────
    ax3 = fig.add_subplot(gs_layout[1, 0])
    corr = returns[tickers].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, ax=ax3, cmap="RdYlGn", center=0,
                vmin=-1, vmax=1, annot=True, fmt=".2f",
                annot_kws={"size": 7}, linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                xticklabels=tickers, yticklabels=tickers)
    ax3.set_title("Correlation Matrix", fontsize=10, pad=8)
    ax3.tick_params(labelsize=7)

    # ── Panel 4: Rolling 30-day Sharpe ───────────────────────────
    ax4 = fig.add_subplot(gs_layout[1, 1])
    for i, t in enumerate(tickers[:4]):
        roll_sr = (returns[t].rolling(30).mean() * TRADING_DAYS
                   - RISK_FREE_RATE) / (returns[t].rolling(30).std()
                                        * np.sqrt(TRADING_DAYS))
        ax4.plot(roll_sr.index, roll_sr,
                 color=colors[i], linewidth=1, label=t, alpha=0.85)
    ax4.axhline(0, color="#555", linestyle="--", linewidth=0.8)
    ax4.axhline(1, color=GS_GOLD, linestyle=":", linewidth=0.8,
                label="SR=1 (good)")
    ax4.set_title("Rolling 30-Day Sharpe Ratio", fontsize=10, pad=8)
    ax4.legend(fontsize=7, framealpha=0.2)
    ax4.grid(True, alpha=0.3)

    # ── Panel 5: Drawdown ─────────────────────────────────────────
    ax5 = fig.add_subplot(gs_layout[1, 2])
    for i, t in enumerate(tickers[:4]):
        dd = compute_max_drawdown(prices[t])["drawdown_series"]
        ax5.fill_between(dd.index, dd, 0,
                         color=colors[i], alpha=0.25, label=t)
        ax5.plot(dd.index, dd, color=colors[i],
                 linewidth=0.8, alpha=0.8)
    ax5.set_title("Portfolio Drawdown Analysis", fontsize=10, pad=8)
    ax5.yaxis.set_major_formatter(FuncFormatter(pct_fmt))
    ax5.legend(fontsize=7, framealpha=0.2)
    ax5.grid(True, alpha=0.3)

    plt.savefig("charts/01_price_dashboard.png", dpi=150,
                bbox_inches="tight", facecolor=GS_DARK)
    plt.close()
    print("  ✅ Saved: charts/01_price_dashboard.png")


def plot_risk_dashboard(prices, returns):
    """Chart 2: VaR, CVaR, Monte Carlo, Stress Testing."""
    tickers = [c for c in prices.columns if c != "SPY"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor=GS_DARK)
    fig.suptitle("  . — Risk Analytics & Stress Testing",
                 fontsize=14, fontweight="bold", color=GS_RED,
                 x=0.02, ha="left")
    axes = axes.flatten()
    colors = [GS_BLUE, GS_GOLD, GS_GREEN, "#FF6B6B", "#A78BFA", "#34D399", "#FBBF24"]

    # ── Panel 1: VaR Comparison (all stocks) ─────────────────────
    ax = axes[0]
    var95  = [abs(compute_var(returns[t], 0.95)) for t in tickers]
    var99  = [abs(compute_var(returns[t], 0.99)) for t in tickers]
    cvar95 = [abs(compute_cvar(returns[t], 0.95)) for t in tickers]
    x      = np.arange(len(tickers))
    ax.bar(x - 0.25, var95,  0.25, label="VaR 95%",  color=GS_BLUE,  alpha=0.85)
    ax.bar(x,        var99,  0.25, label="VaR 99%",  color=GS_RED,   alpha=0.85)
    ax.bar(x + 0.25, cvar95, 0.25, label="CVaR 95%", color=GS_GOLD,  alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(tickers, fontsize=8)
    ax.yaxis.set_major_formatter(FuncFormatter(pct_fmt))
    ax.set_title("Value at Risk (VaR) & Expected Shortfall", fontsize=10, pad=8)
    ax.legend(fontsize=7, framealpha=0.2); ax.grid(True, alpha=0.3)

    # ── Panel 2: Monte Carlo Simulation (GS stock) ───────────────
    ax = axes[1]
    gs_prices = prices["GS"].values
    S0        = gs_prices[-1]
    mu        = returns["GS"].mean()
    sigma     = returns["GS"].std()
    n_sim, horizon = 500, 60

    np.random.seed(0)
    simulations = np.zeros((horizon, n_sim))
    for i in range(n_sim):
        path = [S0]
        for _ in range(horizon - 1):
            r = (mu - 0.5 * sigma**2) + sigma * np.random.normal()
            path.append(path[-1] * np.exp(r))
        simulations[:, i] = path

    for i in range(0, n_sim, 5):
        ax.plot(simulations[:, i], color=GS_BLUE, alpha=0.06, linewidth=0.5)

    p5   = np.percentile(simulations, 5,  axis=1)
    p50  = np.percentile(simulations, 50, axis=1)
    p95  = np.percentile(simulations, 95, axis=1)
    ax.plot(p5,  color=GS_RED,   linewidth=2, label="5th Percentile", linestyle="--")
    ax.plot(p50, color=GS_GOLD,  linewidth=2, label="Median")
    ax.plot(p95, color=GS_GREEN, linewidth=2, label="95th Percentile", linestyle="--")
    ax.fill_between(range(horizon), p5, p95, color=GS_BLUE, alpha=0.12)
    ax.set_title(f"Monte Carlo: GS Stock (500 paths, 60-day)", fontsize=10, pad=8)
    ax.yaxis.set_major_formatter(FuncFormatter(dollar_fmt))
    ax.legend(fontsize=7, framealpha=0.2); ax.grid(True, alpha=0.3)

    # ── Panel 3: Volatility Regime Detection ─────────────────────
    ax = axes[2]
    gs_ret    = returns["GS"]
    roll_vol  = gs_ret.rolling(21).std() * np.sqrt(TRADING_DAYS)
    high_vol  = roll_vol > roll_vol.quantile(0.75)
    ax.plot(roll_vol.index, roll_vol, color=GS_BLUE, linewidth=1)
    ax.fill_between(roll_vol.index, 0, roll_vol,
                    where=high_vol, color=GS_RED, alpha=0.35, label="High Vol Regime")
    ax.fill_between(roll_vol.index, 0, roll_vol,
                    where=~high_vol, color=GS_GREEN, alpha=0.2, label="Low Vol Regime")
    ax.axhline(roll_vol.mean(), color=GS_GOLD, linestyle=":",
               linewidth=1.5, label=f"Mean Vol: {roll_vol.mean()*100:.1f}%")
    ax.set_title("Volatility Regime Detection (21-day Rolling)", fontsize=10, pad=8)
    ax.yaxis.set_major_formatter(FuncFormatter(pct_fmt))
    ax.legend(fontsize=7, framealpha=0.2); ax.grid(True, alpha=0.3)

    # ── Panel 4: Beta & Alpha Bar Chart ──────────────────────────
    ax = axes[3]
    betas  = []
    alphas = []
    bench  = returns["SPY"]
    for t in tickers:
        ba = beta_alpha(returns[t], bench)
        betas.append(ba["beta"])
        alphas.append(ba["alpha"])
    x = np.arange(len(tickers))
    ax.bar(x - 0.2, betas,  0.35, label="Beta",  color=GS_BLUE,  alpha=0.85)
    ax.bar(x + 0.2, [a * 10 for a in alphas],
                           0.35, label="Alpha×10", color=GS_GOLD, alpha=0.85)
    ax.axhline(1, color=GS_WHITE, linestyle=":", linewidth=1)
    ax.axhline(0, color="#555",   linestyle="--", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(tickers, fontsize=8)
    ax.set_title("CAPM Beta & Jensen's Alpha", fontsize=10, pad=8)
    ax.legend(fontsize=7, framealpha=0.2); ax.grid(True, alpha=0.3)

    # ── Panel 5: Stress Test Scenarios ───────────────────────────
    ax = axes[4]
    shocks = {
        "2008 GFC": -0.35,
        "COVID Crash": -0.28,
        "Flash Crash": -0.10,
        "Mild Correction": -0.12,
        "Rate Shock": -0.15,
        "Bull Case": +0.20,
    }
    portfolio_val = 1_000_000
    scenario_results = {}
    weights         = np.array([1/len(tickers)] * len(tickers))

    for scenario, shock in shocks.items():
        shock_impact = np.random.normal(shock, 0.05, len(tickers))
        port_return  = weights @ shock_impact
        pnl          = portfolio_val * port_return
        scenario_results[scenario] = pnl

    bar_colors = [GS_RED if v < 0 else GS_GREEN
                  for v in scenario_results.values()]
    bars = ax.barh(list(scenario_results.keys()),
                   list(scenario_results.values()),
                   color=bar_colors, alpha=0.85)
    ax.axvline(0, color=GS_WHITE, linewidth=0.8)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, _: f"${x/1000:+.0f}K"))
    ax.set_title("Stress Test: Portfolio P&L ($1M Notional)", fontsize=10, pad=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 6: Rolling Correlation to SPY ──────────────────────
    ax = axes[5]
    for i, t in enumerate(tickers[:5]):
        roll_corr = returns[t].rolling(60).corr(returns["SPY"])
        ax.plot(roll_corr.index, roll_corr,
                color=colors[i], linewidth=1, label=t, alpha=0.85)
    ax.axhline(0, color="#555", linestyle="--", linewidth=0.8)
    ax.axhline(1, color=GS_WHITE, linestyle=":", linewidth=0.8)
    ax.set_title("Rolling 60-Day Correlation vs S&P 500", fontsize=10, pad=8)
    ax.legend(fontsize=7, framealpha=0.2); ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("charts/02_risk_dashboard.png", dpi=150,
                bbox_inches="tight", facecolor=GS_DARK)
    plt.close()
    print("  ✅ Saved: charts/02_risk_dashboard.png")


def plot_portfolio_optimization(opt_result):
    """Chart 3: Efficient Frontier & Optimal Portfolio Weights."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor=GS_DARK)
    fig.suptitle("  . — Portfolio Optimization (Markowitz Efficient Frontier)",
                 fontsize=14, fontweight="bold", color=GS_GOLD,
                 x=0.02, ha="left")

    # ── Efficient Frontier ────────────────────────────────────────
    ax = axes[0]
    ef_vols    = np.array(opt_result["ef_vols"])
    ef_rets    = opt_result["ef_returns"]
    valid      = ~np.isnan(ef_vols)
    sharpe_rf  = (ef_rets[valid] - RISK_FREE_RATE) / ef_vols[valid]
    scatter    = ax.scatter(ef_vols[valid], ef_rets[valid],
                            c=sharpe_rf, cmap="RdYlGn",
                            s=20, zorder=5)
    plt.colorbar(scatter, ax=ax, label="Sharpe Ratio")

    ms_r, ms_v, ms_s = opt_result["max_sharpe_perf"]
    mv_r, mv_v, mv_s = opt_result["min_var_perf"]

    ax.scatter(ms_v, ms_r, color=GS_GOLD, s=300, zorder=10,
               marker="*", label=f"Max Sharpe * (SR={ms_s:.2f})")
    ax.scatter(mv_v, mv_r, color=GS_BLUE, s=200, zorder=10,
               marker="D", label=f"Min Variance D (SR={mv_s:.2f})")

    ax.xaxis.set_major_formatter(FuncFormatter(pct_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(pct_fmt))
    ax.set_xlabel("Annual Volatility (Risk)")
    ax.set_ylabel("Annual Return")
    ax.set_title("Efficient Frontier", fontsize=11, pad=10)
    ax.legend(fontsize=8, framealpha=0.2); ax.grid(True, alpha=0.3)

    # ── Optimal Weights ───────────────────────────────────────────
    ax = axes[1]
    tickers = opt_result["tickers"]
    ms_w    = [opt_result["max_sharpe_weights"][t] for t in tickers]
    mv_w    = [opt_result["min_var_weights"][t]    for t in tickers]
    x       = np.arange(len(tickers))

    ax.bar(x - 0.2, ms_w, 0.38, label="Max Sharpe Portfolio",
           color=GS_GOLD, alpha=0.85)
    ax.bar(x + 0.2, mv_w, 0.38, label="Min Variance Portfolio",
           color=GS_BLUE, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(tickers, fontsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(pct_fmt))
    ax.set_title("Optimal Portfolio Weights", fontsize=11, pad=10)
    ax.legend(fontsize=8, framealpha=0.2); ax.grid(True, alpha=0.3)

    metrics_text = (
        f"Max Sharpe:  Ret={ms_r*100:.1f}%  Vol={ms_v*100:.1f}%  SR={ms_s:.2f}\n"
        f"Min Variance: Ret={mv_r*100:.1f}%  Vol={mv_v*100:.1f}%  SR={mv_s:.2f}"
    )
    fig.text(0.5, 0.01, metrics_text,
             ha="center", fontsize=9, color=GS_WHITE,
             bbox=dict(boxstyle="round", facecolor="#1E3A5F", alpha=0.7))

    plt.tight_layout(rect=[0, 0.06, 1, 0.93])
    plt.savefig("charts/03_portfolio_optimization.png", dpi=150,
                bbox_inches="tight", facecolor=GS_DARK)
    plt.close()
    print(" Saved: charts/03_portfolio_optimization.png")


def generate_risk_report(prices, returns) -> pd.DataFrame:
    """Generate comprehensive risk metrics table for all tickers."""
    bench   = returns["SPY"]
    tickers = [c for c in returns.columns if c != "SPY"]
    rows    = []

    for t in tickers:
        ret = returns[t]
        ba  = beta_alpha(ret, bench)
        dd  = compute_max_drawdown(prices[t])
        rows.append({
            "Ticker":        t,
            "Annual Return": f"{ret.mean()*TRADING_DAYS*100:.2f}%",
            "Annual Vol":    f"{compute_volatility(ret)*100:.2f}%",
            "Sharpe Ratio":  f"{sharpe_ratio(ret):.3f}",
            "Sortino Ratio": f"{sortino_ratio(ret):.3f}",
            "VaR 95% (1D)":  f"{compute_var(ret,0.95)*100:.3f}%",
            "VaR 99% (1D)":  f"{compute_var(ret,0.99)*100:.3f}%",
            "CVaR 95%":      f"{compute_cvar(ret,0.95)*100:.3f}%",
            "Max Drawdown":  f"{dd['max_drawdown']*100:.2f}%",
            "Beta (vs SPY)": f"{ba['beta']:.3f}",
            "Alpha (ann.)":  f"{ba['alpha']*100:.2f}%",
            "Skewness":      f"{skew(ret):.3f}",
            "Kurtosis":      f"{kurtosis(ret):.3f}",
        })

    df = pd.DataFrame(rows).set_index("Ticker")
    df.to_csv("data/risk_report.csv")
    return df
