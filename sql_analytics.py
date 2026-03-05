

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

TRADING_DAYS = 252


def run_sql_analytics(prices: pd.DataFrame,
                      returns: pd.DataFrame) -> dict:
    """
    Simulates advanced SQL queries on financial data.
    Each function maps directly to SQL logic that GS analysts use.
    """
    print("\n  Running SQL-style Analytics...")

    results = {}

    # ── Query 1: Monthly Returns Summary (GROUP BY month) ────────
    """
    SQL equivalent:
    SELECT DATE_TRUNC('month', date)   AS month,
           AVG(daily_return)           AS avg_return,
           SUM(daily_return)           AS total_return,
           STDDEV(daily_return)        AS volatility,
           COUNT(*)                    AS trading_days,
           MIN(close)                  AS period_low,
           MAX(close)                  AS period_high
    FROM   market_data
    WHERE  ticker = 'GS'
    GROUP  BY 1
    ORDER  BY 1;
    """
    gs_ret   = returns["GS"].copy()
    gs_price = prices["GS"].copy()

    monthly = gs_ret.groupby(pd.Grouper(freq="ME")).agg(
        avg_return   = ("mean"),
        total_return = ("sum"),
        volatility   = ("std"),
        trading_days = ("count")
    )
    monthly["period_low"]  = gs_price.resample("ME").min()
    monthly["period_high"] = gs_price.resample("ME").max()
    monthly.to_csv("data/sql_monthly_summary.csv")
    results["monthly_summary"] = monthly
    print("    ✓ Query 1: Monthly returns summary")

    # ── Query 2: Rolling 30-Day Sharpe Ranking ────────────────────
    """
    SQL equivalent (Window Function):
    SELECT date, ticker, close,
           AVG(daily_return)  OVER w * 252 - 0.053
             / (STDDEV(daily_return) OVER w * SQRT(252)) AS rolling_sharpe,
           RANK() OVER (PARTITION BY date ORDER BY rolling_sharpe DESC) AS sharpe_rank
    FROM   market_data
    WINDOW w AS (PARTITION BY ticker ORDER BY date ROWS 29 PRECEDING);
    """
    risk_free_daily = 0.053 / TRADING_DAYS
    sharpe_matrix   = pd.DataFrame(index=returns.index)

    for ticker in [t for t in returns.columns if t != "SPY"]:
        roll_mean    = returns[ticker].rolling(30).mean()
        roll_std     = returns[ticker].rolling(30).std()
        sharpe_matrix[ticker] = (
            (roll_mean - risk_free_daily) * TRADING_DAYS
            / (roll_std * np.sqrt(TRADING_DAYS))
        )

    sharpe_matrix.to_csv("data/sql_rolling_sharpe.csv")
    results["rolling_sharpe"] = sharpe_matrix
    print("    ✓ Query 2: Rolling 30-day Sharpe rankings")

    # ── Query 3: Regime Detection (CASE WHEN) ─────────────────────
    """
    SQL equivalent:
    SELECT date, ticker, daily_return,
           rolling_vol_21d,
           CASE WHEN rolling_vol_21d > vol_75pct THEN 'HIGH_VOL'
                WHEN rolling_vol_21d < vol_25pct THEN 'LOW_VOL'
                ELSE 'NORMAL'   END AS volatility_regime,
           CASE WHEN rolling_ret_5d > 0.02 THEN 'TRENDING_UP'
                WHEN rolling_ret_5d < -0.02 THEN 'TRENDING_DOWN'
                ELSE 'SIDEWAYS' END AS trend_regime
    FROM   (SELECT *, ... FROM market_data) sub;
    """
    regime_df = pd.DataFrame(index=returns.index)
    ret        = returns["GS"]

    roll_vol_21 = ret.rolling(21).std() * np.sqrt(TRADING_DAYS)
    roll_ret_5  = ret.rolling(5).sum()
    vol_75 = roll_vol_21.quantile(0.75)
    vol_25 = roll_vol_21.quantile(0.25)

    regime_df["volatility_regime"] = np.where(
        roll_vol_21 > vol_75, "HIGH_VOL",
        np.where(roll_vol_21 < vol_25, "LOW_VOL", "NORMAL")
    )
    regime_df["trend_regime"] = np.where(
        roll_ret_5 > 0.02, "TRENDING_UP",
        np.where(roll_ret_5 < -0.02, "TRENDING_DOWN", "SIDEWAYS")
    )
    regime_df["daily_return"] = ret
    regime_df["rolling_vol"]  = roll_vol_21
    regime_df.to_csv("data/sql_regime_analysis.csv")
    results["regime"] = regime_df

    # Regime stats
    regime_stats = regime_df.groupby("volatility_regime")["daily_return"].agg(
        ["mean", "std", "count"]
    )
    print("    Query 3: Volatility regime classification")
    print(f"\n      Regime Statistics:")
    print(regime_stats.to_string(float_format="%.5f"))

    # ── Query 4: Correlation CTE ──────────────────────────────────
    """
    SQL equivalent:
    WITH rolling_corr AS (
        SELECT a.date, a.ticker AS ticker_1, b.ticker AS ticker_2,
               CORR(a.daily_return, b.daily_return)
                 OVER (ORDER BY a.date ROWS 59 PRECEDING) AS correlation
        FROM market_data a JOIN market_data b USING (date)
        WHERE a.ticker < b.ticker
    )
    SELECT * FROM rolling_corr
    WHERE  ABS(correlation) > 0.8   -- highly correlated pairs
    ORDER  BY date DESC;
    """
    tickers_no_spy = [t for t in returns.columns if t != "SPY"]
    corr_pairs     = {}

    for i, t1 in enumerate(tickers_no_spy):
        for t2 in tickers_no_spy[i+1:]:
            roll_corr = returns[t1].rolling(60).corr(returns[t2])
            key = f"{t1}_{t2}"
            corr_pairs[key] = roll_corr

    corr_df = pd.DataFrame(corr_pairs, index=returns.index).dropna()
    corr_df.to_csv("data/sql_rolling_correlations.csv")
    results["rolling_correlations"] = corr_df
    print("    ✓ Query 4: Rolling 60-day pair correlations")

    # ── Query 5: Performance Attribution ─────────────────────────
    """
    SQL equivalent:
    WITH daily_pnl AS (
        SELECT date, ticker,
               close * 1000 AS portfolio_value,
               close * 1000 - LAG(close * 1000) OVER (
                 PARTITION BY ticker ORDER BY date
               ) AS daily_pnl
        FROM market_data
    ),
    cumulative AS (
        SELECT *, SUM(daily_pnl) OVER (
          PARTITION BY ticker ORDER BY date
        ) AS cumulative_pnl
        FROM daily_pnl
    )
    SELECT ticker,
           SUM(daily_pnl) AS total_pnl,
           MAX(cumulative_pnl) AS peak_pnl,
           MIN(cumulative_pnl) AS worst_pnl
    FROM cumulative
    GROUP BY ticker;
    """
    notional = 100_000   # $100k per position

    pnl_summary = []
    for t in tickers_no_spy:
        daily_pnl = prices[t].diff() * (notional / prices[t].iloc[0])
        cum_pnl   = daily_pnl.cumsum()
        pnl_summary.append({
            "Ticker":       t,
            "Total P&L":    f"${cum_pnl.iloc[-1]:+,.0f}",
            "Peak P&L":     f"${cum_pnl.max():+,.0f}",
            "Worst P&L":    f"${cum_pnl.min():+,.0f}",
            "Win Days":     f"{(daily_pnl > 0).sum()}",
            "Loss Days":    f"{(daily_pnl < 0).sum()}",
            "Best Day":     f"${daily_pnl.max():+,.0f}",
            "Worst Day":    f"${daily_pnl.min():+,.0f}",
        })

    pnl_df = pd.DataFrame(pnl_summary).set_index("Ticker")
    pnl_df.to_csv("data/sql_pnl_attribution.csv")
    results["pnl"] = pnl_df

    print("   Query 5: P&L attribution analysis")
    print(f"\n      P&L Summary ($100K per position):")
    print(pnl_df.to_string())

    # ── Query 6: Z-Score Anomaly (Statistical Outlier) ───────────
    """
    SQL equivalent:
    SELECT date, ticker, daily_return,
           (daily_return - AVG(daily_return) OVER (
              PARTITION BY ticker ORDER BY date ROWS 19 PRECEDING
           )) / NULLIF(STDDEV(daily_return) OVER (
              PARTITION BY ticker ORDER BY date ROWS 19 PRECEDING
           ), 0) AS z_score
    FROM market_data
    HAVING ABS(z_score) > 3   -- 3-sigma events
    ORDER BY ABS(z_score) DESC;
    """
    z_scores = {}
    for t in tickers_no_spy:
        roll_mean = returns[t].rolling(20).mean()
        roll_std  = returns[t].rolling(20).std()
        z         = (returns[t] - roll_mean) / roll_std
        z_scores[t] = z

    z_df      = pd.DataFrame(z_scores, index=returns.index).dropna()
    outliers  = (z_df.abs() > 3).any(axis=1)
    outlier_df = z_df[outliers]
    outlier_df.to_csv("data/sql_zscore_outliers.csv")
    results["z_scores"] = z_df
    results["outliers"] = outlier_df

    print(f"    Query 6: Z-score outliers — {len(outlier_df)} 3-sigma events detected")
    print()

    return results
