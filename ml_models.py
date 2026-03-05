

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
import warnings, os
warnings.filterwarnings("ignore")

GS_DARK  = "#0A1628"
GS_BLUE  = "#00B5CC"
GS_GOLD  = "#B5A642"
GS_RED   = "#CC0000"
GS_GREEN = "#00CC66"
GS_WHITE = "#E8F0F5"

plt.rcParams.update({
    "figure.facecolor": GS_DARK, "axes.facecolor": GS_DARK,
    "axes.edgecolor":   "#1E3A5F", "axes.labelcolor": GS_WHITE,
    "axes.titlecolor":  GS_WHITE, "text.color":      GS_WHITE,
    "xtick.color":      GS_WHITE, "ytick.color":     GS_WHITE,
    "grid.color":       "#1E3A5F", "grid.alpha":     0.5,
    "font.family":      "monospace", "font.size":     9,
})

os.makedirs("charts", exist_ok=True)



# 1. TECHNICAL INDICATOR FEATURE ENGINEERING


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(prices: pd.Series,
                 fast=12, slow=26, signal=9) -> pd.DataFrame:
    ema_f  = prices.ewm(span=fast).mean()
    ema_s  = prices.ewm(span=slow).mean()
    macd   = ema_f - ema_s
    sig    = macd.ewm(span=signal).mean()
    return pd.DataFrame({"macd": macd, "signal": sig,
                          "histogram": macd - sig})


def compute_bollinger(prices: pd.Series,
                      period=20, n_std=2.0) -> pd.DataFrame:
    sma  = prices.rolling(period).mean()
    std  = prices.rolling(period).std()
    return pd.DataFrame({
        "upper": sma + n_std * std,
        "middle": sma,
        "lower":  sma - n_std * std,
        "bandwidth": (2 * n_std * std) / sma
    })


def compute_atr(high, low, close, period=14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature set:
    - Price-based: returns at multiple lags
    - Momentum: RSI, MACD, Rate of Change
    - Volatility: ATR, Bollinger Bandwidth, Rolling Std
    - Volume: relative volume, volume rate of change
    - Calendar: day of week, month (seasonality)
    """
    f = pd.DataFrame(index=df.index)

    # Log returns (multiple lags)
    f["ret_1d"]  = np.log(df["close"] / df["close"].shift(1))
    f["ret_3d"]  = np.log(df["close"] / df["close"].shift(3))
    f["ret_5d"]  = np.log(df["close"] / df["close"].shift(5))
    f["ret_10d"] = np.log(df["close"] / df["close"].shift(10))
    f["ret_20d"] = np.log(df["close"] / df["close"].shift(20))

    # Momentum
    f["rsi_14"]  = compute_rsi(df["close"], 14)
    f["rsi_7"]   = compute_rsi(df["close"], 7)
    macd_df      = compute_macd(df["close"])
    f["macd"]    = macd_df["macd"]
    f["macd_sig"]= macd_df["signal"]
    f["macd_hist"]= macd_df["histogram"]
    f["roc_5"]   = df["close"].pct_change(5)
    f["roc_10"]  = df["close"].pct_change(10)

    # Moving averages & crossovers
    f["ma_5"]    = df["close"].rolling(5).mean()
    f["ma_20"]   = df["close"].rolling(20).mean()
    f["ma_50"]   = df["close"].rolling(50).mean()
    f["ma5_20"]  = f["ma_5"] / f["ma_20"] - 1     # 5-20 crossover signal
    f["ma20_50"] = f["ma_20"] / f["ma_50"] - 1    # 20-50 crossover signal
    f["price_ma20"] = df["close"] / f["ma_20"] - 1

    # Volatility
    bb           = compute_bollinger(df["close"])
    f["bb_pct"]  = (df["close"] - bb["lower"]) / (bb["upper"] - bb["lower"])
    f["bb_width"]= bb["bandwidth"]
    f["vol_5d"]  = f["ret_1d"].rolling(5).std()
    f["vol_20d"] = f["ret_1d"].rolling(20).std()
    f["vol_ratio"]= f["vol_5d"] / f["vol_20d"]

    if "high" in df.columns and "low" in df.columns:
        f["atr_14"] = compute_atr(df["high"], df["low"], df["close"])
        f["atr_pct"] = f["atr_14"] / df["close"]

    # Volume features
    if "volume" in df.columns:
        f["vol_rel"]  = df["volume"] / df["volume"].rolling(20).mean()
        f["vol_roc"]  = df["volume"].pct_change(5)

    # Calendar
    f["day_of_week"] = pd.to_datetime(df.index).dayofweek
    f["month"]       = pd.to_datetime(df.index).month

    # Target: 1 if next-5-day return > 0, else 0
    f["target"] = (np.log(df["close"].shift(-5) / df["close"]) > 0).astype(int)

    return f.dropna()


# 2. WALK-FORWARD VALIDATION (critical for time-series ML!)

def walk_forward_train(features: pd.DataFrame, n_splits: int = 5) -> dict:
    """
    Walk-forward cross-validation — CRITICAL in finance.
    Standard k-fold leaks future data; walk-forward doesn't.
    This is what GS quants actually use.
    """
    X = features.drop("target", axis=1).values
    y = features["target"].values

    scaler  = StandardScaler()
    tscv    = TimeSeriesSplit(n_splits=n_splits)
    model   = RandomForestClassifier(
        n_estimators=200, max_depth=8,
        min_samples_leaf=20, random_state=42,
        class_weight="balanced", n_jobs=-1
    )

    fold_results = []
    all_probs    = []
    all_true     = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        y_tr, y_te = y[train_idx], y[test_idx]

        model.fit(X_tr, y_tr)
        prob   = model.predict_proba(X_te)[:, 1]
        pred   = model.predict(X_te)
        auc    = roc_auc_score(y_te, prob)
        acc    = (pred == y_te).mean()

        fold_results.append({"fold": fold+1, "auc": auc, "accuracy": acc,
                              "n_train": len(train_idx), "n_test": len(test_idx)})
        all_probs.extend(prob)
        all_true.extend(y_te)

    # Final model on all data
    X_all = scaler.fit_transform(X)
    model.fit(X_all, y)

    return {
        "model":          model,
        "scaler":         scaler,
        "feature_names":  features.drop("target", axis=1).columns.tolist(),
        "fold_results":   pd.DataFrame(fold_results),
        "all_probs":      np.array(all_probs),
        "all_true":       np.array(all_true),
        "final_auc":      roc_auc_score(all_true, all_probs),
    }


# 3. ANOMALY DETECTION (Unusual trading activity)


def detect_anomalies(features: pd.DataFrame) -> pd.Series:
    """
    Isolation Forest for anomaly detection.
    Identifies unusual price/volume patterns — market manipulation signals.
    """
    X = features[["ret_1d", "vol_5d", "rsi_14", "vol_rel"]].dropna()
    iso = IsolationForest(n_estimators=200, contamination=0.05,
                          random_state=42)
    labels = iso.fit_predict(X)
    scores = iso.score_samples(X)
    return pd.Series(labels, index=X.index), pd.Series(scores, index=X.index)


# 4. VISUALIZATIONS


def plot_ml_results(features, model_results, prices):
    fig = plt.figure(figsize=(18, 12), facecolor=GS_DARK)
    fig.suptitle("  . — Machine Learning: Signal Generation & Anomaly Detection",
                 fontsize=14, fontweight="bold", color=GS_GREEN,
                 x=0.02, ha="left")

    gs_layout = gridspec.GridSpec(3, 3, figure=fig,
                                  hspace=0.55, wspace=0.35,
                                  left=0.07, right=0.97,
                                  top=0.90, bottom=0.06)

    # ── Panel 1: Feature Importance ──────────────────────────────
    ax1 = fig.add_subplot(gs_layout[0, :2])
    feat_imp = pd.Series(
        model_results["model"].feature_importances_,
        index=model_results["feature_names"]
    ).nlargest(15)
    bars = ax1.barh(feat_imp.index[::-1], feat_imp.values[::-1],
                    color=[GS_BLUE if v > feat_imp.median()
                           else "#2E4057" for v in feat_imp.values[::-1]])
    ax1.axvline(feat_imp.median(), color=GS_GOLD, linestyle=":",
                linewidth=1.5, label="Median Importance")
    ax1.set_title("Top 15 Feature Importances (Random Forest)", fontsize=10, pad=8)
    ax1.legend(fontsize=7, framealpha=0.2); ax1.grid(True, alpha=0.3)

    # ── Panel 2: Walk-Forward AUC ─────────────────────────────────
    ax2 = fig.add_subplot(gs_layout[0, 2])
    folds = model_results["fold_results"]
    ax2.bar(folds["fold"], folds["auc"], color=GS_GOLD, alpha=0.85, width=0.5)
    ax2.axhline(0.5, color=GS_RED, linestyle="--",
                linewidth=1.5, label="Random (AUC=0.5)")
    ax2.axhline(folds["auc"].mean(), color=GS_GREEN, linestyle=":",
                linewidth=1.5, label=f"Mean AUC={folds['auc'].mean():.3f}")
    ax2.set_ylim(0.4, 1.0)
    ax2.set_xlabel("Fold"); ax2.set_ylabel("AUC-ROC")
    ax2.set_title("Walk-Forward CV: AUC per Fold\n(No Future Data Leakage!)",
                  fontsize=10, pad=8)
    ax2.legend(fontsize=7, framealpha=0.2); ax2.grid(True, alpha=0.3)

    # ── Panel 3: ROC Curve ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs_layout[1, 0])
    fpr, tpr, _ = roc_curve(model_results["all_true"],
                             model_results["all_probs"])
    auc_val = model_results["final_auc"]
    ax3.plot(fpr, tpr, color=GS_GOLD, linewidth=2,
             label=f"ROC (AUC = {auc_val:.3f})")
    ax3.plot([0,1], [0,1], color="#555", linestyle="--",
             linewidth=1, label="Random Classifier")
    ax3.fill_between(fpr, tpr, alpha=0.1, color=GS_GOLD)
    ax3.set_xlabel("False Positive Rate"); ax3.set_ylabel("True Positive Rate")
    ax3.set_title("ROC Curve: 5-Day Return Direction", fontsize=10, pad=8)
    ax3.legend(fontsize=8, framealpha=0.2); ax3.grid(True, alpha=0.3)

    # ── Panel 4: Prediction Confidence Histogram ──────────────────
    ax4 = fig.add_subplot(gs_layout[1, 1])
    probs   = model_results["all_probs"]
    labels_ = model_results["all_true"]
    ax4.hist(probs[labels_ == 1], bins=30, alpha=0.7,
             color=GS_GREEN, density=True, label="Actual UP")
    ax4.hist(probs[labels_ == 0], bins=30, alpha=0.7,
             color=GS_RED, density=True, label="Actual DOWN")
    ax4.axvline(0.5, color=GS_WHITE, linestyle="--",
                linewidth=1.5, label="Decision Boundary")
    ax4.set_xlabel("Predicted Probability of UP")
    ax4.set_title("Prediction Probability Distribution", fontsize=10, pad=8)
    ax4.legend(fontsize=7, framealpha=0.2); ax4.grid(True, alpha=0.3)

    # ── Panel 5: RSI Signal Visualization ────────────────────────
    ax5 = fig.add_subplot(gs_layout[1, 2])
    rsi  = features["rsi_14"].iloc[-120:]
    ax5.plot(rsi.index, rsi, color=GS_BLUE, linewidth=1.2)
    ax5.axhline(70, color=GS_RED,   linestyle="--", linewidth=1, label="Overbought (70)")
    ax5.axhline(30, color=GS_GREEN, linestyle="--", linewidth=1, label="Oversold (30)")
    ax5.axhline(50, color="#555",   linestyle=":",  linewidth=0.8)
    ax5.fill_between(rsi.index, 70, 100, alpha=0.1, color=GS_RED)
    ax5.fill_between(rsi.index, 0, 30,  alpha=0.1, color=GS_GREEN)
    ax5.set_ylim(0, 100)
    ax5.set_title("RSI(14) — Momentum Oscillator", fontsize=10, pad=8)
    ax5.legend(fontsize=7, framealpha=0.2); ax5.grid(True, alpha=0.3)

    # ── Panel 6: MACD ─────────────────────────────────────────────
    ax6 = fig.add_subplot(gs_layout[2, :2])
    macd_data = features[["macd", "macd_sig", "macd_hist"]].iloc[-150:]
    ax6.plot(macd_data.index, macd_data["macd"],
             color=GS_BLUE,  linewidth=1.2, label="MACD")
    ax6.plot(macd_data.index, macd_data["macd_sig"],
             color=GS_GOLD,  linewidth=1.2, label="Signal")
    ax6.bar(macd_data.index, macd_data["macd_hist"],
            color=[GS_GREEN if v >= 0 else GS_RED
                   for v in macd_data["macd_hist"]],
            alpha=0.6, label="Histogram")
    ax6.axhline(0, color="#555", linewidth=0.8)
    ax6.set_title("MACD — Moving Average Convergence Divergence",
                  fontsize=10, pad=8)
    ax6.legend(fontsize=7, framealpha=0.2); ax6.grid(True, alpha=0.3)

    # ── Panel 7: Anomaly Detection ────────────────────────────────
    ax7 = fig.add_subplot(gs_layout[2, 2])
    labels_anom, scores_anom = detect_anomalies(features)
    ax7.scatter(features.loc[labels_anom.index, "ret_1d"],
                features.loc[labels_anom.index, "vol_5d"],
                c=labels_anom.map({1: GS_BLUE, -1: GS_RED}),
                alpha=0.5, s=15)
    n_anom = (labels_anom == -1).sum()
    ax7.set_xlabel("1-Day Return")
    ax7.set_ylabel("5-Day Rolling Vol")
    ax7.set_title(f"Anomaly Detection (Isolation Forest)\n"
                  f"{n_anom} Anomalies Detected ({n_anom/len(labels_anom)*100:.1f}%)",
                  fontsize=10, pad=8)
    from matplotlib.patches import Patch
    ax7.legend(handles=[Patch(color=GS_BLUE, label="Normal"),
                        Patch(color=GS_RED, label="Anomaly")],
               fontsize=7, framealpha=0.2)
    ax7.grid(True, alpha=0.3)

    plt.savefig("charts/03_ml_signals.png", dpi=150,
                bbox_inches="tight", facecolor=GS_DARK)
    plt.close()
    print("   Saved: charts/03_ml_signals.png")


def backtest_strategy(features: pd.DataFrame,
                      prices: pd.Series,
                      model_results: dict) -> dict:
    """
    Simple backtest: buy when model predicts UP with >60% confidence.
    Compare to buy-and-hold benchmark.
    """
    X       = features.drop("target", axis=1).values
    X_sc    = model_results["scaler"].transform(X)
    probs   = model_results["model"].predict_proba(X_sc)[:, 1]

    prob_series = pd.Series(probs, index=features.index)
    returns_    = np.log(prices / prices.shift(1)).reindex(features.index)

    # Signal: go long when prob > 0.6
    signal   = (prob_series > 0.60).astype(int)
    strat_ret = signal.shift(1) * returns_

    cum_strat  = (1 + strat_ret.fillna(0)).cumprod()
    cum_bah    = (1 + returns_.fillna(0)).cumprod()

    return {
        "strategy_returns":    strat_ret,
        "cumulative_strategy": cum_strat,
        "cumulative_bah":      cum_bah,
        "total_trades":        signal.diff().abs().sum() / 2,
        "win_rate":            (strat_ret > 0).mean(),
        "strategy_sharpe":     strat_ret.mean() / strat_ret.std() * np.sqrt(252),
        "final_strategy_value": cum_strat.iloc[-1],
        "final_bah_value":      cum_bah.iloc[-1],
    }


def plot_backtest(bt_result: dict):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=GS_DARK)
    fig.suptitle("  . — Strategy Backtest vs Buy-and-Hold",
                 fontsize=14, fontweight="bold", color=GS_GOLD,
                 x=0.02, ha="left")

    ax = axes[0]
    ax.plot(bt_result["cumulative_strategy"].index,
            bt_result["cumulative_strategy"],
            color=GS_GOLD, linewidth=2, label="ML Strategy")
    ax.plot(bt_result["cumulative_bah"].index,
            bt_result["cumulative_bah"],
            color=GS_BLUE, linewidth=2, label="Buy & Hold")
    ax.axhline(1, color="#555", linestyle="--", linewidth=0.8)
    ax.set_title("Cumulative Returns Comparison", fontsize=11, pad=10)
    ax.legend(fontsize=9, framealpha=0.2); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    metrics = {
        "Strategy Final Value": f"${bt_result['final_strategy_value']:.2f}",
        "B&H Final Value":      f"${bt_result['final_bah_value']:.2f}",
        "Total Trades":         f"{int(bt_result['total_trades'])}",
        "Win Rate":             f"{bt_result['win_rate']*100:.1f}%",
        "Strategy Sharpe":      f"{bt_result['strategy_sharpe']:.3f}",
    }
    for i, (k, v) in enumerate(metrics.items()):
        ax2.text(0.1, 0.85 - i * 0.15, k, transform=ax2.transAxes,
                 fontsize=10, color=GS_WHITE, ha="left")
        ax2.text(0.7, 0.85 - i * 0.15, v, transform=ax2.transAxes,
                 fontsize=10, color=GS_GOLD,  ha="left", fontweight="bold")
    ax2.axis("off")
    ax2.set_title("Backtest Performance Summary", fontsize=11, pad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("charts/04_backtest.png", dpi=150,
                bbox_inches="tight", facecolor=GS_DARK)
    plt.close()
    print(" Saved: charts/04_backtest.png")
