"""

HOW TO RUN:
    pip install streamlit plotly pandas numpy scikit-learn scipy
    streamlit run streamlit_app.py

Opens at: http://localhost:8501
====================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm, skew, kurtosis
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="GS Financial Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── . Dark Theme CSS ─────────────────────────────────
st.markdown("""
<style>
  /* Main background */
  .stApp { background-color: #050f1a; }
  section[data-testid="stSidebar"] { background-color: #0a1628; border-right: 1px solid #1a3a5c; }
  
  /* All text white */
  .stApp, .stApp p, .stApp label, .stApp span,
  .stApp div, .stMarkdown { color: #e0eaf2 !important; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #0c1e30;
    border: 1px solid #1a3a5c;
    border-radius: 4px;
    padding: 1rem;
  }
  [data-testid="metric-container"] label { color: #5a8aaa !important; font-size: 0.7rem !important; letter-spacing: 0.2em !important; }
  [data-testid="stMetricValue"] { color: #00c8e8 !important; font-size: 1.8rem !important; font-weight: 800 !important; }
  [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: #0a1628; border-bottom: 1px solid #1a3a5c; gap: 0; }
  .stTabs [data-baseweb="tab"] { 
    background: transparent; color: #5a8aaa; 
    border: none; padding: 0.6rem 1.5rem;
    font-family: monospace; font-size: 0.75rem; letter-spacing: 0.1em;
  }
  .stTabs [aria-selected="true"] { 
    background: #0c1e30 !important; color: #00c8e8 !important;
    border-bottom: 2px solid #00c8e8 !important;
  }

  /* Header */
  .gs-header {
    background: linear-gradient(135deg, #0a1628 0%, #0c1e30 100%);
    border: 1px solid #1a3a5c;
    border-left: 4px solid #b5a642;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.5rem;
    border-radius: 2px;
  }
  .gs-header h1 { color: #b5a642 !important; font-size: 1.4rem; margin: 0; letter-spacing: 0.05em; }
  .gs-header p  { color: #5a8aaa !important; font-size: 0.7rem; margin: 0.2rem 0 0; letter-spacing: 0.2em; }

  /* Section titles */
  .section-title {
    font-family: monospace; font-size: 0.65rem; letter-spacing: 0.3em;
    color: #5a8aaa; text-transform: uppercase; 
    border-bottom: 1px solid #1a3a5c; padding-bottom: 0.4rem;
    margin-bottom: 1rem;
  }

  /* Insight cards */
  .insight-card {
    background: #0c1e30; border: 1px solid #1a3a5c;
    border-left: 3px solid #00c8e8;
    padding: 0.8rem 1rem; margin-bottom: 0.6rem; border-radius: 2px;
  }
  .insight-card.warn { border-left-color: #ff6b35; }
  .insight-card.good { border-left-color: #00cc66; }
  .insight-title { color: #e0eaf2 !important; font-weight: 700; font-size: 0.8rem; }
  .insight-text  { color: #5a8aaa !important; font-size: 0.72rem; line-height: 1.5; }

  /* Selectbox & slider */
  .stSelectbox > div, .stSlider { background: #0c1e30; }
  div[data-baseweb="select"] { background: #0c1e30 !important; border-color: #1a3a5c !important; }

  /* Dataframe */
  .stDataFrame { border: 1px solid #1a3a5c; }
  
  /* Plotly charts background */
  .js-plotly-plot .plotly { background: transparent !important; }
  
  /* Sidebar widgets */
  .stMultiSelect [data-baseweb="tag"] { background: #1a3a5c !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════
TRADING_DAYS    = 252
RISK_FREE_RATE  = 0.053
# ── Plot theme helpers ───────────────────────────────────────────
# PLOT_THEME only has keys that are NEVER overridden in update_layout calls.
# Everything else is applied via T(fig) to avoid "multiple values" errors.
PLOT_THEME = dict(
    paper_bgcolor="#050f1a",
    plot_bgcolor ="#0a1628",
    font=dict(color="#e0eaf2", family="monospace", size=11),
)

def T(fig, height=None, title=None, **extra):
    """
    Apply GS dark theme to a figure WITHOUT duplicate-key conflicts.
    T(fig, Call instead of fig.update_layout(**PLOT_THEME, ...).)
    All axis/legend/margin styling applied via dedicated update_* methods.
    """
    layout = dict(**PLOT_THEME)
    if height: layout["height"] = height
    if title:  layout["title"]  = title
    # extra keys that DON'T conflict (barmode, showlegend, etc.)
    for k, v in extra.items():
        layout[k] = v
    fig.update_layout(**layout)
    # Always apply default axis + legend + margin
    fig.update_xaxes(gridcolor="#1a3a5c", showgrid=True)
    fig.update_yaxes(gridcolor="#1a3a5c", showgrid=True)
    fig.update_layout(
        legend=dict(bgcolor="rgba(10,22,40,0.8)", bordercolor="#1a3a5c", borderwidth=1),
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig
COLORS = ["#00c8e8", "#b5a642", "#00cc66", "#ff6b35",
          "#a78bfa", "#34d399", "#fbbf24", "#f87171"]

def hex_to_rgba(hex_color, alpha=0.15):
    """Convert #rrggbb hex to rgba() string for Plotly fillcolor."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"


STOCKS = {
    "GS":  {"S0": 420, "mu": 0.15, "sigma": 0.22},
    "JPM": {"S0": 198, "mu": 0.12, "sigma": 0.20},
    "MS":  {"S0": 102, "mu": 0.13, "sigma": 0.24},
    "BAC": {"S0": 38,  "mu": 0.10, "sigma": 0.26},
    "C":   {"S0": 62,  "mu": 0.09, "sigma": 0.28},
    "WFC": {"S0": 55,  "mu": 0.11, "sigma": 0.23},
    "BLK": {"S0": 830, "mu": 0.14, "sigma": 0.19},
    "SPY": {"S0": 510, "mu": 0.10, "sigma": 0.15},
}

# ══════════════════════════════════════════════════════════════════
# DATA GENERATION (cached)
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def generate_data(seed=42):
    np.random.seed(seed)
    from datetime import datetime, timedelta

    def gbm(S0, mu, sigma, n=504):
        dt, p = 1/252, [S0]
        for _ in range(n-1):
            p.append(p[-1] * np.exp((mu-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*np.random.normal()))
        return np.array(p)

    def trading_dates(n=504):
        d, cur = [], datetime(2022, 1, 3)
        while len(d) < n:
            if cur.weekday() < 5: d.append(cur)
            cur += timedelta(days=1)
        return pd.DatetimeIndex(d)

    dates = trading_dates()
    prices, ohlcv_gs = {}, None

    for ticker, p in STOCKS.items():
        pr = gbm(p["S0"], p["mu"], p["sigma"])
        # inject 6 market shocks
        for day in np.random.choice(len(pr)-5, 6, replace=False):
            s = np.random.choice([-1,1]) * np.random.uniform(0.03, 0.09)
            pr[day:day+3] *= (1 + s * np.array([1, -0.4, 0.2]))
        prices[ticker] = pr
        if ticker == "GS":
            vol = (1 + 2*np.abs(np.diff(np.log(pr))) + np.random.exponential(0.5, len(pr)-1))
            vol = np.append(1, vol) * int(1e6/p["S0"]*500)
            noise = np.random.uniform(0.003, 0.015, len(pr))
            ohlcv_gs = pd.DataFrame({
                "open":   np.round(pr*(1-noise*0.4), 2),
                "high":   np.round(pr*(1+noise), 2),
                "low":    np.round(pr*(1-noise), 2),
                "close":  np.round(pr, 2),
                "volume": vol.astype(int)
            }, index=dates)

    close_df = pd.DataFrame(prices, index=dates)
    close_df.index.name = "date"
    return close_df, ohlcv_gs

@st.cache_data
def compute_all_metrics(prices_df):
    returns = np.log(prices_df / prices_df.shift(1)).dropna()
    bench   = returns["SPY"]
    tickers = [c for c in returns.columns if c != "SPY"]
    rows    = []
    for t in tickers:
        r  = returns[t]
        mu = r.mean() * TRADING_DAYS
        sv = r.std()  * np.sqrt(TRADING_DAYS)
        sr = (mu - RISK_FREE_RATE) / sv
        v95 = float(np.percentile(r, 5))
        v99 = float(np.percentile(r, 1))
        cvar= float(r[r <= v95].mean())
        dd_s= (prices_df[t] - prices_df[t].cummax()) / prices_df[t].cummax()
        mdd = float(dd_s.min())
        cov = np.cov(r, bench)
        beta= cov[0,1]/cov[1,1]
        alpha=(r.mean()-RISK_FREE_RATE/TRADING_DAYS-beta*(bench.mean()-RISK_FREE_RATE/TRADING_DAYS))*TRADING_DAYS
        rows.append({"Ticker":t, "Annual Return":mu, "Annual Vol":sv,
                     "Sharpe":sr, "VaR 95%":v95, "VaR 99%":v99,
                     "CVaR 95%":cvar, "Max Drawdown":mdd,
                     "Beta":beta, "Alpha":alpha,
                     "Skewness":float(skew(r)), "Kurtosis":float(kurtosis(r))})
    return returns, pd.DataFrame(rows).set_index("Ticker")

@st.cache_data
def build_features_cached(ohlcv_json):
    df = pd.read_json(ohlcv_json)
    df.index = pd.to_datetime(df.index)
    f = pd.DataFrame(index=df.index)
    for lag in [1,3,5,10,20]:
        f[f"ret_{lag}d"] = np.log(df["close"]/df["close"].shift(lag))
    # RSI
    delta = df["close"].diff()
    g,l = delta.clip(lower=0).rolling(14).mean(), (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = 100 - (100/(1+g/l.replace(0,np.nan)))
    f["rsi_7"]  = 100 - (100/(1+(delta.clip(lower=0).rolling(7).mean())/(-delta.clip(upper=0).rolling(7).mean().replace(0,np.nan))))
    # MACD
    ema12,ema26 = df["close"].ewm(12).mean(), df["close"].ewm(26).mean()
    f["macd"] = ema12-ema26; f["macd_sig"]=(ema12-ema26).ewm(9).mean()
    f["macd_hist"]=f["macd"]-f["macd_sig"]
    # Bollinger
    sma20=df["close"].rolling(20).mean(); std20=df["close"].rolling(20).std()
    f["bb_pct"] = (df["close"]-(sma20-2*std20))/(4*std20)
    f["bb_width"]=4*std20/sma20
    # Vol features
    f["vol_5d"] = f["ret_1d"].rolling(5).std()
    f["vol_20d"]= f["ret_1d"].rolling(20).std()
    f["vol_ratio"]=f["vol_5d"]/f["vol_20d"]
    # MAs
    f["ma5_20"] = df["close"].rolling(5).mean()/df["close"].rolling(20).mean()-1
    f["ma20_50"]= df["close"].rolling(20).mean()/df["close"].rolling(50).mean()-1
    f["roc_5"]  = df["close"].pct_change(5)
    f["roc_10"] = df["close"].pct_change(10)
    # Volume
    f["vol_rel"] = df["volume"]/df["volume"].rolling(20).mean()
    f["day_of_week"] = df.index.dayofweek
    f["month"] = df.index.month
    # ATR
    tr = pd.concat([df["high"]-df["low"],
                    (df["high"]-df["close"].shift()).abs(),
                    (df["low"]-df["close"].shift()).abs()], axis=1).max(axis=1)
    f["atr_pct"] = tr.rolling(14).mean()/df["close"]
    # Target
    f["target"] = (np.log(df["close"].shift(-5)/df["close"]) > 0).astype(int)
    return f.dropna()

@st.cache_data
def train_model_cached(features_json):
    features = pd.read_json(features_json)
    features.index = pd.to_datetime(features.index)
    X = features.drop("target", axis=1).values
    y = features["target"].values
    scaler = StandardScaler()
    tscv   = TimeSeriesSplit(n_splits=5)
    model  = RandomForestClassifier(n_estimators=200, max_depth=8,
                                    min_samples_leaf=20, random_state=42,
                                    class_weight="balanced", n_jobs=-1)
    all_probs, all_true, fold_aucs = [], [], []
    for train_i, test_i in tscv.split(X):
        Xtr = scaler.fit_transform(X[train_i])
        Xte = scaler.transform(X[test_i])
        model.fit(Xtr, y[train_i])
        prob = model.predict_proba(Xte)[:,1]
        all_probs.extend(prob); all_true.extend(y[test_i])
        fold_aucs.append(roc_auc_score(y[test_i], prob))
    model.fit(scaler.fit_transform(X), y)
    return {
        "model": model, "scaler": scaler,
        "feat_names": features.drop("target",axis=1).columns.tolist(),
        "all_probs": np.array(all_probs),
        "all_true":  np.array(all_true),
        "fold_aucs": fold_aucs,
        "final_auc": roc_auc_score(all_true, all_probs)
    }

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0; border-bottom:1px solid #1a3a5c; margin-bottom:1rem;'>
      <div style='color:#b5a642; font-size:1.1rem; font-weight:800; letter-spacing:0.1em;'>GS ANALYTICS</div>
      <div style='color:#5a8aaa; font-size:0.6rem; letter-spacing:0.3em;'>FINANCIAL INTELLIGENCE PLATFORM</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Portfolio Configuration</div>', unsafe_allow_html=True)
    all_tickers = ["GS","JPM","MS","BAC","C","WFC","BLK"]
    selected    = st.multiselect("Select Stocks", all_tickers, default=["GS","JPM","MS","WFC"])
    primary     = st.selectbox("Primary Analysis Stock", selected if selected else ["GS"])
    notional    = st.slider("Portfolio Notional ($)", 100_000, 10_000_000, 1_000_000, 100_000,
                            format="$%d")

    st.markdown('<div class="section-title" style="margin-top:1.2rem;">Risk Parameters</div>', unsafe_allow_html=True)
    var_conf    = st.select_slider("VaR Confidence", [0.90, 0.95, 0.99], value=0.95)
    mc_paths    = st.slider("Monte Carlo Paths", 100, 1000, 500, 100)
    mc_horizon  = st.slider("Forecast Horizon (days)", 20, 120, 60, 10)
    rf_rate     = st.slider("Risk-Free Rate (%)", 0.0, 8.0, 5.3, 0.1) / 100

    st.markdown('<div class="section-title" style="margin-top:1.2rem;">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.65rem; color:#5a8aaa; line-height:1.7;'>
    Built for GS Data Analytics<br>
    Application Portfolio<br><br>
    <b style='color:#00c8e8;'>Technologies:</b><br>
    Python · Pandas · NumPy<br>
    Scikit-learn · SciPy<br>
    Plotly · Streamlit<br>
    SQL Analytics · GBM<br>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════
prices_all, gs_ohlcv = generate_data()
sel_tickers = (selected if selected else ["GS"]) + ["SPY"]
prices      = prices_all[sel_tickers]
returns, risk_df = compute_all_metrics(prices)

features = build_features_cached(gs_ohlcv.to_json())
ml_res   = train_model_cached(features.to_json())

# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="gs-header">
  <h1>📊 . — Financial Market Analytics Dashboard</h1>
  <p>QUANTITATIVE RISK ANALYTICS · PORTFOLIO OPTIMIZATION · ML SIGNAL GENERATION</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# KPI METRICS ROW
# ══════════════════════════════════════════════════════════════════
prim_ret = prices[primary]
prim_r   = returns[primary]
ann_ret  = prim_r.mean() * TRADING_DAYS
ann_vol  = prim_r.std()  * np.sqrt(TRADING_DAYS)
sharpe   = (ann_ret - rf_rate) / ann_vol
var95    = np.percentile(prim_r, 5)
max_dd   = ((prim_ret - prim_ret.cummax()) / prim_ret.cummax()).min()
total_return_pct = (prim_ret.iloc[-1] / prim_ret.iloc[0] - 1) * 100

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Return",      f"{total_return_pct:+.1f}%",     f"vs SPY {((prices['SPY'].iloc[-1]/prices['SPY'].iloc[0]-1)*100):+.1f}%")
c2.metric("Annual Return",     f"{ann_ret*100:.1f}%",           f"σ={ann_vol*100:.1f}%")
c3.metric("Sharpe Ratio",      f"{sharpe:.3f}",                  "↑ >1 is excellent")
c4.metric(f"VaR {var_conf*100:.0f}%", f"{var95*100:.2f}%",     "1-day loss estimate")
c5.metric("Max Drawdown",      f"{max_dd*100:.1f}%",            "Peak-to-trough")
c6.metric("ML AUC Score",      f"{ml_res['final_auc']:.3f}",    "5-fold walk-forward")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈  PRICE & RETURNS",
    "⚠️  RISK ANALYTICS",
    "🎯  PORTFOLIO OPT",
    "🤖  ML SIGNALS",
    "🗄️  SQL ANALYTICS",
    "📋  RISK REPORT",
])

# ════════════════════════════════════════════════════════════════
# TAB 1 — PRICE & RETURNS
# ════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown('<div class="section-title">Normalized Price Performance (Base = $1.00)</div>', unsafe_allow_html=True)
        norm_prices = prices[[t for t in sel_tickers if t != "SPY"]].div(
            prices[[t for t in sel_tickers if t != "SPY"]].iloc[0])
        fig = go.Figure()
        for i, t in enumerate([t for t in sel_tickers if t != "SPY"]):
            fig.add_trace(go.Scatter(x=norm_prices.index, y=norm_prices[t],
                                     name=t, line=dict(color=COLORS[i], width=1.8)))
        fig.add_hline(y=1, line_dash="dash", line_color="#333", line_width=1)
        T(fig, height=350, title="")


        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown('<div class="section-title">Key Insights</div>', unsafe_allow_html=True)
        best  = [t for t in sel_tickers if t!="SPY"]
        best_t= max(best, key=lambda t: returns[t].mean())
        worst_t=min(best, key=lambda t: returns[t].mean())
        st.markdown(f"""
        <div class="insight-card good">
          <div class="insight-title">🏆 Best Performer</div>
          <div class="insight-text">{best_t}: {returns[best_t].mean()*TRADING_DAYS*100:+.1f}% annual return</div>
        </div>
        <div class="insight-card warn">
          <div class="insight-title">⚠️ Highest Risk</div>
          <div class="insight-text">{max(best, key=lambda t: returns[t].std())}: {returns[max(best,key=lambda t:returns[t].std())].std()*np.sqrt(252)*100:.1f}% annual vol</div>
        </div>
        <div class="insight-card">
          <div class="insight-title">📊 Avg Sharpe</div>
          <div class="insight-text">Portfolio avg: {np.mean([(returns[t].mean()*252-rf_rate)/(returns[t].std()*np.sqrt(252)) for t in best]):.3f}</div>
        </div>
        <div class="insight-card">
          <div class="insight-title">🔗 Avg Correlation</div>
          <div class="insight-text">vs SPY: {np.mean([returns[t].corr(returns['SPY']) for t in best]):.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    # Returns distribution + Correlation
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-title">Daily Returns Distribution</div>', unsafe_allow_html=True)
        ret_data = prim_r.values
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=ret_data, nbinsx=60, name="Empirical",
                                    histnorm="probability density",
                                    marker_color=COLORS[0], opacity=0.7))
        x_r = np.linspace(ret_data.min(), ret_data.max(), 200)
        fig2.add_trace(go.Scatter(x=x_r, y=norm.pdf(x_r, ret_data.mean(), ret_data.std()),
                                  name="Normal Fit", line=dict(color=COLORS[1], width=2)))
        fig2.add_vline(x=var95, line_dash="dash", line_color="#ff4444",
                       annotation_text=f"VaR: {var95*100:.2f}%", annotation_font_color="#ff4444")
        T(fig2, height=300, title=f"{primary} Returns | Skew={skew(ret_data):.2f} | Kurt={kurtosis(ret_data):.2f}", showlegend=True)
        st.plotly_chart(fig2, width='stretch')

    with col4:
        st.markdown('<div class="section-title">Correlation Matrix</div>', unsafe_allow_html=True)
        corr_tickers = [t for t in sel_tickers if t != "SPY"]
        corr = returns[corr_tickers].corr()
        fig3 = px.imshow(corr, color_continuous_scale="RdYlGn",
                         zmin=-1, zmax=1, text_auto=".2f",
                         aspect="auto")
        T(fig3, height=300)


        fig3.update_coloraxes(colorbar=dict(tickfont=dict(color="#e0eaf2")))
        st.plotly_chart(fig3, width='stretch')

    # Drawdown chart
    st.markdown('<div class="section-title">Drawdown Analysis</div>', unsafe_allow_html=True)
    fig4 = go.Figure()
    for i, t in enumerate([t for t in sel_tickers if t!="SPY"]):
        dd = (prices[t] - prices[t].cummax()) / prices[t].cummax()
        fig4.add_trace(go.Scatter(x=dd.index, y=dd*100, name=t, fill="tozeroy",
                                  line=dict(color=COLORS[i], width=1),
                                  fillcolor=hex_to_rgba(COLORS[i], 0.15)))
    T(fig4, height=280, title="Portfolio Drawdowns (%)")
    fig.update_yaxes(ticksuffix="%", gridcolor="#1a3a5c")
    st.plotly_chart(fig4, width='stretch')


# ════════════════════════════════════════════════════════════════
# TAB 2 — RISK ANALYTICS
# ════════════════════════════════════════════════════════════════
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">VaR / CVaR Comparison</div>', unsafe_allow_html=True)
        tickers_no_spy = [t for t in sel_tickers if t!="SPY"]
        var_95 = [abs(np.percentile(returns[t], 5)) for t in tickers_no_spy]
        var_99 = [abs(np.percentile(returns[t], 1)) for t in tickers_no_spy]
        cvar_95= [abs(returns[t][returns[t]<=np.percentile(returns[t],5)].mean()) for t in tickers_no_spy]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="VaR 95%",  x=tickers_no_spy, y=[v*100 for v in var_95],  marker_color=COLORS[0]))
        fig.add_trace(go.Bar(name="VaR 99%",  x=tickers_no_spy, y=[v*100 for v in var_99],  marker_color="#ff4444"))
        fig.add_trace(go.Bar(name="CVaR 95%", x=tickers_no_spy, y=[v*100 for v in cvar_95], marker_color=COLORS[1]))
        T(fig, height=350, title="Daily Risk Metrics (%)", barmode="group")
        fig.update_yaxes(ticksuffix="%", gridcolor="#1a3a5c", showgrid=True)
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown('<div class="section-title">Monte Carlo Simulation</div>', unsafe_allow_html=True)
        np.random.seed(42)
        S0    = prices[primary].iloc[-1]
        mu_mc = prim_r.mean(); sig_mc = prim_r.std()
        sims  = np.zeros((mc_horizon, mc_paths))
        for i in range(mc_paths):
            path = [S0]
            for _ in range(mc_horizon-1):
                path.append(path[-1]*np.exp((mu_mc-0.5*sig_mc**2)+sig_mc*np.random.normal()))
            sims[:,i] = path

        fig = go.Figure()
        for i in range(0, mc_paths, max(1, mc_paths//50)):
            fig.add_trace(go.Scatter(y=sims[:,i], mode="lines",
                                     line=dict(color="#00c8e8", width=0.5),
                                     showlegend=False, opacity=0.15))
        for label, pct, color in [("5th %ile", 5, "#ff4444"),
                                   ("Median",  50, COLORS[1]),
                                   ("95th %ile",95, "#00cc66")]:
            fig.add_trace(go.Scatter(y=np.percentile(sims, pct, axis=1),
                                     name=label, line=dict(color=color, width=2.5)))
        T(fig, height=350, title=f"Monte Carlo: {primary} ({mc_paths} paths, {mc_horizon}d)")
        fig.update_yaxes(tickprefix="$", gridcolor="#1a3a5c", showgrid=True)
        st.plotly_chart(fig, width='stretch')

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-title">Volatility Regime Detection</div>', unsafe_allow_html=True)
        roll_vol = prim_r.rolling(21).std() * np.sqrt(TRADING_DAYS)
        high_vol = roll_vol > roll_vol.quantile(0.75)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol*100, name="21d Rolling Vol",
                                 line=dict(color=COLORS[0], width=1.5)))
        fig.add_hrect(y0=roll_vol.quantile(0.75)*100, y1=roll_vol.max()*100*1.05,
                      fillcolor="#ff4444", opacity=0.08, annotation_text="High Vol Regime",
                      annotation_font_color="#ff4444")
        fig.add_hline(y=roll_vol.mean()*100, line_dash="dot",
                      line_color=COLORS[1], annotation_text=f"Mean: {roll_vol.mean()*100:.1f}%")
        T(fig, height=300)
        fig.update_yaxes(ticksuffix="%", gridcolor="#1a3a5c", showgrid=True)


        st.plotly_chart(fig, width='stretch')

    with col4:
        st.markdown('<div class="section-title">Stress Test Scenarios</div>', unsafe_allow_html=True)
        scenarios = {"2008 GFC": -0.35, "COVID-19 Crash": -0.28,
                     "Flash Crash": -0.10, "Rate Shock": -0.15,
                     "Mild Correction": -0.08, "Bull Case": +0.20}
        n_sel     = len(tickers_no_spy)
        weights   = np.array([1/n_sel]*n_sel)
        pnl_results = {}
        for sc, shock in scenarios.items():
            impacts = np.random.normal(shock, 0.04, n_sel)
            pnl_results[sc] = float(weights @ impacts) * notional

        fig = go.Figure(go.Bar(
            x=list(pnl_results.values()),
            y=list(pnl_results.keys()),
            orientation="h",
            marker_color=["#00cc66" if v > 0 else "#ff4444" for v in pnl_results.values()],
            opacity=0.85
        ))
        fig.add_vline(x=0, line_color="#e0eaf2", line_width=1)
        T(fig, height=300, title=f"P&L Stress Test (${notional:,} Portfolio)")
        fig.update_xaxes(tickprefix="$", tickformat=",.0f", gridcolor="#1a3a5c", showgrid=True)
        st.plotly_chart(fig, width='stretch')

    # Rolling Sharpe
    st.markdown('<div class="section-title">Rolling 30-Day Sharpe Ratio</div>', unsafe_allow_html=True)
    fig = go.Figure()
    for i, t in enumerate(tickers_no_spy):
        rs = (returns[t].rolling(30).mean()*TRADING_DAYS - rf_rate) / (returns[t].rolling(30).std()*np.sqrt(TRADING_DAYS))
        fig.add_trace(go.Scatter(x=rs.index, y=rs, name=t, line=dict(color=COLORS[i], width=1.5)))
    fig.add_hline(y=1, line_dash="dot", line_color=COLORS[1], annotation_text="SR=1 (Good)")
    fig.add_hline(y=0, line_dash="dash", line_color="#444", line_width=1)
    T(fig, height=280)


    st.plotly_chart(fig, width='stretch')


# ════════════════════════════════════════════════════════════════
# TAB 3 — PORTFOLIO OPTIMIZATION
# ════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Markowitz Mean-Variance Optimization</div>', unsafe_allow_html=True)

    tickers_opt = [t for t in sel_tickers if t!="SPY"]
    ret_opt     = returns[tickers_opt]
    mean_r      = ret_opt.mean()
    cov_m       = ret_opt.cov()
    n_opt       = len(tickers_opt)

    def port_perf(w):
        r = (w @ mean_r.values) * TRADING_DAYS
        v = np.sqrt(w @ cov_m.values @ w) * np.sqrt(TRADING_DAYS)
        return r, v, (r-rf_rate)/v

    bounds = tuple((0.02, 0.45) for _ in range(n_opt))
    cons   = {"type":"eq","fun":lambda w: w.sum()-1}
    w0     = np.array([1/n_opt]*n_opt)

    res_ms = minimize(lambda w: -port_perf(w)[2], w0, method="SLSQP", bounds=bounds, constraints=cons)
    res_mv = minimize(lambda w:  port_perf(w)[1], w0, method="SLSQP", bounds=bounds, constraints=cons)

    ms_r, ms_v, ms_s = port_perf(res_ms.x)
    mv_r, mv_v, mv_s = port_perf(res_mv.x)

    # Efficient frontier
    ef_rets = np.linspace(mean_r.min()*252, mean_r.max()*252, 50)
    ef_vols = []
    for tr in ef_rets:
        c2 = [cons, {"type":"eq","fun":lambda w,r=tr: (w@mean_r.values)*252-r}]
        rr = minimize(lambda w: port_perf(w)[1], w0, method="SLSQP", bounds=bounds, constraints=c2)
        ef_vols.append(rr.fun if rr.success else np.nan)
    ef_v  = np.array(ef_vols)
    ef_sr = (ef_rets - rf_rate) / ef_v

    col1, col2 = st.columns([3, 2])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ef_v*100, y=ef_rets*100, mode="markers",
                                 marker=dict(color=ef_sr, colorscale="RdYlGn",
                                             size=8, colorbar=dict(title="Sharpe")),
                                 name="Efficient Frontier"))
        fig.add_trace(go.Scatter(x=[ms_v*100], y=[ms_r*100], mode="markers",
                                 marker=dict(color=COLORS[1], size=18, symbol="star"),
                                 name=f"Max Sharpe (SR={ms_s:.2f})"))
        fig.add_trace(go.Scatter(x=[mv_v*100], y=[mv_r*100], mode="markers",
                                 marker=dict(color=COLORS[0], size=15, symbol="diamond"),
                                 name=f"Min Variance (SR={mv_s:.2f})"))
        T(fig, height=420, title="Efficient Frontier — Markowitz Optimization")
        fig.update_xaxes(ticksuffix="%", title="Annual Volatility", gridcolor="#1a3a5c", showgrid=True)
        fig.update_yaxes(ticksuffix="%", title="Annual Return", gridcolor="#1a3a5c", showgrid=True)
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown("**Max Sharpe Portfolio**")
        c1f, c2f, c3f = st.columns(3)
        c1f.metric("Return",   f"{ms_r*100:.1f}%")
        c2f.metric("Vol",      f"{ms_v*100:.1f}%")
        c3f.metric("Sharpe",   f"{ms_s:.2f}")

        fig_w = go.Figure(go.Bar(
            x=[f"{w*100:.1f}%" for w in res_ms.x],
            y=tickers_opt,
            orientation="h",
            marker_color=COLORS[1], opacity=0.85,
            text=[f"{w*100:.1f}%" for w in res_ms.x],
            textposition="outside",
        ))
        T(fig_w, height=200, title="Max Sharpe Weights", showlegend=False)
        st.plotly_chart(fig_w, width='stretch')

        st.markdown("**Min Variance Portfolio**")
        c1g, c2g, c3g = st.columns(3)
        c1g.metric("Return", f"{mv_r*100:.1f}%")
        c2g.metric("Vol",    f"{mv_v*100:.1f}%")
        c3g.metric("Sharpe", f"{mv_s:.2f}")

        fig_w2 = go.Figure(go.Bar(
            x=[f"{w*100:.1f}%" for w in res_mv.x],
            y=tickers_opt, orientation="h",
            marker_color=COLORS[0], opacity=0.85,
            text=[f"{w*100:.1f}%" for w in res_mv.x], textposition="outside",
        ))
        T(fig_w2, height=200, title="Min Variance Weights", showlegend=False)
        st.plotly_chart(fig_w2, width='stretch')


# ════════════════════════════════════════════════════════════════
# TAB 4 — ML SIGNALS
# ════════════════════════════════════════════════════════════════
with tab4:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
        fi = pd.Series(ml_res["model"].feature_importances_,
                       index=ml_res["feat_names"]).nlargest(15)
        fig = go.Figure(go.Bar(
            x=fi.values[::-1], y=fi.index[::-1],
            orientation="h",
            marker_color=[COLORS[0] if v > fi.median() else "#2e4a5c" for v in fi.values[::-1]],
        ))
        fig.add_vline(x=fi.median(), line_dash="dot", line_color=COLORS[1])
        T(fig, height=380, title="Top 15 Features")


        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown('<div class="section-title">ROC Curve & Walk-Forward AUC</div>', unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(ml_res["all_true"], ml_res["all_probs"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC (AUC={ml_res['final_auc']:.3f})",
                                 line=dict(color=COLORS[1], width=2.5), fill="tozeroy",
                                 fillcolor="rgba(181,166,66,0.1)"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random", mode="lines",
                                 line=dict(color="#444", dash="dash")))
        T(fig, height=380, title=f"ROC Curve | 5-Fold Walk-Forward CV | AUC={ml_res['final_auc']:.3f}")
        fig.update_xaxes(title="False Positive Rate", gridcolor="#1a3a5c", showgrid=True)
        fig.update_yaxes(title="True Positive Rate", gridcolor="#1a3a5c", showgrid=True)
        st.plotly_chart(fig, width='stretch')

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-title">RSI + MACD Signals</div>', unsafe_allow_html=True)
        feat_plot = features.iloc[-150:]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5,0.5],
                            vertical_spacing=0.05)
        fig.add_trace(go.Scatter(x=feat_plot.index, y=feat_plot["rsi_14"],
                                 line=dict(color=COLORS[0]), name="RSI(14)"), row=1, col=1)
        fig.add_hline(y=70, line_color="#ff4444", line_dash="dash", row=1, col=1)
        fig.add_hline(y=30, line_color="#00cc66", line_dash="dash", row=1, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="#ff4444", opacity=0.07, row=1, col=1)
        fig.add_hrect(y0=0, y1=30,   fillcolor="#00cc66", opacity=0.07, row=1, col=1)
        fig.add_trace(go.Scatter(x=feat_plot.index, y=feat_plot["macd"],
                                 line=dict(color=COLORS[0]), name="MACD"), row=2, col=1)
        fig.add_trace(go.Scatter(x=feat_plot.index, y=feat_plot["macd_sig"],
                                 line=dict(color=COLORS[1]), name="Signal"), row=2, col=1)
        fig.add_trace(go.Bar(x=feat_plot.index, y=feat_plot["macd_hist"],
                             marker_color=[COLORS[2] if v>=0 else "#ff4444" for v in feat_plot["macd_hist"]],
                             name="Histogram", showlegend=False), row=2, col=1)
        T(fig, height=380, title="Technical Indicators")


        fig.update_yaxes(range=[0,100], row=1, col=1)
        st.plotly_chart(fig, width='stretch')

    with col4:
        st.markdown('<div class="section-title">ML Strategy Backtest</div>', unsafe_allow_html=True)
        X_all = ml_res["scaler"].transform(
            features.drop("target", axis=1).values)
        probs_all = ml_res["model"].predict_proba(X_all)[:,1]
        prob_ser  = pd.Series(probs_all, index=features.index)
        ret_ser   = np.log(prices["GS"]/prices["GS"].shift(1)).reindex(features.index)
        signal    = (prob_ser > 0.60).astype(int)
        strat_ret = signal.shift(1) * ret_ser
        cum_strat = (1+strat_ret.fillna(0)).cumprod()
        cum_bah   = (1+ret_ser.fillna(0)).cumprod()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum_strat.index, y=cum_strat, name="ML Strategy",
                                 line=dict(color=COLORS[1], width=2.5)))
        fig.add_trace(go.Scatter(x=cum_bah.index, y=cum_bah, name="Buy & Hold",
                                 line=dict(color=COLORS[0], width=2)))
        fig.add_hline(y=1, line_dash="dash", line_color="#333")
        wr = (strat_ret>0).mean()
        sr_bt = strat_ret.mean()/strat_ret.std()*np.sqrt(252)
        T(fig, height=380, title=f"Strategy (WinRate={wr*100:.0f}% | SR={sr_bt:.2f}) vs Buy&Hold")
        st.plotly_chart(fig, width='stretch')

    # Anomaly detection
    st.markdown('<div class="section-title">Anomaly Detection (Isolation Forest)</div>', unsafe_allow_html=True)
    anom_feat = features[["ret_1d","vol_5d","rsi_14","vol_rel"]].dropna()
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    anom_labels = iso.fit_predict(anom_feat)
    anom_scores = iso.score_samples(anom_feat)
    fig = go.Figure()
    for label, color, name in [(1, COLORS[0], "Normal"), (-1, "#ff4444", "Anomaly")]:
        mask = anom_labels==label
        fig.add_trace(go.Scatter(
            x=anom_feat.loc[mask, "ret_1d"]*100,
            y=anom_feat.loc[mask, "vol_5d"]*100,
            mode="markers", name=f"{name} ({mask.sum()})",
            marker=dict(color=color, size=5, opacity=0.6)
        ))
    T(fig, height=300, title=f"Isolation Forest: {(anom_labels==-1).sum()} anomalies in {len(anom_labels)} observations")
    fig.update_xaxes(title="1-Day Return (%)", ticksuffix="%", gridcolor="#1a3a5c", showgrid=True)
    fig.update_yaxes(title="5-Day Rolling Vol (%)", ticksuffix="%", gridcolor="#1a3a5c", showgrid=True)
    st.plotly_chart(fig, width='stretch')


# ════════════════════════════════════════════════════════════════
# TAB 5 — SQL ANALYTICS
# ════════════════════════════════════════════════════════════════
with tab5:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Monthly Returns Heatmap (SQL GROUP BY)</div>', unsafe_allow_html=True)
        monthly = prim_r.resample("ME").sum() * 100
        monthly_df = pd.DataFrame({
            "Year":  monthly.index.year,
            "Month": monthly.index.month_name().str[:3],
            "Return": monthly.values
        })
        pivot = monthly_df.pivot_table(values="Return", index="Year", columns="Month",
                                        aggfunc="first")
        month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        pivot = pivot[[m for m in month_order if m in pivot.columns]]
        fig = px.imshow(pivot, color_continuous_scale="RdYlGn",
                        text_auto=".1f", aspect="auto",
                        color_continuous_midpoint=0)
        T(fig, height=300, title=f"{primary} Monthly Returns (%)")


        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown('<div class="section-title">Volatility Regime (SQL CASE WHEN)</div>', unsafe_allow_html=True)
        rv21 = prim_r.rolling(21).std() * np.sqrt(TRADING_DAYS)
        regime = np.where(rv21 > rv21.quantile(0.75), "HIGH_VOL",
                 np.where(rv21 < rv21.quantile(0.25), "LOW_VOL", "NORMAL"))
        regime_counts = pd.Series(regime).value_counts()
        fig = go.Figure(go.Pie(
            labels=regime_counts.index, values=regime_counts.values,
            hole=0.55,
            marker=dict(colors=["#ff4444", COLORS[0], "#00cc66"]),
        ))
        T(fig, height=300, title="Regime Distribution", legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig, width='stretch')

    st.markdown('<div class="section-title">P&L Attribution (SQL Window Functions — LAG + SUM)</div>', unsafe_allow_html=True)
    pnl_data = []
    for t in tickers_no_spy:
        daily_pnl = prices[t].diff() * (notional / prices[t].iloc[0])
        pnl_data.append({
            "Ticker": t,
            "Total P&L": daily_pnl.sum(),
            "Best Day":  daily_pnl.max(),
            "Worst Day": daily_pnl.min(),
            "Win Days":  (daily_pnl > 0).sum(),
            "Loss Days": (daily_pnl < 0).sum(),
        })
    pnl_df = pd.DataFrame(pnl_data).set_index("Ticker")

    fig = go.Figure()
    colors_pnl = [COLORS[2] if v > 0 else "#ff4444" for v in pnl_df["Total P&L"]]
    fig.add_trace(go.Bar(x=pnl_df.index, y=pnl_df["Total P&L"],
                         marker_color=colors_pnl, name="Total P&L",
                         text=[f"${v:+,.0f}" for v in pnl_df["Total P&L"]],
                         textposition="outside"))
    T(fig, height=300, title=f"Total P&L per Stock (${notional:,} Notional)")
    fig.update_yaxes(tickprefix="$", tickformat=",.0f", gridcolor="#1a3a5c", showgrid=True)
    st.plotly_chart(fig, width='stretch')

    st.markdown('<div class="section-title">Z-Score Outliers (SQL 3-Sigma Events)</div>', unsafe_allow_html=True)
    z_scores_out = []
    for t in tickers_no_spy:
        rm, rs = returns[t].rolling(20).mean(), returns[t].rolling(20).std()
        z      = (returns[t] - rm) / rs
        extreme= z[z.abs() > 3].dropna()
        for dt, zv in extreme.items():
            z_scores_out.append({"Date": dt, "Ticker": t, "Z-Score": round(float(zv),3),
                                  "Return %": round(float(returns[t][dt]*100), 3)})
    if z_scores_out:
        z_df = pd.DataFrame(z_scores_out).sort_values("Z-Score", key=abs, ascending=False).head(20)
        z_df["Date"] = z_df["Date"].astype(str)
        st.dataframe(z_df, width='stretch', height=250)
    else:
        st.info("No 3-sigma events detected in selected period.")


# ════════════════════════════════════════════════════════════════
# TAB 6 — RISK REPORT
# ════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-title">Full Risk Metrics Report</div>', unsafe_allow_html=True)

    display_df = risk_df[[c for c in risk_df.columns if "Return" in c or "Vol" in c or "Sharpe" in c or "VaR" in c or "CVaR" in c or "Drawdown" in c or "Beta" in c or "Alpha" in c]].copy()

    def fmt(df):
        for col in df.columns:
            if "Return" in col or "Vol" in col or "VaR" in col or "CVaR" in col or "Drawdown" in col or "Alpha" in col:
                df[col] = df[col].map(lambda x: f"{x*100:.2f}%")
            elif "Sharpe" in col or "Sortino" in col or "Beta" in col:
                df[col] = df[col].map(lambda x: f"{x:.3f}")
        return df
    st.dataframe(fmt(display_df.copy()), width='stretch', height=300)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Sharpe vs Volatility Scatter</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for i, t in enumerate(risk_df.index):
            fig.add_trace(go.Scatter(
                x=[risk_df.loc[t,"Annual Vol"]*100],
                y=[risk_df.loc[t,"Sharpe"]],
                mode="markers+text",
                text=[t], textposition="top center",
                marker=dict(size=15, color=COLORS[i%len(COLORS)]),
                name=t
            ))
        fig.add_hline(y=1, line_dash="dot", line_color=COLORS[1], annotation_text="SR=1")
        fig.add_hline(y=0, line_dash="dash", line_color="#333")
        T(fig, height=350, title="Risk-Return Scatter", showlegend=False)
        fig.update_xaxes(ticksuffix="%", title="Annual Volatility", gridcolor="#1a3a5c", showgrid=True)
        fig.update_yaxes(title="Sharpe Ratio", gridcolor="#1a3a5c", showgrid=True)
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown('<div class="section-title">Beta & Alpha (CAPM Decomposition)</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Beta", x=risk_df.index,
                             y=risk_df["Beta"], marker_color=COLORS[0], yaxis="y"))
        fig.add_trace(go.Bar(name="Alpha (%)", x=risk_df.index,
                             y=risk_df["Alpha"]*100, marker_color=COLORS[1], yaxis="y2"))
        fig.update_layout(
            paper_bgcolor="#050f1a", plot_bgcolor="#0a1628",
            font=dict(color="#e0eaf2", family="monospace", size=11),
            legend=dict(bgcolor="rgba(10,22,40,0.8)", bordercolor="#1a3a5c", borderwidth=1),
            margin=dict(l=50, r=20, t=40, b=40),
            height=350,
            yaxis=dict(title="Beta", gridcolor="#1a3a5c", showgrid=True),
            yaxis2=dict(title="Alpha (%)", overlaying="y", side="right", ticksuffix="%"),
            barmode="group",
            title="CAPM: Beta (Systematic Risk) & Alpha (Skill)")
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    st.markdown(f"""
    <div style='font-family:monospace; font-size:0.65rem; color:#5a8aaa; 
         background:#0a1628; border:1px solid #1a3a5c; padding:1rem; border-radius:2px;'>
    <b style='color:#b5a642;'>. — PROJECT SUMMARY</b><br><br>
    Data Period: {prices.index[0].date()} → {prices.index[-1].date()} ({len(prices)} trading days)<br>
    Stocks Analyzed: {', '.join([t for t in sel_tickers if t!='SPY'])} + SPY (benchmark)<br>
    Risk-Free Rate: {rf_rate*100:.1f}% | Notional: ${notional:,}<br><br>
    <b style='color:#00c8e8;'>Technologies:</b> Python · Pandas · NumPy · Scikit-learn · SciPy · Plotly · Streamlit<br>
    <b style='color:#00c8e8;'>Methods:</b> GBM Simulation · VaR/CVaR · CAPM · Markowitz · Random Forest · Isolation Forest · Walk-Forward CV
    </div>
    """, unsafe_allow_html=True)
