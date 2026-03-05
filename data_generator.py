

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# ── Reproducibility ──────────────────────────────────────────────
np.random.seed(42)

# ── Parameters — GS Portfolio Universe ──────────────────────────
STOCKS = {
    "GS":   {"S0": 420.00, "mu": 0.15, "sigma": 0.22},   # .
    "JPM":  {"S0": 198.00, "mu": 0.12, "sigma": 0.20},   # J.P. Morgan
    "MS":   {"S0": 102.00, "mu": 0.13, "sigma": 0.24},   # Morgan Stanley
    "BAC":  {"S0": 38.00,  "mu": 0.10, "sigma": 0.26},   # Bank of America
    "C":    {"S0": 62.00,  "mu": 0.09, "sigma": 0.28},   # Citigroup
    "WFC":  {"S0": 55.00,  "mu": 0.11, "sigma": 0.23},   # Wells Fargo
    "BLK":  {"S0": 830.00, "mu": 0.14, "sigma": 0.19},   # BlackRock
    "SPY":  {"S0": 510.00, "mu": 0.10, "sigma": 0.15},   # S&P 500 ETF (benchmark)
}

TRADING_DAYS = 504   # 2 years of data
START_DATE   = datetime(2022, 1, 3)


def generate_gbm_prices(S0, mu, sigma, n_days, dt=1/252):
    """
    Geometric Brownian Motion:  dS = μS dt + σS dW
    Same stochastic model used in Black-Scholes & GS risk systems.
    """
    prices = [S0]
    for _ in range(n_days - 1):
        drift     = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * np.random.normal()
        prices.append(prices[-1] * np.exp(drift + diffusion))
    return np.array(prices)


def generate_trading_dates(start, n_days):
    dates, current = [], start
    while len(dates) < n_days:
        if current.weekday() < 5:          # Mon–Fri only
            dates.append(current)
        current += timedelta(days=1)
    return dates


def add_market_events(price_series, dates):
    """Inject realistic market shocks (earnings, Fed events, crashes)."""
    prices = price_series.copy()
    events = {}

    # Simulate ~4 volatility spikes per year
    shock_days = np.random.choice(len(prices) - 5, size=8, replace=False)
    for day in shock_days:
        shock = np.random.choice([-1, 1]) * np.random.uniform(0.03, 0.10)
        prices[day:day+3] *= (1 + shock * np.array([1, -0.4, 0.2]))
        events[dates[day]] = f"Market Event: {shock*100:+.1f}%"

    return prices, events


def generate_volume(prices, base_volume=1_000_000):
    """Volume inversely correlated with price change magnitude."""
    returns     = np.diff(np.log(prices))
    vol_factor  = 1 + 2 * np.abs(returns) + np.random.exponential(0.5, len(returns))
    volume      = base_volume * vol_factor
    volume      = np.append(base_volume, volume).astype(int)
    return volume


def build_ohlcv(close_prices, volume):
    """Construct realistic OHLCV from close prices."""
    rows = []
    for i, (close, vol) in enumerate(zip(close_prices, volume)):
        noise  = np.random.uniform(0.003, 0.015)
        high   = close * (1 + noise)
        low    = close * (1 - noise)
        open_  = low + np.random.uniform(0, 1) * (high - low)
        rows.append({"open": round(open_, 2),
                     "high": round(high, 2),
                     "low":  round(low, 2),
                     "close": round(close, 2),
                     "volume": vol})
    return rows


def main():
    os.makedirs("data", exist_ok=True)
    dates = generate_trading_dates(START_DATE, TRADING_DAYS)

    all_close = {}
    print("=" * 60)
    print("  . Analytics Project — Data Generator")
    print("=" * 60)

    for ticker, params in STOCKS.items():
        print(f"  Generating {ticker:4s} | S0=${params['S0']:,.0f} "
              f"| μ={params['mu']:.0%} | σ={params['sigma']:.0%}")

        prices, events = generate_market_events(
            generate_gbm_prices(params["S0"], params["mu"],
                                params["sigma"], TRADING_DAYS),
            dates
        )
        volume = generate_volume(prices,
                                 base_volume=int(1e6 / params["S0"] * 500))
        ohlcv  = build_ohlcv(prices, volume)

        df = pd.DataFrame(ohlcv, index=pd.DatetimeIndex(dates))
        df.index.name = "date"
        df.to_csv(f"data/{ticker}.csv")
        all_close[ticker] = prices

    # ── Consolidated close-price matrix ──────────────────────────
    close_df = pd.DataFrame(all_close,
                            index=pd.DatetimeIndex(dates))
    close_df.index.name = "date"
    close_df.to_csv("data/close_prices.csv")

    # ── Market metadata ──────────────────────────────────────────
    meta = []
    for ticker, p in STOCKS.items():
        meta.append({
            "ticker":    ticker,
            "sector":    "Financials",
            "annual_return_pct": round(p["mu"] * 100, 1),
            "volatility_pct":   round(p["sigma"] * 100, 1),
            "initial_price":    p["S0"]
        })
    pd.DataFrame(meta).to_csv("data/metadata.csv", index=False)

    print("\n  Data saved to /data/  (504 trading days × 8 tickers)")
    print("=" * 60)


def generate_market_events(prices, dates):
    return add_market_events(prices, dates)


if __name__ == "__main__":
    main()
