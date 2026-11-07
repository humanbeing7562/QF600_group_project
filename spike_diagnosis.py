"""
Why did MVO / Ridge / the Average spike in late 2017 - early 2018?

Idea: a portfolio with net exposure N (sum of weights) behaves partly like N x the
market. So each day's return splits into
    market part = N x S&P 500 return that day
    rest        = everything else (the long-short bets between stocks)
If the market part explains most of the spike, the spike was a levered market bet.

Reads:
  data/ohlcv_raw.csv, data/spy.csv
  naive_mvo_results/weights_naive_mvo.csv, ridge_mvo_results/weights_ridge_mvo.csv (monthly)
  pairs_results/pairs_weights_daily.csv
  ensemble_results/daily_returns.csv  (run ensemble.py first)

Writes (in spike_diagnosis/):
  monthly_breakdown.csv      per month and strategy: gross, net, S&P return,
                             strategy return, market part, rest
  top_contributors.csv       stocks that made / lost the most in the biggest months
  exposure_vs_volatility.png net exposure vs the market's trailing volatility

Usage:
  python spike_diagnosis.py
  python spike_diagnosis.py --start 2017-06 --end 2018-12
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = "spike_diagnosis"
INK = "#2b2b29"
MUTED = "#8a897f"
GRID = "#e6e5df"
SURFACE = "#fcfcfb"
COLORS = {"MVO": "#eda100", "Ridge": "#e87ba4", "Average": "#2a78d6"}


def load_daily_weights():
    """Daily weights held at each close, same rule as ensemble.py."""
    px = pd.read_csv(os.path.join("data", "ohlcv_raw.csv"), parse_dates=["date"])
    prices = px.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    returns = prices.ffill().pct_change(fill_method=None).fillna(0.0)
    dates = returns.index
    tickers = list(returns.columns)

    weights = {}
    monthly_files = {
        "MVO": os.path.join("naive_mvo_results", "weights_naive_mvo.csv"),
        "Ridge": os.path.join("ridge_mvo_results", "weights_ridge_mvo.csv"),
    }
    for name in monthly_files:
        monthly = pd.read_csv(monthly_files[name], index_col=0)
        monthly.index = monthly.index.astype(str)
        monthly = monthly.reindex(columns=tickers).fillna(0.0)
        rows = []
        for i in range(len(dates)):
            if i + 1 < len(dates):
                month = str(dates[i + 1].to_period("M"))
            else:
                month = str(dates[i].to_period("M"))
            if month in monthly.index:
                rows.append(monthly.loc[month].values)
            else:
                rows.append(np.zeros(len(tickers)))
        weights[name] = pd.DataFrame(rows, index=dates, columns=tickers)

    pairs = pd.read_csv(os.path.join("pairs_results", "pairs_weights_daily.csv"),
                        index_col=0, parse_dates=True)
    if "n_cohorts" in pairs.columns:
        pairs = pairs.drop(columns=["n_cohorts"])
    weights["Pairs"] = pairs.reindex(index=dates, columns=tickers).fillna(0.0)
    weights["Average"] = (weights["MVO"] + weights["Ridge"] + weights["Pairs"]) / 3
    return weights, returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2017-06")
    parser.add_argument("--end", default="2018-12")
    args = parser.parse_args()
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    weights, returns = load_daily_weights()
    daily = pd.read_csv(os.path.join("ensemble_results", "daily_returns.csv"),
                        index_col=0, parse_dates=True)
    spy = daily["S&P 500"]
    dates = daily.index
    names = ["MVO", "Ridge", "Average"]

    # net and gross exposure held going INTO each day (yesterday's close)
    exposure = {}
    for name in names:
        held = weights[name].shift(1).reindex(dates)
        exposure[name] = pd.DataFrame({
            "net": held.sum(axis=1),
            "gross": held.abs().sum(axis=1),
        })

    # daily split into market part and the rest
    months = dates.to_period("M")
    rows = []
    for name in names:
        strat = daily[name]
        market_part = exposure[name]["net"] * spy
        rest = strat - market_part
        for m in months.unique():
            mask = months == m
            rows.append({
                "month": str(m),
                "strategy": name,
                "gross_exposure": exposure[name]["gross"][mask].mean(),
                "net_exposure": exposure[name]["net"][mask].mean(),
                "sp500_return": spy[mask].sum(),
                "strategy_return": strat[mask].sum(),
                "market_part": market_part[mask].sum(),
                "rest": rest[mask].sum(),
            })
    breakdown = pd.DataFrame(rows)
    breakdown.to_csv(os.path.join(OUT_DIR, "monthly_breakdown.csv"), index=False)

    pd.set_option("display.width", 200)
    window = breakdown[(breakdown["month"] >= args.start) & (breakdown["month"] <= args.end)]
    for name in names:
        block = window[window["strategy"] == name].drop(columns=["strategy"]).copy()
        for col in ["sp500_return", "strategy_return", "market_part", "rest"]:
            formatted = []
            for x in block[col]:
                formatted.append("{:+.1f}%".format(x * 100))
            block[col] = formatted
        block["gross_exposure"] = block["gross_exposure"].round(1)
        block["net_exposure"] = block["net_exposure"].round(1)
        print()
        print(name + " (returns are sums of daily returns, i.e. not compounded)")
        print(block.to_string(index=False))

    # how much of each strategy's daily swings the market part explains
    print()
    print("share of daily return variance explained by net exposure x S&P 500:")
    periods = [("2017-01-01", "2018-12-31"), ("2019-01-01", "2025-12-31"), ("2017-01-01", "2025-12-31")]
    for name in names:
        parts = []
        for start, end in periods:
            s = daily[name].loc[start:end]
            mp = (exposure[name]["net"] * spy).loc[start:end]
            r2 = np.corrcoef(s, mp)[0, 1] ** 2
            parts.append("{}-{}: {:.0%}".format(start[:4], end[:4], r2))
        print("  {:8s} {}".format(name, "   ".join(parts)))

    # biggest months for the Average: which stocks drove them
    avg = breakdown[breakdown["strategy"] == "Average"].copy()
    avg["abs_ret"] = avg["strategy_return"].abs()
    biggest = avg.sort_values("abs_ret", ascending=False).head(4)["month"].tolist()
    contrib_rows = []
    held = weights["Average"].shift(1).reindex(dates)
    r = returns.reindex(dates)
    for m in biggest:
        mask = months == pd.Period(m, "M")
        contrib = (held[mask] * r[mask]).sum(axis=0)
        ordered = contrib.sort_values()
        picks = list(ordered.head(5).items()) + list(ordered.tail(5)[::-1].items())
        for ticker, value in picks:
            contrib_rows.append({
                "month": m,
                "ticker": ticker,
                "contribution": value,
                "avg_weight": held.loc[mask, ticker].mean(),
                "stock_return": r.loc[mask, ticker].sum(),
            })
    contrib = pd.DataFrame(contrib_rows)
    contrib.to_csv(os.path.join(OUT_DIR, "top_contributors.csv"), index=False)
    print()
    print("biggest Average months (by size of return):", ", ".join(biggest))
    print("top stock contributors saved to", os.path.join(OUT_DIR, "top_contributors.csv"))

    # chart: net exposure (top) and the market's trailing 1-year volatility (bottom)
    trailing_vol = spy.rolling(252, min_periods=60).std() * np.sqrt(252) * 100
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2]})
    fig.patch.set_facecolor(SURFACE)
    ax = axes[0]
    for name in names:
        net = exposure[name]["net"]
        ax.plot(net.index, net.values, color=COLORS[name], linewidth=1.8, label=name)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_title("Net exposure (sum of weights) held each day", color=INK, loc="left", fontsize=12)
    ax.set_ylabel("Net exposure (x capital)", color=INK)
    ax.legend(frameon=False, loc="upper right", labelcolor=INK)

    ax = axes[1]
    ax.plot(trailing_vol.index, trailing_vol.values, color=MUTED, linewidth=1.8)
    ax.set_title("S&P 500 volatility over the previous year", color=INK, loc="left", fontsize=12)
    ax.set_ylabel("Annualized vol (%)", color=INK)

    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "exposure_vs_volatility.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    main()