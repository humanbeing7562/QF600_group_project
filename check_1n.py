"""
Are the good sweep results just a scaled-down 1/N?

Two checks:
  1. Weights: for each MVO / ridge variant, how much of the book is net long,
     what share of positions are long, and how similar the weights are to equal
     weights (cosine similarity, 1 = same direction as 1/N).
  2. Returns: regress each strategy's daily return on the 1/N daily return,
         r = a + b * r_1N + e
     b = how much 1/N it holds, R^2 = how much of its daily moves 1/N explains,
     and the Sharpe of the residual (a + e) = what is left after removing 1/N.

Reads:
  sweep_results/mvo/, sweep_results/ridge/        monthly weights
  sweep_results/ensemble/daily_returns.csv        ensemble runs
  ensemble_results/daily_returns.csv              the 1/N column
Writes:
  sweep_results/check_1n_weights.csv, sweep_results/check_1n_returns.csv

Usage:
  python check_1n.py
"""

import os

import numpy as np
import pandas as pd

from ensemble import START, END, load_stock_returns, sleeve_returns
from sweep_ensemble import load_monthly_as_daily


WEIGHT_FILES = {
    "MVO lam4": os.path.join("sweep_results", "mvo", "weights_naive_mvo_lam4.csv"),
    "MVO lam200": os.path.join("sweep_results", "mvo", "weights_naive_mvo_lam200.csv"),
    "Ridge full": os.path.join("sweep_results", "ridge", "weights_ridge_mvo_full.csv"),
    "Ridge from1e-4": os.path.join("sweep_results", "ridge", "weights_ridge_mvo_from1e-4.csv"),
    "Ridge from1e-2": os.path.join("sweep_results", "ridge", "weights_ridge_mvo_from1e-2.csv"),
}

# ensemble runs to test (column names in sweep_results/ensemble/daily_returns.csv)
ENSEMBLE_RUNS = [
    "lam4_full_sig2_top20_Average",          # baseline
    "lam200_from1e-2_sig1_top5_Average",     # best Average
    "lam200_from1e-2_sig2_top20_Average",
    "lam200_from1e-4_sig3_p101_120_FTL_daily",
]

TRADING_DAYS = 252


def weight_stats(path):
    monthly = pd.read_csv(path, index_col=0)
    rows = []
    for month in monthly.index:
        w = monthly.loc[month]
        held = w[w != 0]
        n = len(held)
        equal = np.ones(n) / n
        cosine = float(np.dot(held.values, equal) /
                       (np.linalg.norm(held.values) * np.linalg.norm(equal)))
        n_long = 0
        for x in held.values:
            if x > 0:
                n_long = n_long + 1
        rows.append({
            "net": held.sum(),
            "gross": held.abs().sum(),
            "pct_long": n_long / n,
            "cosine_vs_1N": cosine,
        })
    return pd.DataFrame(rows).mean()


def regress_on_1n(r, r1n):
    x = r1n.values
    y = r.values
    b = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    a = y.mean() - b * x.mean()
    resid = y - b * x                       # a + e, the part not explained by 1/N
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    sharpe = y.mean() / y.std(ddof=1) * np.sqrt(TRADING_DAYS)
    resid_sharpe = resid.mean() / resid.std(ddof=1) * np.sqrt(TRADING_DAYS)
    return {
        "beta_on_1N": b,
        "r2": r2,
        "alpha_annual": a * TRADING_DAYS,
        "sharpe": sharpe,
        "sharpe_after_removing_1N": resid_sharpe,
    }


def main():
    print("1. weights (averaged over months)")
    weight_rows = []
    for name in WEIGHT_FILES:
        s = weight_stats(WEIGHT_FILES[name])
        row = {"strategy": name}
        for key in s.index:
            row[key] = s[key]
        weight_rows.append(row)
    weight_table = pd.DataFrame(weight_rows)
    print(weight_table.round(3).to_string(index=False))
    weight_table.to_csv(os.path.join("sweep_results", "check_1n_weights.csv"), index=False)

    print()
    print("2. daily returns regressed on 1/N")
    ens = pd.read_csv(os.path.join("ensemble_results", "daily_returns.csv"),
                      index_col=0, parse_dates=True)
    r1n = ens["1/N"].loc[START:END]

    series = {}
    returns = load_stock_returns()
    for name in WEIGHT_FILES:
        daily_w = load_monthly_as_daily(WEIGHT_FILES[name], list(returns.columns), returns.index)
        series[name] = sleeve_returns(daily_w, returns).loc[START:END].fillna(0.0)

    sweep = pd.read_csv(os.path.join("sweep_results", "ensemble", "daily_returns.csv"),
                        index_col=0, parse_dates=True)
    for col in ENSEMBLE_RUNS:
        series[col] = sweep[col]

    series["1/N itself"] = r1n

    return_rows = []
    for name in series:
        r = series[name].reindex(r1n.index).fillna(0.0)
        row = {"strategy": name}
        stats = regress_on_1n(r, r1n)
        for key in stats:
            row[key] = stats[key]
        return_rows.append(row)
    return_table = pd.DataFrame(return_rows)
    pd.set_option("display.width", 200)
    print(return_table.round(3).to_string(index=False))
    return_table.to_csv(os.path.join("sweep_results", "check_1n_returns.csv"), index=False)


if __name__ == "__main__":
    main()