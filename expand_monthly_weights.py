"""
Expand monthly weights (one row per investment month) into daily weights.

Input: a CSV like weights_naive_mvo.csv
    investment_month, AAPL, ABBV, ...
    2017-01,          -0.58, 2.64, ...
  where the row for 2017-01 is the portfolio HELD during January 2017.

Output: one row per trading day, same convention as pairs_weights_daily.csv:
  the row dated d = weights held at the close of day d, earning day d+1's return.
  So the last trading day of December 2016 already carries the 2017-01 weights,
  and every day in January carries them too, except Jan's last day, which
  carries the February weights.

Usage:
  python expand_monthly_weights.py --weights weights_naive_mvo.csv --out weights_naive_mvo_daily.csv
  python expand_monthly_weights.py --weights weights_ridge.csv --out weights_ridge_daily.csv

Trading days come from --dates (default results/pairs_weights_daily.csv), so the
output lines up row for row and column for column with the pairs weights.
"""

import argparse

import pandas as pd


def load_trading_days(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.index


def load_tickers(path):
    header = pd.read_csv(path, index_col=0, nrows=0)
    tickers = []
    for col in header.columns:
        if col != "n_cohorts":
            tickers.append(col)
    return tickers


def expand(monthly, days):
    """For each day, use the weights of the month that the NEXT trading day falls in."""
    rows = []
    for i in range(len(days)):
        if i + 1 < len(days):
            holding_month = str(days[i + 1].to_period("M"))
        else:
            holding_month = str(days[i].to_period("M"))

        if holding_month in monthly.index:
            rows.append(monthly.loc[holding_month].values)
        else:
            rows.append([0.0] * monthly.shape[1])     # no portfolio yet: cash

    daily = pd.DataFrame(rows, index=days, columns=monthly.columns)
    daily.index.name = "date"
    return daily


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="monthly weights CSV")
    parser.add_argument("--out", required=True, help="daily weights CSV to write")
    parser.add_argument("--dates", default="pairs_results/pairs_weights_daily.csv",
                        help="CSV whose date index gives the trading days and ticker columns")
    args = parser.parse_args()

    monthly = pd.read_csv(args.weights, index_col=0)
    monthly.index = monthly.index.astype(str)

    days = load_trading_days(args.dates)
    tickers = load_tickers(args.dates)

    # line the columns up with the pairs file: missing tickers get 0
    extra = []
    for t in monthly.columns:
        if t not in tickers:
            extra.append(t)
    missing = []
    for t in tickers:
        if t not in monthly.columns:
            missing.append(t)
    if len(extra) > 0:
        print("warning: tickers in the monthly file but not in the pairs file (dropped):", extra)
    if len(missing) > 0:
        print("note: tickers not in the monthly file, set to 0:", missing)
    monthly = monthly.reindex(columns=tickers).fillna(0.0)

    daily = expand(monthly, days)
    daily.to_csv(args.out)

    # quick summary
    first_month = monthly.index[0]
    held = daily[daily.abs().sum(axis=1) > 0]
    print("monthly rows:", len(monthly), " (", first_month, "to", monthly.index[-1], ")")
    print("daily rows:", len(daily), " first day holding weights:", held.index[0].date())
    print("saved", args.out)


if __name__ == "__main__":
    main()