"""
Replication of Gatev, Goetzmann & Rouwenhorst (2006) pairs trading.

Reads data/ohlcv_raw.csv (columns: date, ticker, open, high, low, close, adj_close, volume)
and writes results to results/.

Rule (same as the paper):
  - Formation: 12 months. Keep stocks with a price on every day and no zero-volume days.
    Normalize each stock's total return index (adj_close) to 1 on the first day.
    Rank all pairs by sum of squared deviations (SSD) between normalized prices.
  - Trading: next 6 months. Open when |spread| > 2 sigma (sigma from formation),
    short the higher stock, long the lower stock, $1 each side.
    Close when the normalized prices cross. Force close on the last day.
  - A new cohort starts every month, so up to 6 cohorts trade at once.
    Monthly return = average of the active cohorts' monthly returns.
  - Two timing versions: wait 0 days (trade at the signal close) and wait 1 day.

Usage:
  python pairs_replication.py
  python pairs_replication.py --data data/ohlcv_raw.csv --out results
"""

import argparse
import os

import numpy as np
import pandas as pd


FORMATION_MONTHS = 12
TRADING_MONTHS = 6
TRIGGER_SIGMAS = 2.0
NW_LAGS = 6                 # Newey-West lags, as in the paper

# rank ranges, 0-based and end-exclusive: top 5, top 20, pairs 101-120
TIERS = {
    "top5": (0, 5),
    "top20": (0, 20),
    "p101_120": (100, 120),
}
WAITS = [0, 1]

# which portfolio to export per-stock weights for (for the ensemble)
WEIGHT_TIER = "top20"
WEIGHT_WAIT = 1


# "reset": rescale prices to 1 at the start of trading (common in replications)
# "continue": keep the index from the start of formation
TRADING_NORMALIZATION = "reset"


# ---------------------------------------------------------------- data

def load_data(path):
    df = pd.read_csv(path, parse_dates=["date"])
    adj = df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    vol = df.pivot(index="date", columns="ticker", values="volume").sort_index()
    vol = vol.reindex(index=adj.index, columns=adj.columns)
    return adj, vol


def build_cohorts(dates):
    months = dates.to_period("M")
    unique_months = months.unique()

    cohorts = []
    k = 0
    while k + FORMATION_MONTHS < len(unique_months):
        form_months = unique_months[k:k + FORMATION_MONTHS]
        trade_months = unique_months[k + FORMATION_MONTHS:k + FORMATION_MONTHS + TRADING_MONTHS]

        form_mask = months.isin(form_months)
        trade_mask = months.isin(trade_months)

        cohorts.append({
            "id": k,
            "form_dates": dates[form_mask],
            "trade_dates": dates[trade_mask],
            "trade_start": str(trade_months[0]),
        })
        k = k + 1
    return cohorts


# ---------------------------------------------------------------- formation

def eligible_stocks(adj, vol, form_dates):
    keep = []
    for t in adj.columns:
        p = adj.loc[form_dates, t]
        v = vol.loc[form_dates, t]
        if p.isna().any():
            continue
        if v.isna().any() or (v <= 0).any():
            continue
        keep.append(t)
    return keep


def form_pairs(adj, tickers, form_dates, n_needed):
    prices = adj.loc[form_dates, tickers].values
    norm = prices / prices[0, :]          # each column starts at 1

    candidates = []
    n = len(tickers)
    for i in range(n - 1):
        diff = norm[:, i + 1:] - norm[:, [i]]
        ssd_row = (diff * diff).sum(axis=0)
        for k in range(len(ssd_row)):
            j = i + 1 + k
            candidates.append((ssd_row[k], i, j))

    candidates.sort()

    pairs = []
    for rank in range(min(n_needed, len(candidates))):
        ssd, i, j = candidates[rank]
        spread = norm[:, i] - norm[:, j]
        pairs.append({
            "rank": rank + 1,
            "stock_a": tickers[i],
            "stock_b": tickers[j],
            "ssd": ssd,
            "sigma": float(np.std(spread, ddof=1)),
        })
    return pairs


# ---------------------------------------------------------------- trading

def simulate_pair(adj, pair, form_dates, trade_dates, wait):
    a = pair["stock_a"]
    b = pair["stock_b"]
    sigma = pair["sigma"]
    trigger = TRIGGER_SIGMAS * sigma

    price_a = adj.loc[trade_dates, a].values
    price_b = adj.loc[trade_dates, b].values

    if TRADING_NORMALIZATION == "reset":
        base_a = adj.loc[form_dates[-1], a]
        base_b = adj.loc[form_dates[-1], b]
    else:
        base_a = adj.loc[form_dates[0], a]
        base_b = adj.loc[form_dates[0], b]

    # daily returns inside the trading window (first day vs last formation day)
    prev_a = adj.loc[form_dates[-1], a]
    prev_b = adj.loc[form_dates[-1], b]

    n_days = len(trade_dates)
    payoff = np.zeros(n_days)        # daily P&L per $1 per leg
    is_open = np.zeros(n_days, dtype=bool)
    hold_a = np.zeros(n_days)        # signed $ held in stock a at today's close
    hold_b = np.zeros(n_days)
    trades = []

    position = 0       # +1 = long a / short b, -1 = short a / long b
    long_value = 0.0
    short_value = 0.0
    pending_open = None     # (day index, direction)
    pending_close = None    # day index
    entry_day = None
    trade_pnl = 0.0
    last_day = n_days - 1

    for t in range(n_days):
        pa = price_a[t]
        pb = price_b[t]

        # a stock stopped trading (delisting): close at the last price and stop
        if np.isnan(pa) or np.isnan(pb):
            if position != 0:
                trades.append(make_trade(pair, trade_dates, entry_day, t - 1,
                                         position, trade_pnl, False))
                position = 0
            break

        ret_a = pa / prev_a - 1
        ret_b = pb / prev_b - 1
        prev_a = pa
        prev_b = pb

        # 1. P&L for a position held overnight into today's close
        if position != 0:
            if position == 1:
                r_long = ret_a
                r_short = ret_b
            else:
                r_long = ret_b
                r_short = ret_a
            day_pnl = long_value * r_long - short_value * r_short
            long_value = long_value * (1 + r_long)
            short_value = short_value * (1 + r_short)
            payoff[t] = day_pnl
            trade_pnl = trade_pnl + day_pnl
            is_open[t] = True

        # 2. orders scheduled for today from earlier signals
        position, long_value, short_value, entry_day, trade_pnl, pending_open, pending_close = \
            execute_orders(t, pair, trade_dates, position, long_value, short_value,
                           entry_day, trade_pnl, pending_open, pending_close, trades)

        # 3. new signals at today's close
        spread = pa / base_a - pb / base_b
        target_day = t + wait

        # an open on the last day would be force-closed at the same close
        # (a zero-length trade), so only allow opens before the last day
        if position == 0 and pending_open is None and target_day < last_day:
            if spread > trigger:
                pending_open = (target_day, -1)     # a is high: short a, long b
            elif spread < -trigger:
                pending_open = (target_day, 1)      # b is high: long a, short b

        if position != 0 and pending_close is None:
            crossed = False
            if position == -1 and spread <= 0:
                crossed = True
            if position == 1 and spread >= 0:
                crossed = True
            if crossed:
                pending_close = target_day

        # 4. with wait = 0, today's signals execute at today's close
        position, long_value, short_value, entry_day, trade_pnl, pending_open, pending_close = \
            execute_orders(t, pair, trade_dates, position, long_value, short_value,
                           entry_day, trade_pnl, pending_open, pending_close, trades)

        # 5. end of the window: close whatever is still open
        if t == last_day and position != 0:
            trades.append(make_trade(pair, trade_dates, entry_day, t,
                                     position, trade_pnl, False))
            position = 0

        # 6. record end-of-day holdings (these earn tomorrow's return)
        if position == 1:
            hold_a[t] = long_value
            hold_b[t] = -short_value
        elif position == -1:
            hold_a[t] = -short_value
            hold_b[t] = long_value

    return payoff, is_open, trades, hold_a, hold_b


def execute_orders(t, pair, trade_dates, position, long_value, short_value,
                   entry_day, trade_pnl, pending_open, pending_close, trades):
    if pending_close is not None and pending_close == t and position != 0:
        trades.append(make_trade(pair, trade_dates, entry_day, t,
                                 position, trade_pnl, True))
        position = 0
        pending_close = None

    if pending_open is not None and pending_open[0] == t and position == 0:
        position = pending_open[1]
        long_value = 1.0
        short_value = 1.0
        entry_day = t
        trade_pnl = 0.0
        pending_open = None

    return position, long_value, short_value, entry_day, trade_pnl, pending_open, pending_close


def make_trade(pair, trade_dates, entry_day, exit_day, position, pnl, converged):
    if position == 1:
        long_leg = pair["stock_a"]
        short_leg = pair["stock_b"]
    else:
        long_leg = pair["stock_b"]
        short_leg = pair["stock_a"]
    return {
        "rank": pair["rank"],
        "long": long_leg,
        "short": short_leg,
        "entry_date": trade_dates[entry_day],
        "exit_date": trade_dates[exit_day],
        "days_open": exit_day - entry_day,
        "pnl": pnl,
        "converged": converged,
    }


# ---------------------------------------------------------------- returns

def cohort_daily_returns(pair_results, lo, hi):
    payoffs = []
    opens = []
    for k in range(lo, min(hi, len(pair_results))):
        payoffs.append(pair_results[k]["payoff"])
        opens.append(pair_results[k]["is_open"])

    n_pairs = hi - lo
    n_days = len(payoffs[0])
    committed = np.zeros(n_days)
    fully_invested = np.zeros(n_days)

    for t in range(n_days):
        total = 0.0
        n_open = 0
        for k in range(len(payoffs)):
            total = total + payoffs[k][t]
            if opens[k][t]:
                n_open = n_open + 1
        committed[t] = total / n_pairs
        if n_open > 0:
            fully_invested[t] = total / n_open

    return committed, fully_invested


def daily_to_monthly(dates, daily):
    months = dates.to_period("M")
    out = {}
    for m in months.unique():
        mask = months == m
        growth = 1.0
        for r in daily[mask]:
            growth = growth * (1 + r)
        out[m] = growth - 1
    return out


def average_cohorts(cohort_monthly):
    # cohort_monthly: list of {month: return}
    all_months = set()
    for d in cohort_monthly:
        for m in d:
            all_months.add(m)

    rows = []
    for m in sorted(all_months):
        values = []
        for d in cohort_monthly:
            if m in d:
                values.append(d[m])
        rows.append((m, sum(values) / len(values), len(values)))

    index = []
    values_out = []
    counts_out = []
    for r in rows:
        index.append(r[0])
        values_out.append(r[1])
        counts_out.append(r[2])

    series = pd.Series(values_out, index=index)
    counts = pd.Series(counts_out, index=index)
    return series, counts


# ---------------------------------------------------------------- weights

def build_daily_weights(cohort_records, lo, hi, all_dates, tickers):
    """
    Per-stock weights of the pairs sleeve at each day's close.

    Within a cohort, each pair has $1 committed, so a stock's weight is
    its signed $ holding divided by the number of pairs in the tier.
    Across cohorts, each active cohort gets an equal share of capital.
    Weights at the close of day t earn the return of day t+1.
    """
    col_of = {}
    for k in range(len(tickers)):
        col_of[tickers[k]] = k
    row_of = {}
    for k in range(len(all_dates)):
        row_of[all_dates[k]] = k

    weights = np.zeros((len(all_dates), len(tickers)))
    n_active = np.zeros(len(all_dates))
    n_pairs = hi - lo

    for rec in cohort_records:
        if len(rec["results"]) < hi:
            continue
        trade_dates = rec["cohort"]["trade_dates"]

        for d in trade_dates:
            n_active[row_of[d]] = n_active[row_of[d]] + 1

        for k in range(lo, hi):
            pair = rec["pairs"][k]
            result = rec["results"][k]
            col_a = col_of[pair["stock_a"]]
            col_b = col_of[pair["stock_b"]]
            for t in range(len(trade_dates)):
                row = row_of[trade_dates[t]]
                weights[row, col_a] = weights[row, col_a] + result["hold_a"][t] / n_pairs
                weights[row, col_b] = weights[row, col_b] + result["hold_b"][t] / n_pairs

    # Holdings at the close of day t earn day t+1's return, so split capital
    # across the cohorts trading on day t+1 (a cohort that ends on day t holds
    # nothing at its last close, and a cohort starting on t+1 holds nothing yet).
    for row in range(len(all_dates)):
        if row + 1 < len(all_dates):
            divisor = n_active[row + 1]
        else:
            divisor = n_active[row]
        if divisor > 0:
            weights[row, :] = weights[row, :] / divisor

    df = pd.DataFrame(weights, index=all_dates, columns=tickers)
    df.index.name = "date"
    df["n_cohorts"] = n_active
    return df


def monthly_weights(daily_weights):
    stock_cols = []
    for col in daily_weights.columns:
        if col != "n_cohorts":
            stock_cols.append(col)

    months = daily_weights.index.to_period("M")
    end_rows = []
    avg_rows = []
    index = []
    for m in months.unique():
        block = daily_weights.loc[months == m, stock_cols]
        if (block.abs().sum(axis=1) == 0).all():
            continue        # no trading cohorts this month (formation-only period)
        index.append(str(m))
        end_rows.append(block.iloc[-1].values)
        avg_rows.append(block.mean(axis=0).values)

    month_end = pd.DataFrame(end_rows, index=index, columns=stock_cols)
    month_avg = pd.DataFrame(avg_rows, index=index, columns=stock_cols)
    month_end.index.name = "month"
    month_avg.index.name = "month"
    return month_end, month_avg


# ---------------------------------------------------------------- statistics

def newey_west_tstat(x, lags):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = x.mean()
    e = x - mean

    s = (e * e).sum() / n
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        cov = (e[lag:] * e[:-lag]).sum() / n
        s = s + 2 * weight * cov

    se = np.sqrt(s / n)
    return mean / se


def summarize(series):
    x = series.values
    return {
        "mean": x.mean(),
        "nw_tstat": newey_west_tstat(x, NW_LAGS),
        "median": np.median(x),
        "std": x.std(ddof=1),
        "skew": pd.Series(x).skew(),
        "kurtosis": pd.Series(x).kurt(),
        "min": x.min(),
        "max": x.max(),
        "pct_negative": (x < 0).mean(),
        "sharpe_annual": x.mean() / x.std(ddof=1) * np.sqrt(12),
        "n_months": len(x),
    }


def trade_stats(cohort_records, lo, hi):
    triggers = []
    pairs_opened = []
    round_trips = []
    months_open = []
    n_converged = 0
    n_trades = 0

    for rec in cohort_records:
        opened_here = 0
        for k in range(lo, min(hi, len(rec["pairs"]))):
            pair = rec["pairs"][k]
            result = rec["results"][k]
            triggers.append(TRIGGER_SIGMAS * pair["sigma"])
            n = len(result["trades"])
            if n > 0:
                opened_here = opened_here + 1
                round_trips.append(n)
                days = int(result["is_open"].sum())
                months_open.append(days / 21.0)
            for tr in result["trades"]:
                n_trades = n_trades + 1
                if tr["converged"]:
                    n_converged = n_converged + 1
        pairs_opened.append(opened_here)

    return {
        "avg_trigger_gap": np.mean(triggers),
        "avg_pairs_opened": np.mean(pairs_opened),
        "avg_round_trips_per_pair": np.mean(round_trips) if round_trips else 0.0,
        "avg_months_open_per_pair": np.mean(months_open) if months_open else 0.0,
        "pct_trades_converged": n_converged / n_trades if n_trades else 0.0,
        "n_trades": n_trades,
    }


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join("data", "ohlcv_raw.csv"))
    parser.add_argument("--out", default="pairs_results")
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    adj, vol = load_data(args.data)
    cohorts = build_cohorts(adj.index)
    print("tickers:", adj.shape[1], " days:", adj.shape[0], " cohorts:", len(cohorts))

    n_needed = 0
    for name in TIERS:
        if TIERS[name][1] > n_needed:
            n_needed = TIERS[name][1]

    # records[wait] = list of cohort records
    records = {}
    for wait in WAITS:
        records[wait] = []

    pair_rows = []
    trade_rows = []

    for c in cohorts:
        tickers = eligible_stocks(adj, vol, c["form_dates"])
        pairs = form_pairs(adj, tickers, c["form_dates"], n_needed)

        for p in pairs:
            row = dict(p)
            row["cohort"] = c["id"]
            row["trade_start"] = c["trade_start"]
            row["n_eligible"] = len(tickers)
            pair_rows.append(row)

        for wait in WAITS:
            results = []
            for p in pairs:
                payoff, is_open, trades, hold_a, hold_b = simulate_pair(
                    adj, p, c["form_dates"], c["trade_dates"], wait)
                results.append({"payoff": payoff, "is_open": is_open, "trades": trades,
                                "hold_a": hold_a, "hold_b": hold_b})
                for tr in trades:
                    tr["cohort"] = c["id"]
                    tr["wait"] = wait
                    trade_rows.append(tr)
            records[wait].append({"cohort": c, "pairs": pairs, "results": results})

        print("  cohort", c["id"], "trading from", c["trade_start"],
              "-", len(tickers), "eligible stocks")

    # monthly returns, Table 1 style summary, Table 2 style trade stats
    monthly_cols = {}
    summary_rows = []
    stats_rows = []

    for wait in WAITS:
        for name in TIERS:
            lo, hi = TIERS[name]

            committed_by_cohort = []
            invested_by_cohort = []
            for rec in records[wait]:
                if len(rec["results"]) < hi:
                    continue
                committed, invested = cohort_daily_returns(rec["results"], lo, hi)
                dates = rec["cohort"]["trade_dates"]
                committed_by_cohort.append(daily_to_monthly(dates, committed))
                invested_by_cohort.append(daily_to_monthly(dates, invested))

            for label, by_cohort in [("committed", committed_by_cohort),
                                     ("fully_invested", invested_by_cohort)]:
                series, counts = average_cohorts(by_cohort)
                col = name + "_wait" + str(wait) + "_" + label
                monthly_cols[col] = series

                row = summarize(series)
                row["portfolio"] = name
                row["wait"] = wait
                row["capital"] = label
                summary_rows.append(row)

            row = trade_stats(records[wait], lo, hi)
            row["portfolio"] = name
            row["wait"] = wait
            stats_rows.append(row)

    monthly = pd.DataFrame(monthly_cols)
    monthly.index = monthly.index.astype(str)
    monthly.index.name = "month"
    monthly.to_csv(os.path.join(args.out, "monthly_returns.csv"))

    summary = pd.DataFrame(summary_rows)
    first_cols = ["portfolio", "wait", "capital"]
    other_cols = []
    for col in summary.columns:
        if col not in first_cols:
            other_cols.append(col)
    summary = summary[first_cols + other_cols]
    summary.to_csv(os.path.join(args.out, "table1_returns.csv"), index=False)

    stats = pd.DataFrame(stats_rows)
    stat_cols = ["portfolio", "wait"]
    for col in stats.columns:
        if col not in stat_cols:
            stat_cols.append(col)
    stats = stats[stat_cols]
    stats.to_csv(os.path.join(args.out, "table2_trade_stats.csv"), index=False)

    # per-stock weights for the ensemble
    lo, hi = TIERS[WEIGHT_TIER]
    tickers = list(adj.columns)
    daily_w = build_daily_weights(records[WEIGHT_WAIT], lo, hi, adj.index, tickers)
    month_end, month_avg = monthly_weights(daily_w)
    daily_w.to_csv(os.path.join(args.out, "pairs_weights_daily.csv"))
    month_end.to_csv(os.path.join(args.out, "pairs_weights_month_end.csv"))
    month_avg.to_csv(os.path.join(args.out, "pairs_weights_month_avg.csv"))

    # sanity check: yesterday's weights x today's returns should equal the
    # average committed daily return across active cohorts
    daily_ret = adj.pct_change()
    w_only = daily_w[tickers]
    implied = (w_only.shift(1) * daily_ret).sum(axis=1)
    print("weights check - mean daily return implied by weights:",
          round(float(implied.mean()), 6))

    pd.DataFrame(pair_rows).to_csv(os.path.join(args.out, "pairs_selected.csv"), index=False)
    pd.DataFrame(trade_rows).to_csv(os.path.join(args.out, "trades.csv"), index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    print()
    print("Table 1 style (monthly excess returns)")
    print(summary[["portfolio", "wait", "capital", "mean", "nw_tstat", "std",
                   "pct_negative", "sharpe_annual"]].to_string(index=False))
    print()
    print("Table 2 style (trade statistics)")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()