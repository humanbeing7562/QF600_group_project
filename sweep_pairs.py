"""
Pairs trading weights for several trigger sizes and weight tiers.

Uses the functions in pairs_trading.py as they are. The pairs are formed once
per cohort (formation does not depend on the trigger), then every pair is
traded again for each trigger value, with the one-day wait.

Nothing in pairs_results/ is overwritten.

Outputs (in sweep_results/pairs/):
  pairs_weights_daily_sig{S}_{tier}.csv   daily weights, same layout as
                                          pairs_weights_daily.csv (incl. n_cohorts)
  summary.csv                             Table 1 / Table 2 style numbers for
                                          every trigger and tier (wait = 1)

Usage:
  python sweep_pairs.py
"""

import os

import pandas as pd

import pairs_trading as pt


TRIGGERS = [1.0, 1.5, 2.0, 2.5, 3.0]
WEIGHT_TIERS = ["top5", "top20", "p101_120"]
WAIT = 1

DATA_FILE = os.path.join("data", "ohlcv_raw.csv")
OUT_DIR = os.path.join("sweep_results", "pairs")


def form_all_cohorts(adj, vol, cohorts, n_needed):
    """Formation step from pairs_trading.main(), done once for all triggers."""
    formed = []
    for c in cohorts:
        tickers = pt.eligible_stocks(adj, vol, c["form_dates"])
        pairs = pt.form_pairs(adj, tickers, c["form_dates"], n_needed)
        formed.append((c, pairs))
        print("  formed cohort", c["id"], "trading from", c["trade_start"])
    return formed


def trade_all_cohorts(adj, formed, wait):
    """Trading step from pairs_trading.main() for one trigger (set beforehand)."""
    records = []
    for c, pairs in formed:
        results = []
        for p in pairs:
            payoff, is_open, trades, hold_a, hold_b = pt.simulate_pair(
                adj, p, c["form_dates"], c["trade_dates"], wait)
            results.append({"payoff": payoff, "is_open": is_open, "trades": trades,
                            "hold_a": hold_a, "hold_b": hold_b})
        records.append({"cohort": c, "pairs": pairs, "results": results})
    return records


def tier_summary(records, lo, hi):
    """Same numbers as table1_returns.csv and table2_trade_stats.csv for one tier."""
    committed_by_cohort = []
    invested_by_cohort = []
    for rec in records:
        if len(rec["results"]) < hi:
            continue
        committed, invested = pt.cohort_daily_returns(rec["results"], lo, hi)
        dates = rec["cohort"]["trade_dates"]
        committed_by_cohort.append(pt.daily_to_monthly(dates, committed))
        invested_by_cohort.append(pt.daily_to_monthly(dates, invested))

    committed_series, counts = pt.average_cohorts(committed_by_cohort)
    invested_series, counts = pt.average_cohorts(invested_by_cohort)
    committed_stats = pt.summarize(committed_series)
    invested_stats = pt.summarize(invested_series)
    trade = pt.trade_stats(records, lo, hi)

    row = {
        "mean_committed": committed_stats["mean"],
        "nw_tstat_committed": committed_stats["nw_tstat"],
        "sharpe_annual_committed": committed_stats["sharpe_annual"],
        "mean_fully_invested": invested_stats["mean"],
        "nw_tstat_fully_invested": invested_stats["nw_tstat"],
        "std_fully_invested": invested_stats["std"],
        "sharpe_annual_fully_invested": invested_stats["sharpe_annual"],
        "pct_negative_fully_invested": invested_stats["pct_negative"],
    }
    for key in trade:
        row[key] = trade[key]
    return row


def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    adj, vol = pt.load_data(DATA_FILE)
    cohorts = pt.build_cohorts(adj.index)
    tickers = list(adj.columns)

    n_needed = 0
    for name in WEIGHT_TIERS:
        if pt.TIERS[name][1] > n_needed:
            n_needed = pt.TIERS[name][1]

    print("forming pairs (once for all triggers)")
    formed = form_all_cohorts(adj, vol, cohorts, n_needed)

    summary_rows = []
    original_trigger = pt.TRIGGER_SIGMAS
    try:
        for trigger in TRIGGERS:
            # simulate_pair and trade_stats read pt.TRIGGER_SIGMAS when they run
            pt.TRIGGER_SIGMAS = trigger
            print("trigger {:g} sigma".format(trigger))
            records = trade_all_cohorts(adj, formed, WAIT)

            for tier in WEIGHT_TIERS:
                lo, hi = pt.TIERS[tier]
                daily_w = pt.build_daily_weights(records, lo, hi, adj.index, tickers)
                path = os.path.join(OUT_DIR, "pairs_weights_daily_sig{:g}_{}.csv".format(trigger, tier))
                daily_w.to_csv(path)

                row = {"trigger_sigmas": trigger, "tier": tier, "wait": WAIT}
                stats = tier_summary(records, lo, hi)
                for key in stats:
                    row[key] = stats[key]
                summary_rows.append(row)
                print("   saved", path)
    finally:
        pt.TRIGGER_SIGMAS = original_trigger

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    print()
    print(summary[["trigger_sigmas", "tier", "mean_fully_invested", "nw_tstat_fully_invested",
                   "sharpe_annual_fully_invested", "avg_pairs_opened",
                   "avg_round_trips_per_pair", "n_trades"]].to_string(index=False))
    print()
    print("saved", os.path.join(OUT_DIR, "summary.csv"))


if __name__ == "__main__":
    main()