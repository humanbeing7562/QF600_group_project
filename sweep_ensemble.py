"""
Ensemble sweep: every combination of the sleeve variants in sweep_results/,
run through the Average, FTL daily and FTL monthly methods of ensemble.py.

Sleeve variants (made by sweep_mvo.py, sweep_ridge.py and sweep_pairs.py):
  MVO    lambda in LAMBDAS            sweep_results/mvo/weights_naive_mvo_lam{L}.csv     (monthly)
  Ridge  theta grid in RIDGE_GRIDS    sweep_results/ridge/weights_ridge_mvo_{grid}.csv   (monthly)
  Pairs  trigger x tier               sweep_results/pairs/pairs_weights_daily_sig{S}_{tier}.csv (daily)

Monthly weights are expanded to daily with the same rule as
expand_monthly_weights.py, and every method uses the functions in ensemble.py
unchanged. Nothing in ensemble_results/ is overwritten.

Outputs (in sweep_results/ensemble/):
  sleeve_metrics.csv     metrics of each sleeve variant on its own
  metrics.csv            one row per combination and method
  daily_returns.csv      daily returns, one column per combination and method,
                         named like lam4_full_sig2_top20_Average

Usage:
  python sweep_ensemble.py
"""

import os

import pandas as pd

from ensemble import (START, END, load_stock_returns, load_weights, monthly_to_daily,
                      sleeve_returns, average_method, ftl_daily, ftl_monthly,
                      compute_metrics)


LAMBDAS = [1, 4, 10, 50, 200]
RIDGE_GRIDS = ["full", "from1e-4", "from1e-2"]
TRIGGERS = [1.0, 1.5, 2.0, 2.5, 3.0]
PAIR_TIERS = ["top5", "top20", "p101_120"]

# the setting used in the main results, for the check at the end
BASELINE = {"lambda": 4, "ridge_grid": "full", "trigger": 2.0, "tier": "top20"}

SWEEP_DIR = "sweep_results"
OUT_DIR = os.path.join(SWEEP_DIR, "ensemble")
ENSEMBLE_RETURNS = os.path.join("ensemble_results", "daily_returns.csv")
NAMES = ["MVO", "Ridge", "Pairs"]
METHODS = ["Average", "FTL daily", "FTL monthly"]


# ---------------------------------------------------------------- loading

def load_monthly_as_daily(path, tickers, dates):
    """Monthly weights file -> daily weights, same rule as expand_monthly_weights.py."""
    monthly = pd.read_csv(path, index_col=0)
    monthly.index = monthly.index.astype(str)
    monthly = monthly.reindex(columns=tickers).fillna(0.0)
    return monthly_to_daily(monthly, dates)


def load_all_variants(tickers, dates):
    """Returns {sleeve: {variant label: daily weights}} and the settings of each label."""
    variants = {"MVO": {}, "Ridge": {}, "Pairs": {}}
    settings = {}

    for lam in LAMBDAS:
        label = "lam{:g}".format(lam)
        path = os.path.join(SWEEP_DIR, "mvo", "weights_naive_mvo_lam{:g}.csv".format(lam))
        variants["MVO"][label] = load_monthly_as_daily(path, tickers, dates)
        settings[label] = {"lambda": lam}
        print("  loaded", path)

    for grid in RIDGE_GRIDS:
        label = grid
        path = os.path.join(SWEEP_DIR, "ridge", "weights_ridge_mvo_{}.csv".format(grid))
        variants["Ridge"][label] = load_monthly_as_daily(path, tickers, dates)
        settings[label] = {"ridge_grid": grid}
        print("  loaded", path)

    for trigger in TRIGGERS:
        for tier in PAIR_TIERS:
            label = "sig{:g}_{}".format(trigger, tier)
            path = os.path.join(SWEEP_DIR, "pairs",
                                "pairs_weights_daily_sig{:g}_{}.csv".format(trigger, tier))
            variants["Pairs"][label] = load_weights(path, tickers, dates)
            settings[label] = {"trigger": trigger, "tier": tier}
            print("  loaded", path)

    return variants, settings


# ---------------------------------------------------------------- main

def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    returns = load_stock_returns()
    tickers = list(returns.columns)
    dates = returns.index

    print("loading sleeve variants")
    variants, settings = load_all_variants(tickers, dates)

    # each variant's own daily returns, computed once and reused in every combination
    variant_returns = {}
    sleeve_rows = []
    for sleeve in NAMES:
        variant_returns[sleeve] = {}
        for label in variants[sleeve]:
            r = sleeve_returns(variants[sleeve][label], returns)
            variant_returns[sleeve][label] = r
            m = compute_metrics(sleeve + " " + label, r.loc[START:END].fillna(0.0))
            m["sleeve"] = sleeve
            m["variant"] = label
            sleeve_rows.append(m)
    sleeve_metrics = pd.DataFrame(sleeve_rows)
    sleeve_metrics.to_csv(os.path.join(OUT_DIR, "sleeve_metrics.csv"), index=False)
    print("saved", os.path.join(OUT_DIR, "sleeve_metrics.csv"))

    combos = []
    for mvo_label in variants["MVO"]:
        for ridge_label in variants["Ridge"]:
            for pairs_label in variants["Pairs"]:
                combos.append((mvo_label, ridge_label, pairs_label))
    print("running", len(combos), "combinations x", len(METHODS), "methods")

    all_returns = {}
    metric_rows = []
    for k in range(len(combos)):
        mvo_label, ridge_label, pairs_label = combos[k]
        combo_name = "{}_{}_{}".format(mvo_label, ridge_label, pairs_label)

        sleeve_weights = {
            "MVO": variants["MVO"][mvo_label],
            "Ridge": variants["Ridge"][ridge_label],
            "Pairs": variants["Pairs"][pairs_label],
        }
        sleeve_rets = pd.DataFrame(index=dates)
        sleeve_rets["MVO"] = variant_returns["MVO"][mvo_label]
        sleeve_rets["Ridge"] = variant_returns["Ridge"][ridge_label]
        sleeve_rets["Pairs"] = variant_returns["Pairs"][pairs_label]

        # same steps as ensemble.main()
        w_avg = average_method(sleeve_weights, NAMES)
        w_ftl_d, lead_d = ftl_daily(sleeve_weights, sleeve_rets, NAMES, dates)
        w_ftl_m, lead_m = ftl_monthly(sleeve_weights, sleeve_rets, NAMES, dates)

        method_returns = {
            "Average": sleeve_returns(w_avg, returns),
            "FTL daily": sleeve_returns(w_ftl_d, returns),
            "FTL monthly": sleeve_returns(w_ftl_m, returns),
        }

        for method in METHODS:
            daily = method_returns[method].loc[START:END].fillna(0.0)
            column = combo_name + "_" + method.replace(" ", "_")
            all_returns[column] = daily

            row = {}
            for part in [settings[mvo_label], settings[ridge_label], settings[pairs_label]]:
                for key in part:
                    row[key] = part[key]
            row["method"] = method
            m = compute_metrics(column, daily)
            for key in m:
                row[key] = m[key]
            metric_rows.append(row)

        if (k + 1) % 15 == 0 or k + 1 == len(combos):
            print("  done", k + 1, "of", len(combos))

    daily_out = pd.DataFrame(all_returns)
    daily_out.to_csv(os.path.join(OUT_DIR, "daily_returns.csv"))
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False)
    print("saved", os.path.join(OUT_DIR, "daily_returns.csv"))
    print("saved", os.path.join(OUT_DIR, "metrics.csv"))

    # best Sharpe per method, as a quick look
    pd.set_option("display.width", 200)
    print()
    print("highest Sharpe per method:")
    for method in METHODS:
        block = metrics[metrics["method"] == method].sort_values("sharpe", ascending=False)
        print(block[["method", "lambda", "ridge_grid", "trigger", "tier", "sharpe",
                     "annual_return", "max_drawdown", "wiped_out_on"]].head(3).to_string(index=False))

    # check: the baseline combination should match ensemble.py
    base_name = "lam{:g}_{}_sig{:g}_{}".format(BASELINE["lambda"], BASELINE["ridge_grid"],
                                              BASELINE["trigger"], BASELINE["tier"])
    if os.path.exists(ENSEMBLE_RETURNS):
        ens = pd.read_csv(ENSEMBLE_RETURNS, index_col=0, parse_dates=True)
        print()
        for method in METHODS:
            column = base_name + "_" + method.replace(" ", "_")
            if method in ens.columns and column in daily_out.columns:
                gap = (daily_out[column].reindex(ens.index).fillna(0.0) - ens[method]).abs().max()
                print("check - {} vs ensemble.py {}, max daily gap: {:.2e}".format(column, method, gap))


if __name__ == "__main__":
    main()