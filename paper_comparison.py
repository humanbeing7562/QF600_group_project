"""
Compare our pairs trading results with Gatev, Goetzmann & Rouwenhorst (2006).

Reads the tables written by pairs_trading.py for two universes:
  pairs_results/   top 100 S&P 500 stocks by market cap
  results_sp500/   all S&P 500 stocks

and puts them next to the paper's numbers (Table 1 Panel B, Table 2, Table 4),
all for the realistic version: trade one day after the signal.

Outputs (in paper_comparison/):
  paper_vs_ours.csv           one row per measure, one column per source, per portfolio
  paper_vs_ours_returns.png   monthly return (fully invested) for each portfolio

Usage:
  python paper_comparison.py
  python paper_comparison.py --top100 pairs_results --sp500 results_sp500
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = "paper_comparison"
PORTFOLIOS = ["top5", "top20", "p101_120"]
PORTFOLIO_LABELS = {"top5": "Top 5", "top20": "Top 20", "p101_120": "Pairs 101-120"}

# Paper, 1962-2002, wait one day (Table 1 Panel B, Table 2, Table 4).
# Returns are monthly, in decimals. Sharpe is the paper's monthly Sharpe x sqrt(12).
PAPER = {
    "top5": {
        "mean_fully_invested": 0.0075, "nw_tstat_fully_invested": 6.26,
        "mean_committed": 0.0046, "std_fully_invested": 0.0210,
        "pct_negative_fully_invested": 0.35, "sharpe_annual_fully_invested": 0.35 * np.sqrt(12),
        "mean_fully_invested_wait0": 0.0131,
        "avg_trigger_gap": 0.0476, "avg_pairs_opened": 4.81,
        "avg_round_trips_per_pair": 2.02, "avg_months_open_per_pair": 3.75,
    },
    "top20": {
        "mean_fully_invested": 0.0090, "nw_tstat_fully_invested": 9.29,
        "mean_committed": 0.0052, "std_fully_invested": 0.0153,
        "pct_negative_fully_invested": 0.23, "sharpe_annual_fully_invested": 0.59 * np.sqrt(12),
        "mean_fully_invested_wait0": 0.0144,
        "avg_trigger_gap": 0.0528, "avg_pairs_opened": 19.30,
        "avg_round_trips_per_pair": 1.96, "avg_months_open_per_pair": 3.76,
    },
    "p101_120": {
        "mean_fully_invested": 0.0080, "nw_tstat_fully_invested": 9.40,
        "mean_committed": 0.0050, "std_fully_invested": 0.0144,
        "pct_negative_fully_invested": 0.28, "sharpe_annual_fully_invested": 0.55 * np.sqrt(12),
        "mean_fully_invested_wait0": 0.0108,
        "avg_trigger_gap": 0.0756, "avg_pairs_opened": 19.41,
        "avg_round_trips_per_pair": 1.78, "avg_months_open_per_pair": 3.98,
    },
}

# measure key, label shown in the table, how to format it
MEASURES = [
    ("mean_fully_invested", "Monthly return, fully invested", "pct"),
    ("nw_tstat_fully_invested", "Newey-West t-stat, fully invested", "num"),
    ("mean_committed", "Monthly return, committed capital", "pct"),
    ("std_fully_invested", "Monthly volatility, fully invested", "pct"),
    ("sharpe_annual_fully_invested", "Annualized Sharpe, fully invested", "num"),
    ("pct_negative_fully_invested", "Losing months, fully invested", "pct0"),
    ("wait_cost", "Return lost by waiting one day (per month)", "pct"),
    ("avg_trigger_gap", "2-sigma trigger gap", "pct"),
    ("avg_pairs_opened", "Pairs opened per 6-month window", "num"),
    ("avg_round_trips_per_pair", "Round trips per pair", "num"),
    ("avg_months_open_per_pair", "Months open per pair", "num"),
]

SOURCE_COLORS = {"Paper (1962-2002)": "#2a78d6", "Ours: top 100": "#eb6834", "Ours: S&P 500": "#1baf7a"}
INK = "#2b2b29"
MUTED = "#8a897f"
GRID = "#e6e5df"
SURFACE = "#fcfcfb"


def read_ours(folder):
    """Pull the same measures out of one run's table1_returns.csv and table2_trade_stats.csv."""
    t1 = pd.read_csv(os.path.join(folder, "table1_returns.csv"))
    t2 = pd.read_csv(os.path.join(folder, "table2_trade_stats.csv"))

    out = {}
    for p in PORTFOLIOS:
        fi1 = t1[(t1["portfolio"] == p) & (t1["wait"] == 1) & (t1["capital"] == "fully_invested")].iloc[0]
        co1 = t1[(t1["portfolio"] == p) & (t1["wait"] == 1) & (t1["capital"] == "committed")].iloc[0]
        fi0 = t1[(t1["portfolio"] == p) & (t1["wait"] == 0) & (t1["capital"] == "fully_invested")].iloc[0]
        st1 = t2[(t2["portfolio"] == p) & (t2["wait"] == 1)].iloc[0]

        out[p] = {
            "mean_fully_invested": fi1["mean"],
            "nw_tstat_fully_invested": fi1["nw_tstat"],
            "mean_committed": co1["mean"],
            "std_fully_invested": fi1["std"],
            "pct_negative_fully_invested": fi1["pct_negative"],
            "sharpe_annual_fully_invested": fi1["sharpe_annual"],
            "mean_fully_invested_wait0": fi0["mean"],
            "avg_trigger_gap": st1["avg_trigger_gap"],
            "avg_pairs_opened": st1["avg_pairs_opened"],
            "avg_round_trips_per_pair": st1["avg_round_trips_per_pair"],
            "avg_months_open_per_pair": st1["avg_months_open_per_pair"],
        }
    return out


def add_wait_cost(values):
    for p in values:
        values[p]["wait_cost"] = values[p]["mean_fully_invested_wait0"] - values[p]["mean_fully_invested"]


def fmt(x, kind):
    if pd.isna(x):
        return ""
    if kind == "pct":
        return "{:.2f}%".format(x * 100)
    if kind == "pct0":
        return "{:.0f}%".format(x * 100)
    return "{:.2f}".format(x)


def plot_returns(sources, filename):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    names = list(sources.keys())
    n = len(names)
    width = 0.8 / n
    x = np.arange(len(PORTFOLIOS))

    for k in range(n):
        name = names[k]
        heights = []
        tstats = []
        for p in PORTFOLIOS:
            heights.append(sources[name][p]["mean_fully_invested"] * 100)
            tstats.append(sources[name][p]["nw_tstat_fully_invested"])
        positions = x - 0.4 + width * (k + 0.5)
        ax.bar(positions, heights, width=width * 0.92, color=SOURCE_COLORS[name], label=name)
        for i in range(len(positions)):
            h = heights[i]
            if h >= 0:
                va = "bottom"
                offset = 0.02
            else:
                va = "top"
                offset = -0.02
            ax.text(positions[i], h + offset, "t = {:.1f}".format(tstats[i]),
                    ha="center", va=va, fontsize=8, color=INK)

    ax.axhline(0, color=MUTED, linewidth=1)
    labels = []
    for p in PORTFOLIOS:
        labels.append(PORTFOLIO_LABELS[p])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=INK)
    ax.set_ylabel("Monthly excess return (%)", color=INK)
    ax.set_title("Pairs trading returns: paper vs our replication (one-day wait, fully invested)",
                 color=INK, loc="left", fontsize=12)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=len(names), labelcolor=INK)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top100", default="pairs_results")
    parser.add_argument("--sp500", default="results_sp500")
    args = parser.parse_args()

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    sources = {"Paper (1962-2002)": PAPER}
    sources["Ours: top 100"] = read_ours(args.top100)
    if os.path.exists(os.path.join(args.sp500, "table1_returns.csv")):
        sources["Ours: S&P 500"] = read_ours(args.sp500)
    else:
        print("note: no S&P 500 results found in", args.sp500, "- comparing with the top 100 only")
    for name in sources:
        add_wait_cost(sources[name])

    rows = []
    for p in PORTFOLIOS:
        for key, label, kind in MEASURES:
            row = {"portfolio": PORTFOLIO_LABELS[p], "measure": label}
            for name in sources:
                row[name] = fmt(sources[name][p][key], kind)
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT_DIR, "paper_vs_ours.csv"), index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    for p in PORTFOLIOS:
        print()
        print(PORTFOLIO_LABELS[p] + " (trade one day after the signal)")
        block = table[table["portfolio"] == PORTFOLIO_LABELS[p]].drop(columns=["portfolio"])
        print(block.to_string(index=False))
    print()
    print("saved", os.path.join(OUT_DIR, "paper_vs_ours.csv"))

    plot_returns(sources, "paper_vs_ours_returns.png")


if __name__ == "__main__":
    main()