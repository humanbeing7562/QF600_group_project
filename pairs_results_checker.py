"""
Plain-English summary of the pairs trading results + equity curve.

Reads results/monthly_returns.csv (made by pairs_replication.py) and prints,
for each version: what $1 grew to, total and yearly return, and how many
months made money. Saves the growth-of-$1 chart to results/equity_curve.png.

Usage:
  python did_we_make_money.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = "pairs_results"

# the realistic version (trade one day after the signal), committed capital
MAIN_COLUMNS = [
    ("top5_wait1_committed", "Top 5 pairs", "#eb6834"),
    ("top20_wait1_committed", "Top 20 pairs", "#2a78d6"),
    ("p101_120_wait1_committed", "Pairs 101-120", "#1baf7a"),
]

INK = "#2b2b29"
MUTED = "#8a897f"
GRID = "#e6e5df"
SURFACE = "#fcfcfb"


def growth_of_one_dollar(monthly):
    values = []
    value = 1.0
    for r in monthly:
        value = value * (1 + r)
        values.append(value)
    return values


def describe(name, monthly):
    growth = growth_of_one_dollar(monthly)
    final = growth[-1]
    n_months = len(monthly)
    years = n_months / 12.0
    yearly = final ** (1 / years) - 1

    n_up = 0
    for r in monthly:
        if r > 0:
            n_up = n_up + 1

    # worst peak-to-trough fall
    peak = 1.0
    worst_drop = 0.0
    for v in growth:
        if v > peak:
            peak = v
        drop = v / peak - 1
        if drop < worst_drop:
            worst_drop = drop

    if final > 1:
        verdict = "MADE money"
    else:
        verdict = "LOST money"

    print(name)
    print("  " + verdict + ": $1.00 became ${:.2f}".format(final))
    print("  total return:  {:+.1f}%  over {:.1f} years".format((final - 1) * 100, years))
    print("  per year:      {:+.2f}%".format(yearly * 100))
    print("  winning months: {} of {} ({:.0f}%)".format(n_up, n_months, n_up / n_months * 100))
    print("  worst fall from a peak: {:.1f}%".format(worst_drop * 100))
    print()


def main():
    path = os.path.join(RESULTS_DIR, "monthly_returns.csv")
    data = pd.read_csv(path, index_col=0)
    data.index = pd.PeriodIndex(data.index, freq="M").to_timestamp()

    print("Strategy: trade one day after the signal, return on committed capital")
    print("(no transaction costs included)")
    print()
    for col, label, color in MAIN_COLUMNS:
        describe(label, data[col].values)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.axhline(1.0, color=MUTED, linewidth=1, linestyle="--", label="Break-even ($1)")

    for col, label, color in MAIN_COLUMNS:
        growth = growth_of_one_dollar(data[col].values)
        ax.plot(data.index, growth, color=color, linewidth=2, label=label)
        # direct label at the end of each line
        ax.annotate("{}  ${:.2f}".format(label, growth[-1]),
                    xy=(data.index[-1], growth[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)

    ax.set_title("Growth of $1 in the pairs strategy", color=INK, loc="left", fontsize=13)
    ax.set_ylabel("Value of $1 invested", color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "equity_curve.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print("chart saved to", out)


if __name__ == "__main__":
    main()