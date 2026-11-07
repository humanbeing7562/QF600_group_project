"""
Chart for the results section: growth of $1 for each sleeve run on its own
(naive MVO, ridge MVO, pairs), with 1/N and the S&P 500 for reference.

Styled to match the group deck (colors, background, Calibri-style font) and
saved at the same size as the deck's other chart images.

Linear scale. The y-axis is sized to the biggest sleeve other than naive MVO
(ridge's 2018 peak), with that peak labelled. If MVO's early spike goes higher,
it runs off the top and its peak value is written there. An x on the zero line
marks when a sleeve first fell below $0.01.

Reads:  ensemble_results/daily_returns.csv   (run ensemble.py first)
Writes: deck_charts/sleeves_growth.png

Usage:
  python make_sleeve_chart.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


IN_FILE = os.path.join("ensemble_results", "daily_returns.csv")
OUT_DIR = "deck_charts"
DEAD = 0.01            # "effectively dead" once below 1 cent
CAP_EXCLUDE = "MVO"    # this sleeve's spike may run off the top of the chart
ANNOTATE_PEAK = ["Ridge"]

# colors taken from the deck's existing charts
SERIES = [
    ("MVO", "Naive MVO", "#e8a935", 2.0),
    ("Ridge", "Ridge MVO", "#d94545", 2.0),
    ("Pairs", "Pairs", "#1baf7a", 2.0),
    ("1/N", "1/N", "#2a78d6", 1.6),
    ("S&P 500", "S&P 500", "#8a9bb0", 1.6),
]
BACKGROUND = "#f5f7f8"
AXIS_TEXT = "#6a7a87"
INK = "#2b3742"
GRID = "#e1e6ea"

def pick_font():
    """Calibri if installed (Windows), else its look-alike Carlito, else the default."""
    from matplotlib import font_manager
    installed = set()
    for f in font_manager.fontManager.ttflist:
        installed.add(f.name)
    for name in ["Calibri", "Carlito"]:
        if name in installed:
            return name
    return "DejaVu Sans"


plt.rcParams["font.family"] = pick_font()


def growth_of_one_dollar(daily):
    values = []
    value = 1.0
    for r in daily:
        if value > 0:
            value = value * (1 + r)
            if value < 0:
                value = 0.0
        values.append(value)
    return pd.Series(values, index=daily.index)


def place_end_labels(ax, labels):
    """Write each final value at the right end of its line, nudged apart so they don't overlap."""
    fig = ax.get_figure()
    fig.canvas.draw()
    to_px = ax.transData
    placed = []
    for x, y, text, color in labels:
        px, py = to_px.transform((matplotlib.dates.date2num(x), y))
        placed.append([py, py, x, y, text, color])
    placed.sort()
    min_gap = 22
    for i in range(1, len(placed)):
        if placed[i][0] - placed[i - 1][0] < min_gap:
            placed[i][0] = placed[i - 1][0] + min_gap
    for py, true_py, x, y, text, color in placed:
        ax.annotate(text, xy=(x, y), xytext=(6, py - true_py), textcoords="offset pixels",
                    va="center", fontsize=10, color=color, fontweight="bold")


def dollar_label(x, pos):
    return "${:,.0f}".format(x)


def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    daily = pd.read_csv(IN_FILE, index_col=0, parse_dates=True)

    # same pixel size as the deck's chart images (1720 x 990)
    fig, ax = plt.subplots(figsize=(8.6, 4.95), dpi=200)
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    ax.axhline(1.0, color=AXIS_TEXT, linewidth=0.8, linestyle="--", alpha=0.6)
    dead_labels = []

    # y-axis sized to the largest sleeve other than naive MVO (e.g. ridge's 2018 peak),
    # so that peak is clearly visible. If MVO's own spike goes higher, it is cut off
    # at the top and its peak value is written there instead.
    curves = {}
    for key, label, color, width in SERIES:
        curves[key] = growth_of_one_dollar(daily[key])
    top = 0.0
    for key, label, color, width in SERIES:
        if key != CAP_EXCLUDE and curves[key].max() > top:
            top = curves[key].max()
    y_max = top * 1.12

    end_labels = []
    for key, label, color, width in SERIES:
        eq = curves[key]
        ax.plot(eq.index, eq.values, color=color, linewidth=width, label=label, clip_on=True)

        # peak label for the sleeve that sets the axis (ridge in our data)
        if key != CAP_EXCLUDE and eq.max() >= top and key in ANNOTATE_PEAK:
            peak_day = eq.idxmax()
            ax.annotate("{} peaks at ${:,.0f}\n{}".format(label, eq.max(), peak_day.strftime("%b %Y")),
                        xy=(peak_day, eq.max()), xytext=(14, -6), textcoords="offset points",
                        fontsize=10, color=INK, va="top")

        # a spike that runs off the top of the chart
        if eq.max() > y_max:
            peak_day = eq.idxmax()
            ax.annotate("{} spikes to ${:,.0f} ({})".format(label, eq.max(), peak_day.strftime("%b %Y")),
                        xy=(peak_day, y_max), xytext=(10, -14), textcoords="offset points",
                        fontsize=10, color=INK, va="top",
                        arrowprops={"arrowstyle": "->", "color": AXIS_TEXT, "lw": 0.8})

        # mark the first time it fell below 1 cent
        below = eq[eq < DEAD]
        if len(below) > 0:
            first_day = below.index[0]
            ax.plot([first_day], [0], marker="X", markersize=9, color=color,
                    markeredgecolor=BACKGROUND, markeredgewidth=1.2, zorder=5, clip_on=False)
            dead_labels.append((first_day, label))

        end_labels.append((eq.index[-1], eq.iloc[-1], "${:,.2f}".format(eq.iloc[-1]), color))

    ax.set_ylim(0, y_max)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(dollar_label))
    ax.set_ylabel("Growth of $1", color=AXIS_TEXT, fontsize=11)

    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=AXIS_TEXT, labelsize=10)

    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=len(SERIES), fontsize=10.5, labelcolor=INK)

    fig.tight_layout()
    place_end_labels(ax, end_labels)

    # list when each sleeve first fell below 1 cent, in a small note inside the chart
    if len(dead_labels) > 0:
        lines = []
        for day, label in dead_labels:
            lines.append("x  {} below $0.01: {}".format(label, day.strftime("%b %Y")))
        ax.text(0.99, 0.97, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
                fontsize=9.5, color=INK)
    path = os.path.join(OUT_DIR, "sleeves_growth.png")
    fig.savefig(path, facecolor=BACKGROUND)
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    main()