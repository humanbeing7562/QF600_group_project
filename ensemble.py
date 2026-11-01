"""
Ensemble the three sleeves (naive MVO, ridge, pairs) and compare with the S&P 500.

Methods:
  1. Average        - 1/3 of each sleeve's weights, every day
  2. FTL daily      - hold today the sleeve with the best return YESTERDAY
  3. FTL monthly    - hold this month the sleeve with the best return LAST MONTH
  4. Oracle         - hold today the sleeve with the best return TODAY. Uses future
                      information, so it is not tradable: it is the upper bound of
                      what switching between the three sleeves could achieve.
  Benchmarks        - buy and hold SPY (adjusted close, so dividends included)
                    - 1/N: equal weight in every eligible stock, rebalanced monthly.
                      Eligible = a return on every day of the previous 12 calendar
                      months, the same rule naive_mvo.py uses, so 1/N and MVO choose
                      from the same stocks each month.

Weights are used raw (no rescaling). All weight files follow the same rule:
the row dated d = weights at the close of day d, earning day d+1's return.

Outputs (in ensemble_results/):
  daily_returns.csv       daily returns of every sleeve, method and SPY
  metrics.csv             annual return, volatility, Sharpe, max drawdown, ...
  weights_average.csv     daily stock weights of the average method
  weights_ftl_daily.csv   daily stock weights of FTL daily
  weights_ftl_monthly.csv daily stock weights of FTL monthly
  weights_oracle.csv      daily stock weights of the Oracle
  weights_equal.csv       daily stock weights of the 1/N benchmark
  leaders.csv             which sleeve each FTL method held each day
  equity_curve.png              growth of $1, compounded
  cumulative_return_simple.png  sum of daily returns (still readable if a strategy is wiped out)
  drawdown.png, rolling_sharpe.png
  each chart also as *_with_sleeves.png, adding the MVO, ridge and pairs curves,
  and as *_with_oracle.png, adding the Oracle

Usage:
  python ensemble.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRICES_FILE = os.path.join("data", "ohlcv_raw.csv")
SPY_FILE = os.path.join("data", "spy.csv")          # downloaded once, then reused
SLEEVE_FILES = {
    "MVO": os.path.join("naive_mvo_results", "weights_naive_mvo_daily.csv"),
    "Ridge": os.path.join("ridge_mvo_results", "weights_ridge_mvo_daily.csv"),
    "Pairs": os.path.join("pairs_results", "pairs_weights_daily.csv"),
}
OUT_DIR = "ensemble_results"

START = "2017-01-01"          # first month all sleeves hold positions
END = "2025-12-31"
TRADING_DAYS = 252
ROLLING_WINDOW = 252          # rolling Sharpe window (about 1 year)

COLORS = {
    "Average": "#2a78d6",
    "FTL daily": "#eb6834",
    "FTL monthly": "#1baf7a",
    "S&P 500": "#8a897f",
    "Oracle": "#6250d6",
    "1/N": "#e34948",
    "MVO": "#eda100",
    "Ridge": "#e87ba4",
    "Pairs": "#008300",
}
INK = "#2b2b29"
MUTED = "#8a897f"
GRID = "#e6e5df"
SURFACE = "#fcfcfb"


# ---------------------------------------------------------------- data

def load_prices():
    df = pd.read_csv(PRICES_FILE, parse_dates=["date"])
    prices = df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    return prices


def load_stock_returns():
    prices = load_prices()
    # A missing price mid-sample is carried forward (a 0% day), so a position
    # held through a data gap still gets the full move when trading resumes.
    # Before a stock's first price the return stays missing (no position anyway).
    returns = prices.ffill().pct_change(fill_method=None)
    return returns


def load_weights(path, tickers, dates):
    w = pd.read_csv(path, index_col=0, parse_dates=True)
    if "n_cohorts" in w.columns:
        w = w.drop(columns=["n_cohorts"])
    w = w.reindex(index=dates, columns=tickers).fillna(0.0)
    return w


def load_spy():
    if not os.path.exists(SPY_FILE):
        import yfinance as yf
        print("downloading SPY ...")
        raw = yf.download("SPY", start="2016-01-01", end="2026-01-01",
                          auto_adjust=False, progress=False)
        adj = raw["Adj Close"]
        if isinstance(adj, pd.DataFrame):
            adj = adj.iloc[:, 0]
        spy = pd.DataFrame({"adj_close": adj})
        spy.index.name = "date"
        spy.to_csv(SPY_FILE)
    spy = pd.read_csv(SPY_FILE, index_col=0, parse_dates=True)
    return spy["adj_close"].ffill().pct_change(fill_method=None)


# ---------------------------------------------------------------- sleeves

def sleeve_returns(weights, returns):
    """Return on day t = weights at the close of day t-1 x stock returns on day t."""
    held = weights.shift(1)
    r = returns.fillna(0.0)
    return (held * r).sum(axis=1)


def equal_weight_monthly(prices, lookback_months=12):
    """
    1/N weights for each investment month, using the same eligibility rule as
    naive_mvo.py: a stock is eligible if it has a return on every day of the
    previous 12 calendar months (returns with no filling of missing prices).
    """
    raw_returns = prices.pct_change(fill_method=None).iloc[1:]
    first_month = prices.index.min().to_period("M") + lookback_months
    last_month = prices.index.max().to_period("M")

    rows = []
    labels = []
    for month in pd.period_range(first_month, last_month, freq="M"):
        start = (month - lookback_months).start_time
        end = month.start_time
        window = raw_returns.loc[(raw_returns.index >= start) & (raw_returns.index < end)]
        eligible = np.isfinite(window).all(axis=0)
        n = int(eligible.sum())
        w = pd.Series(0.0, index=prices.columns)
        if n > 0:
            w[eligible] = 1.0 / n
        rows.append(w.values)
        labels.append(str(month))
    return pd.DataFrame(rows, index=labels, columns=prices.columns)


def monthly_to_daily(monthly, dates):
    """Same rule as expand_monthly_weights.py: row d holds the weights of the
    month that the next trading day falls in."""
    rows = []
    for i in range(len(dates)):
        if i + 1 < len(dates):
            month = str(dates[i + 1].to_period("M"))
        else:
            month = str(dates[i].to_period("M"))
        if month in monthly.index:
            rows.append(monthly.loc[month].values)
        else:
            rows.append(np.zeros(monthly.shape[1]))
    return pd.DataFrame(rows, index=dates, columns=monthly.columns)


# ---------------------------------------------------------------- ensembles

def average_method(sleeve_weights, names):
    total = None
    for name in names:
        if total is None:
            total = sleeve_weights[name].copy()
        else:
            total = total + sleeve_weights[name]
    return total / len(names)


def best_sleeve(values, names):
    """
    Name of the sleeve with the highest value.
    None if all are missing or all exactly zero (nothing was held yet), so the
    method sits in cash instead of picking the first sleeve on a tie.
    """
    leader = None
    best = None
    all_zero = True
    for name in names:
        v = values[name]
        if pd.isna(v):
            continue
        if v != 0:
            all_zero = False
        if best is None or v > best:
            best = v
            leader = name
    if all_zero:
        return None
    return leader


def ftl_daily(sleeve_weights, sleeve_rets, names, dates):
    """
    At the close of day d we know each sleeve's return on day d,
    so we pick the best one and hold its weights (row d) for day d+1.
    """
    tickers = sleeve_weights[names[0]].columns
    rows = []
    leaders = []
    for d in dates:
        leader = best_sleeve(sleeve_rets.loc[d], names)
        leaders.append(leader)
        if leader is None:
            rows.append(np.zeros(len(tickers)))
        else:
            rows.append(sleeve_weights[leader].loc[d].values)
    weights = pd.DataFrame(rows, index=dates, columns=tickers)
    return weights, pd.Series(leaders, index=dates)


def oracle(sleeve_weights, sleeve_rets, names, dates):
    """
    Look-ahead benchmark: the weight row dated d earns day d+1's return, so it
    holds the sleeve whose return on day d+1 turns out to be the highest.
    """
    tickers = sleeve_weights[names[0]].columns
    rows = []
    leaders = []
    for i in range(len(dates)):
        if i + 1 < len(dates):
            leader = best_sleeve(sleeve_rets.iloc[i + 1], names)
        else:
            leader = None
        leaders.append(leader)
        if leader is None:
            rows.append(np.zeros(len(tickers)))
        else:
            rows.append(sleeve_weights[leader].iloc[i].values)
    weights = pd.DataFrame(rows, index=dates, columns=tickers)
    return weights, pd.Series(leaders, index=dates)


def ftl_monthly(sleeve_weights, sleeve_rets, names, dates):
    """
    At each month-end we pick the sleeve with the best return over that month
    and hold it for the whole next month. The weight row dated d earns day d+1,
    so it uses the leader chosen for the month that day d+1 falls in.
    """
    months = dates.to_period("M")

    # best sleeve of each calendar month (compounded return over the month)
    month_leader = {}
    for m in months.unique():
        block = sleeve_rets.loc[months == m]
        month_return = {}
        for name in names:
            growth = 1.0
            for r in block[name]:
                growth = growth * (1 + r)
            month_return[name] = growth - 1
        month_leader[m] = best_sleeve(month_return, names)

    tickers = sleeve_weights[names[0]].columns
    rows = []
    leaders = []
    for i in range(len(dates)):
        if i + 1 < len(dates):
            holding_month = months[i + 1]
        else:
            holding_month = months[i]
        previous_month = holding_month - 1
        leader = month_leader.get(previous_month)

        leaders.append(leader)
        if leader is None:
            rows.append(np.zeros(len(tickers)))
        else:
            rows.append(sleeve_weights[leader].iloc[i].values)

    weights = pd.DataFrame(rows, index=dates, columns=tickers)
    return weights, pd.Series(leaders, index=dates)


# ---------------------------------------------------------------- metrics

def equity_curve(daily):
    """
    Growth of $1 with returns reinvested. If a day loses 100% or more the
    account is gone: the value is set to 0 and stays there.
    """
    values = []
    value = 1.0
    for r in daily:
        if value > 0:
            value = value * (1 + r)
            if value < 0:
                value = 0.0
        values.append(value)
    return pd.Series(values, index=daily.index)


def simple_cumulative(daily):
    """
    Sum of daily returns: the P&L of a fixed $1 bet that is never reinvested.
    Unlike compounding it still means something after a day worse than -100%,
    which can happen with very large gross exposure (e.g. raw naive MVO).
    """
    values = []
    total = 0.0
    for r in daily:
        total = total + r
        values.append(total)
    return pd.Series(values, index=daily.index)


def simple_drawdown(cumulative):
    """Fall from the previous peak of the summed returns, in return points."""
    peak = 0.0
    out = []
    for v in cumulative:
        if v > peak:
            peak = v
        out.append(v - peak)
    return pd.Series(out, index=cumulative.index)


def drawdown_series(equity):
    """Fall from the highest value so far, starting from the initial $1."""
    peak = 1.0
    out = []
    for v in equity:
        if v > peak:
            peak = v
        out.append(v / peak - 1)
    return pd.Series(out, index=equity.index)


def compute_metrics(name, daily):
    equity = equity_curve(daily)
    dd = drawdown_series(equity)
    n_years = len(daily) / TRADING_DAYS

    wiped_out = None
    for d in daily.index:
        if daily[d] <= -1:
            wiped_out = d
            break

    final = equity.iloc[-1]
    if final > 0:
        annual_return = final ** (1 / n_years) - 1
    else:
        annual_return = np.nan

    vol = daily.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = daily.mean() / daily.std(ddof=1) * np.sqrt(TRADING_DAYS)

    n_up = 0
    for r in daily:
        if r > 0:
            n_up = n_up + 1

    cum = simple_cumulative(daily)

    return {
        "strategy": name,
        "final_value_of_1": final,
        "total_return_simple": cum.iloc[-1],
        "max_drawdown_simple": simple_drawdown(cum).min(),
        "annual_return": annual_return,
        "annual_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": dd.min(),
        "worst_day": daily.min(),
        "best_day": daily.max(),
        "pct_up_days": n_up / len(daily),
        "wiped_out_on": "" if wiped_out is None else str(wiped_out.date()),
    }


def rolling_sharpe(daily, window):
    mean = daily.rolling(window).mean()
    std = daily.rolling(window).std(ddof=1)
    return mean / std * np.sqrt(TRADING_DAYS)


# ---------------------------------------------------------------- charts

def money(x):
    if abs(x) >= 1e6:
        return "${:.2e}".format(x)
    return "${:,.2f}".format(x)


def percent(x):
    if pd.isna(x):
        return "NaN"
    if abs(x) >= 100:          # more than 10,000%
        return "{:.2e}%".format(x * 100)
    return "{:.2f}%".format(x * 100)


def style_axes(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, loc="left", fontsize=13)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def save_chart(fig, name):
    fig.tight_layout()
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def label_line_ends(ax, items):
    """
    Write 'name  value' at the right end of each line, nudging labels apart
    so lines that end at nearly the same value don't overlap.
    items: list of (name, last_date, last_value, text)
    """
    fig = ax.get_figure()
    fig.canvas.draw()
    to_pixels = ax.transData

    placed = []
    for name, x, y, text in items:
        px, py = to_pixels.transform((matplotlib.dates.date2num(x), y))
        placed.append([py, px, y, x, text])
    placed.sort()

    min_gap = 16          # pixels between labels
    for i in range(1, len(placed)):
        if placed[i][0] - placed[i - 1][0] < min_gap:
            placed[i][0] = placed[i - 1][0] + min_gap

    for py, px, y, x, text in placed:
        _, true_py = to_pixels.transform((matplotlib.dates.date2num(x), y))
        ax.annotate(text, xy=(x, y), xytext=(6, py - true_py),
                    textcoords="offset pixels", va="center", fontsize=9, color=INK)


SLEEVE_NAMES = ["MVO", "Ridge", "Pairs"]


def line_width(name):
    # sleeves drawn thinner so the ensemble methods stay in front
    if name in SLEEVE_NAMES:
        return 1.3
    return 2


def plot_equity(daily_rets, names, filename):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(SURFACE)

    # Log scale when nothing is wiped out, or when values span many orders of
    # magnitude (e.g. the Oracle). On a log scale a wiped-out curve just ends.
    any_wiped = False
    biggest = 0.0
    for name in names:
        eq = equity_curve(daily_rets[name])
        if (eq <= 0).any():
            any_wiped = True
        if eq.max() > biggest:
            biggest = eq.max()
    use_log = (not any_wiped) or biggest > 1000

    labels = []
    for name in names:
        eq = equity_curve(daily_rets[name])
        if use_log:
            eq = eq.where(eq > 0)          # drop zeros, the line stops at the wipe-out
        ax.plot(eq.index, eq.values, color=COLORS[name], linewidth=line_width(name), label=name)
        last_date = eq.last_valid_index()
        last_value = eq[last_date]
        if eq.iloc[-1] != eq.iloc[-1] or eq.iloc[-1] <= 0.005:
            text = "{}  wiped out".format(name)
            if not use_log:
                last_date = eq.index[-1]
                last_value = eq.iloc[-1]
        else:
            text = "{}  {}".format(name, money(eq.iloc[-1]))
        labels.append((name, last_date, last_value, text))
    ax.axhline(1.0, color=MUTED, linewidth=1, linestyle="--", label="Break-even ($1)")
    ylabel = "Value of $1 invested"
    if use_log:
        ax.set_yscale("log")
        ylabel = ylabel + " (log scale)"
    style_axes(ax, "Growth of $1: ensemble methods vs S&P 500", ylabel)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    label_line_ends(ax, labels)
    save_chart(fig, filename)


def plot_simple_cumulative(daily_rets, names, filename):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(SURFACE)
    labels = []
    for name in names:
        cum = simple_cumulative(daily_rets[name]) * 100
        ax.plot(cum.index, cum.values, color=COLORS[name], linewidth=line_width(name), label=name)
        labels.append((name, cum.index[-1], cum.iloc[-1], "{}  {:+,.0f}%".format(name, cum.iloc[-1])))
    ax.axhline(0.0, color=MUTED, linewidth=1)
    style_axes(ax, "Cumulative return, not reinvested (sum of daily returns)",
               "Cumulative return (%)")
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    label_line_ends(ax, labels)
    save_chart(fig, filename)


def plot_drawdown(daily_rets, names, filename):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor(SURFACE)
    for name in names:
        dd = drawdown_series(equity_curve(daily_rets[name]))
        ax.plot(dd.index, dd.values * 100, color=COLORS[name], linewidth=line_width(name) * 0.75, label=name)
    style_axes(ax, "Drawdown from previous peak", "Drawdown (%)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1),
              ncol=len(names), labelcolor=INK)
    save_chart(fig, filename)


def plot_rolling_sharpe(daily_rets, names, filename):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor(SURFACE)
    for name in names:
        rs = rolling_sharpe(daily_rets[name], ROLLING_WINDOW)
        ax.plot(rs.index, rs.values, color=COLORS[name], linewidth=line_width(name) * 0.75, label=name)
    ax.axhline(0.0, color=MUTED, linewidth=1)
    style_axes(ax, "Rolling 1-year Sharpe ratio", "Sharpe (annualized)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1),
              ncol=len(names), labelcolor=INK)
    save_chart(fig, filename)


# ---------------------------------------------------------------- main

def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    prices = load_prices()
    returns = load_stock_returns()
    tickers = list(returns.columns)
    dates = returns.index

    names = ["MVO", "Ridge", "Pairs"]
    sleeve_weights = {}
    for name in names:
        sleeve_weights[name] = load_weights(SLEEVE_FILES[name], tickers, dates)
        gross = sleeve_weights[name].abs().sum(axis=1)
        held = gross[gross > 0]
        print("{:6s} average gross exposure on held days: {:.2f}".format(name, held.mean()))

    sleeve_rets = pd.DataFrame(index=dates)
    for name in names:
        sleeve_rets[name] = sleeve_returns(sleeve_weights[name], returns)

    # build each method's weights, then its returns the same way as a sleeve
    w_avg = average_method(sleeve_weights, names)
    w_ftl_d, lead_d = ftl_daily(sleeve_weights, sleeve_rets, names, dates)
    w_ftl_m, lead_m = ftl_monthly(sleeve_weights, sleeve_rets, names, dates)
    w_oracle, lead_o = oracle(sleeve_weights, sleeve_rets, names, dates)

    daily = sleeve_rets.copy()
    daily["Average"] = sleeve_returns(w_avg, returns)
    daily["FTL daily"] = sleeve_returns(w_ftl_d, returns)
    daily["FTL monthly"] = sleeve_returns(w_ftl_m, returns)
    daily["Oracle"] = sleeve_returns(w_oracle, returns)
    daily["S&P 500"] = load_spy().reindex(dates)

    w_equal = monthly_to_daily(equal_weight_monthly(prices), dates)
    daily["1/N"] = sleeve_returns(w_equal, returns)

    # evaluation window
    daily = daily.loc[START:END].fillna(0.0)

    # check: the average method's return should equal the average of the sleeves
    gap = (daily["Average"] - (daily["MVO"] + daily["Ridge"] + daily["Pairs"]) / 3).abs().max()
    print("check - average method vs mean of sleeve returns, max gap: {:.2e}".format(gap))

    daily.to_csv(os.path.join(OUT_DIR, "daily_returns.csv"))
    w_avg.loc[:END].to_csv(os.path.join(OUT_DIR, "weights_average.csv"))
    w_ftl_d.loc[:END].to_csv(os.path.join(OUT_DIR, "weights_ftl_daily.csv"))
    w_ftl_m.loc[:END].to_csv(os.path.join(OUT_DIR, "weights_ftl_monthly.csv"))
    w_oracle.loc[:END].to_csv(os.path.join(OUT_DIR, "weights_oracle.csv"))
    w_equal.loc[:END].to_csv(os.path.join(OUT_DIR, "weights_equal.csv"))
    leaders = pd.DataFrame({"ftl_daily_holds": lead_d, "ftl_monthly_holds": lead_m,
                            "oracle_holds": lead_o})
    leaders.loc[:END].to_csv(os.path.join(OUT_DIR, "leaders.csv"))

    order = ["Average", "FTL daily", "FTL monthly", "Oracle", "S&P 500", "1/N", "MVO", "Ridge", "Pairs"]
    rows = []
    for name in order:
        rows.append(compute_metrics(name, daily[name]))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False)

    pd.set_option("display.width", 200)
    print()
    print("Results {} to {} (no transaction costs)".format(daily.index[0].date(), daily.index[-1].date()))
    show = metrics.copy()
    for col in ["total_return_simple", "max_drawdown_simple", "annual_return", "annual_vol",
                "max_drawdown", "worst_day", "best_day", "pct_up_days"]:
        formatted = []
        for x in show[col]:
            formatted.append(percent(x))
        show[col] = formatted
    show["sharpe"] = show["sharpe"].round(2)
    formatted = []
    for x in show["final_value_of_1"]:
        formatted.append(money(x))
    show["final_value_of_1"] = formatted
    print(show.to_string(index=False))

    print()
    print("how often each sleeve was held:")
    held_window = leaders.loc[START:END]
    for col in held_window.columns:
        counts = held_window[col].value_counts(normalize=True)
        parts = []
        for name in names:
            parts.append("{} {:.0f}%".format(name, counts.get(name, 0.0) * 100))
        print("  {:18s} {}".format(col, ", ".join(parts)))

    # each chart twice: ensemble methods only, and with the individual sleeves added
    methods = ["Average", "FTL daily", "FTL monthly", "S&P 500", "1/N"]
    with_sleeves = methods + SLEEVE_NAMES
    # the Oracle gets its own charts: its curve dwarfs the others and would flatten them
    with_oracle = methods + ["Oracle"]
    chart_sets = [(methods, ""), (with_sleeves, "_with_sleeves"), (with_oracle, "_with_oracle")]
    for chart_names, suffix in chart_sets:
        plot_equity(daily, chart_names, "equity_curve" + suffix + ".png")
        plot_simple_cumulative(daily, chart_names, "cumulative_return_simple" + suffix + ".png")
        plot_drawdown(daily, chart_names, "drawdown" + suffix + ".png")
        plot_rolling_sharpe(daily, chart_names, "rolling_sharpe" + suffix + ".png")


if __name__ == "__main__":
    main()