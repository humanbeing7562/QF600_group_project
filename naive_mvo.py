"""Generate monthly naive MVO weights from the shared adjusted-price dataset.

Run: python naive_mvo.py
Optional: python naive_mvo.py --risk-aversion 8
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


LOOKBACK_MONTHS = 12
RISK_AVERSION = 4.0


def monthly_windows(returns, first_price_date):
    """Yield an investment month and its complete-history return panel.

    A month's weights will use only the previous 12 calendar months.
    The first price date anchors the initial estimation period, even though
    its return is unavailable because we have no preceding price.
    """
    first_month = first_price_date.to_period("M") + LOOKBACK_MONTHS
    last_month = returns.index.max().to_period("M")

    for month in pd.period_range(first_month, last_month, freq="M"):
        start = (month - LOOKBACK_MONTHS).start_time
        end = month.start_time
        window = returns.loc[(returns.index >= start) & (returns.index < end)]

        # Exclude stocks with any missing or infinite return in this window.
        eligible = np.isfinite(window).all(axis=0)
        panel = window.loc[:, eligible]
        if panel.empty:
            raise ValueError(f"{month}: no usable historical returns")
        # Centering returns reduces the maximum covariance rank by one.
        # This check is necessary, but does not guarantee invertibility.
        if len(panel) <= panel.shape[1]:
            raise ValueError(f"{month}: too few observations for sample covariance")
        yield month, panel


def summarize_windows(returns, first_price_date):
    rows = []
    for month, panel in monthly_windows(returns, first_price_date):
        excluded = returns.columns.difference(panel.columns)
        rows.append({
            "investment_month": str(month),
            "history_start": panel.index.min().date(),
            "history_end": panel.index.max().date(),
            "observations": len(panel),
            "eligible_stocks": panel.shape[1],
            "excluded_stocks": ", ".join(excluded),
        })
    return pd.DataFrame(rows)


def calculate_weights(returns, first_price_date, risk_aversion=RISK_AVERSION):
    """Use sample daily means and covariance; preserve the raw MVO solution."""
    if not np.isfinite(risk_aversion) or risk_aversion <= 0:
        raise ValueError("Risk aversion must be a finite positive number")

    weight_rows = []
    diagnostics = []
    for month, panel in monthly_windows(returns, first_price_date):
        mean_returns = panel.mean().to_numpy()
        covariance = panel.cov().to_numpy()  # sample covariance, ddof=1
        try:
            # Fail explicitly if the sample matrix is not positive definite.
            # No ridge term or pseudoinverse is added to this naive baseline.
            np.linalg.cholesky(covariance)
            eligible_weights = np.linalg.solve(covariance, mean_returns) / risk_aversion
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"{month}: covariance cannot support naive MVO") from exc
        if not np.isfinite(eligible_weights).all():
            raise ValueError(f"{month}: non-finite portfolio weights")

        # Every output row has all stock columns; ineligible stocks stay zero.
        weights = pd.Series(0.0, index=returns.columns, name=str(month))
        weights.loc[panel.columns] = eligible_weights
        weight_rows.append(weights)
        residual = risk_aversion * covariance @ eligible_weights - mean_returns
        diagnostics.append({
            "investment_month": str(month),
            "risk_aversion": risk_aversion,
            "long_exposure": float(weights.clip(lower=0).sum()),
            "short_exposure": float(-weights.clip(upper=0).sum()),
            "net_exposure": float(weights.sum()),
            "gross_exposure": float(weights.abs().sum()),
            "max_abs_weight": float(weights.abs().max()),
            "covariance_condition_number": float(np.linalg.cond(covariance)),
            "max_abs_equation_residual": float(np.abs(residual).max()),
        })

    if not weight_rows:
        raise ValueError("Not enough data for a complete initial lookback")
    matrix = pd.DataFrame(weight_rows)
    matrix.index.name = "investment_month"
    return matrix, pd.DataFrame(diagnostics)


def main():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risk-aversion", type=float, default=RISK_AVERSION)
    args = parser.parse_args()
    prices = pd.read_csv(
        project_dir / "data" / "prices_adj.csv",
        index_col=0,
        parse_dates=True,
    ).sort_index()
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans:
        raise ValueError("Prices must have valid date labels")
    if not prices.index.is_unique or not prices.columns.is_unique:
        raise ValueError("Price dates and stock tickers must be unique")
    observed = prices.to_numpy()
    if ((observed <= 0) | np.isinf(observed)).any():
        raise ValueError("Observed adjusted prices must be positive and finite")

    print("Dataset shape (rows, stocks):", prices.shape)
    print("First date:", prices.index.min())
    print("Last date:", prices.index.max())

    print("\nFirst five rows (first five stocks):")
    print(prices.iloc[:5, :5])

    print("\nStocks with missing prices:")
    missing = prices.isna().sum()
    print(missing[missing > 0].sort_values(ascending=False))

    # Daily simple return = today's adjusted price / yesterday's price - 1.
    # Do not fill missing prices: unavailable returns must remain missing.
    returns = prices.pct_change(fill_method=None)
    print("\nFirst five daily return rows (first five stocks):")
    print(returns.iloc[:5, :5])
    print("\nReturns are decimal fractions: 0.01 means a 1% return.")

    # Drop only the first date, whose return is unavailable for every stock.
    # Keep later missing values so the eligibility check can detect them.
    estimation_returns = returns.iloc[1:]
    summary = summarize_windows(estimation_returns, prices.index.min())
    print(f"\nMonthly estimation windows ({LOOKBACK_MONTHS}-month lookback):")
    print(summary.head(3).to_string(index=False))
    print("...")
    print(summary.tail(3).to_string(index=False))
    print("\nTotal investment months:", len(summary))
    print("Eligible stock count range:",
          summary["eligible_stocks"].min(), "to", summary["eligible_stocks"].max())

    weights, diagnostics = calculate_weights(
        estimation_returns, prices.index.min(), args.risk_aversion
    )
    diagnostics = summary.merge(diagnostics, on="investment_month", validate="one_to_one")
    output_dir = project_dir / "naive_mvo_results"
    output_dir.mkdir(exist_ok=True)
    weights.to_csv(output_dir / "weights_naive_mvo.csv")
    diagnostics.to_csv(output_dir / "monthly_diagnostics.csv", index=False)
    method = f"""Naive MVO method

Input: data/prices_adj.csv, the group's fixed 100-stock adjusted-price panel.
Daily simple returns: P[t] / P[t-1] - 1, with no missing-price filling.
Lookback: previous {LOOKBACK_MONTHS} calendar months; rebalance monthly.
Timing: each row labels the investment month, using data strictly before it.
Eligibility: require a finite return on every observed date in that window.
The first dataset date has no preceding price and is omitted from returns.
Ineligible stocks receive exactly zero; eligibility is reassessed monthly.
All input ticker columns are retained in their original order.

Inputs to MVO: sample mean daily simple returns and sample daily covariance
(ddof=1), estimated from the same dates for all eligible stocks. No risk-free
rate is subtracted. Both inputs use daily units.
Formula: w = numpy.linalg.solve(covariance, mean_returns) / lambda.
Lambda: {args.risk_aversion:g}. The default 4 follows the Week 5 page 10 example;
it is a fixed baseline choice, not a value optimized on historical results.
No ridge, pseudoinverse, leverage cap, budget constraint or exposure scaling.
Short positions are allowed. Weights need not sum to one or be dollar neutral.
The code stops if covariance is not positive definite or cannot be solved.

Output: weights_naive_mvo.csv, with investment months as rows and tickers as
columns. A value of 0.10 represents a long position worth 10% of portfolio
capital; -0.10 represents a 10% short. Positions may exceed portfolio capital.
Any residual cash/financing, exposure scaling, holding-period weight drift,
transaction costs and performance metrics require the group's backtest rules.
An equal average of model weights does not imply equal risk contributions.

monthly_diagnostics.csv records historical dates, eligibility, exposures,
covariance condition numbers and the residual of lambda * covariance * w = mu.
With this dataset, the initial year is calibration and the output investment
months run from January 2017 to December 2025 (108 months).

Data limitation: selecting a fixed universe by present-day membership/market
capitalization does not reconstruct the historical top 100. It can introduce
survivorship and look-ahead selection bias. The historical window filtering
does not remove that universe-selection limitation.

Reproduce: python naive_mvo.py
Change lambda: python naive_mvo.py --risk-aversion 8
Running again overwrites the result files using the selected lambda.
Dependencies: numpy and pandas.
"""
    (output_dir / "method_note.txt").write_text(method, encoding="utf-8")
    print("\nSaved monthly weights:", output_dir / "weights_naive_mvo.csv")
    print("Weights matrix shape:", weights.shape)
    print("Gross exposure range: {:.2f}x to {:.2f}x".format(
        diagnostics["gross_exposure"].min(), diagnostics["gross_exposure"].max()))
    print("Largest absolute stock weight: {:.2f}x".format(
        diagnostics["max_abs_weight"].max()))
    print("Weights are raw MVO positions; exposure scaling has not been applied.")


if __name__ == "__main__":
    main()
