"""Generate monthly ridge MVO weights from the shared adjusted-price dataset.

Same pipeline as naive_mvo.py (same 12-month lookback, same eligibility rule,
same lambda), but the covariance is regularised with a ridge term:

    w(theta) = (Sigma + theta * I)^-1 mu / lambda

Each month we solve this for every theta in THETA_GRID and take the equal
average of the resulting weight vectors.

Run: python ridge_mvo.py
Optional: python ridge_mvo.py --risk-aversion 8
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from naive_mvo import LOOKBACK_MONTHS, RISK_AVERSION, monthly_windows, summarize_windows


# 1-5 steps per decade: 0.0000001, 0.0000005, 0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, ..., 50
THETA_GRID = np.array([
    1e-6, 5e-6,
    1e-5, 5e-5,
    1e-4, 5e-4,
    1e-3, 5e-3,
    1e-2, 5e-2,
    1e-1, 5e-1,
    1e0, 5e0,
    1e1, 5e1,
])


def ridge_weights(mean_returns, covariance, theta, risk_aversion):
    """Ridge MVO solution for one theta parameter: (Sigma + theta*I)^-1 mu / lambda."""
    ridged = covariance + theta * np.eye(covariance.shape[0])
    return np.linalg.solve(ridged, mean_returns) / risk_aversion


def calculate_weights_ridge(returns, first_price_date, risk_aversion=RISK_AVERSION,
                            theta_grid=THETA_GRID):
    """Monthly ridge MVO weights, averaged across the theta grid.

    Returns (weights, diagnostics, weights_by_theta):
      weights          investment_month x ticker, same layout as weights_naive_mvo
      diagnostics      one row per month
      weights_by_theta {theta: investment_month x ticker} for each grid point
    """
    if not np.isfinite(risk_aversion) or risk_aversion <= 0:
        raise ValueError("Risk aversion must be a finite positive number")
    theta_grid = np.asarray(theta_grid, dtype=float)
    if theta_grid.size == 0 or not np.isfinite(theta_grid).all() or (theta_grid <= 0).any():
        raise ValueError("Every theta must be finite and positive")

    weight_rows = []
    theta_rows = {theta: [] for theta in theta_grid}
    diagnostics = []
    for month, panel in monthly_windows(returns, first_price_date):
        mean_returns = panel.mean().to_numpy()
        covariance = panel.cov().to_numpy()  # sample covariance, ddof=1

        per_theta = []
        for theta in theta_grid:
            try:
                w_theta = ridge_weights(mean_returns, covariance, theta, risk_aversion)
            except np.linalg.LinAlgError as exc:
                raise ValueError(f"{month}: ridge system unsolvable at theta={theta:g}") from exc
            if not np.isfinite(w_theta).all():
                raise ValueError(f"{month}: non-finite weights at theta={theta:g}")
            per_theta.append(w_theta)

            full = pd.Series(0.0, index=returns.columns, name=str(month))
            full.loc[panel.columns] = w_theta
            theta_rows[theta].append(full)

        eligible_weights = np.mean(per_theta, axis=0)

        # Every output row has all stock columns; ineligible stocks stay zero.
        weights = pd.Series(0.0, index=returns.columns, name=str(month))
        weights.loc[panel.columns] = eligible_weights
        weight_rows.append(weights)
        diagnostics.append({
            "investment_month": str(month),
            "risk_aversion": risk_aversion,
            "n_thetas": len(theta_grid),
            "long_exposure": float(weights.clip(lower=0).sum()),
            "short_exposure": float(-weights.clip(upper=0).sum()),
            "net_exposure": float(weights.sum()),
            "gross_exposure": float(weights.abs().sum()),
            "max_abs_weight": float(weights.abs().max()),
            "covariance_condition_number": float(np.linalg.cond(covariance)),
        })

    if not weight_rows:
        raise ValueError("Not enough data for a complete initial lookback")
    matrix = pd.DataFrame(weight_rows)
    matrix.index.name = "investment_month"
    by_theta = {}
    for theta, rows in theta_rows.items():
        frame = pd.DataFrame(rows)
        frame.index.name = "investment_month"
        by_theta[theta] = frame
    return matrix, pd.DataFrame(diagnostics), by_theta


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

    # Same return construction as naive_mvo.py: no filling of missing prices.
    returns = prices.pct_change(fill_method=None)
    estimation_returns = returns.iloc[1:]
    summary = summarize_windows(estimation_returns, prices.index.min())
    print("Dataset shape (rows, stocks):", prices.shape)
    print("Total investment months:", len(summary))
    print("Theta grid:", ", ".join(f"{t:g}" for t in THETA_GRID))

    weights, diagnostics, by_theta = calculate_weights_ridge(
        estimation_returns, prices.index.min(), args.risk_aversion
    )
    diagnostics = summary.merge(diagnostics, on="investment_month", validate="one_to_one")

    output_dir = project_dir / "ridge_mvo_results"
    output_dir.mkdir(exist_ok=True)
    weights.to_csv(output_dir / "weights_ridge_mvo.csv")
    diagnostics.to_csv(output_dir / "monthly_diagnostics.csv", index=False)

    theta_summary = pd.DataFrame([{
        "theta": theta,
        "mean_gross_exposure": float(frame.abs().sum(axis=1).mean()),
        "mean_max_abs_weight": float(frame.abs().max(axis=1).mean()),
    } for theta, frame in by_theta.items()])
    theta_summary.to_csv(output_dir / "theta_summary.csv", index=False)

    print("\nSaved monthly weights:", output_dir / "weights_ridge_mvo.csv")
    print("Weights matrix shape:", weights.shape)
    print("\nPer-theta behaviour (averaged over months):")
    print(theta_summary.to_string(index=False))
    print("\nAveraged-weights gross exposure range: {:.2f}x to {:.2f}x".format(
        diagnostics["gross_exposure"].min(), diagnostics["gross_exposure"].max()))


if __name__ == "__main__":
    main()
