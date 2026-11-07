"""
Ridge MVO weights averaged over different theta grids.

Uses ridge_mvo.calculate_weights_ridge as is (same data, same 12-month lookback,
same lambda). Only the set of thetas being averaged changes. Each grid's weights
are saved under their own name, so nothing in ridge_mvo_results/ is overwritten.

Grids:
  full     the current grid in ridge_mvo.py (1e-6 to 50)
  from1e-4 drops the smallest thetas, which give the biggest weights and pull
           the average back toward naive MVO
  from1e-2 only the heavier thetas, a more strongly regularised average

Outputs (in sweep_results/ridge/):
  weights_ridge_mvo_{grid}.csv       monthly weights, same layout as weights_ridge_mvo.csv
  monthly_diagnostics_{grid}.csv     same diagnostics as ridge_mvo.py

Usage:
  python sweep_ridge.py

Then expand to daily as usual, e.g.
  python expand_monthly_weights.py --weights sweep_results/ridge/weights_ridge_mvo_from1e-4.csv --out sweep_results/ridge/weights_ridge_mvo_from1e-4_daily.csv
"""

import os

import numpy as np
import pandas as pd

from ridge_mvo import THETA_GRID, calculate_weights_ridge


PRICES_FILE = os.path.join("data", "prices_adj.csv")
OUT_DIR = os.path.join("sweep_results", "ridge")


def grid_from(smallest):
    """The current grid, keeping only thetas >= smallest."""
    kept = []
    for theta in THETA_GRID:
        if theta >= smallest:
            kept.append(theta)
    return np.array(kept)


GRIDS = {
    "full": THETA_GRID,
    "from1e-4": grid_from(1e-4),
    "from1e-2": grid_from(1e-2),
}


def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # same steps as ridge_mvo.main()
    prices = pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True).sort_index()
    returns = prices.pct_change(fill_method=None)
    estimation_returns = returns.iloc[1:]

    for name in GRIDS:
        grid = GRIDS[name]
        thetas = []
        for t in grid:
            thetas.append("{:g}".format(t))
        print("grid {}: {}".format(name, ", ".join(thetas)))

        weights, diagnostics, by_theta = calculate_weights_ridge(
            estimation_returns, prices.index.min(), theta_grid=grid)

        weights_path = os.path.join(OUT_DIR, "weights_ridge_mvo_{}.csv".format(name))
        diag_path = os.path.join(OUT_DIR, "monthly_diagnostics_{}.csv".format(name))
        weights.to_csv(weights_path)
        diagnostics.to_csv(diag_path, index=False)

        print("   saved {}  (gross exposure {:.2f}x to {:.2f}x)".format(
            weights_path, diagnostics["gross_exposure"].min(), diagnostics["gross_exposure"].max()))


if __name__ == "__main__":
    main()