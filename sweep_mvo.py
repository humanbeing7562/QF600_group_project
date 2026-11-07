"""
Naive MVO weights for several risk aversion (lambda) values.

Uses naive_mvo.calculate_weights as is (same data, same 12-month lookback),
and saves each lambda's weights under its own file name, so nothing in
naive_mvo_results/ is overwritten.

Outputs (in sweep_results/mvo/):
  weights_naive_mvo_lam{L}.csv       monthly weights, same layout as weights_naive_mvo.csv
  monthly_diagnostics_lam{L}.csv     same diagnostics as naive_mvo.py

Usage:
  python sweep_mvo.py

Then expand to daily as usual, e.g.
  python expand_monthly_weights.py --weights sweep_results/mvo/weights_naive_mvo_lam10.csv --out sweep_results/mvo/weights_naive_mvo_lam10_daily.csv
"""

import os

import pandas as pd

from naive_mvo import calculate_weights


LAMBDAS = [1, 4, 10, 50, 200]
PRICES_FILE = os.path.join("data", "prices_adj.csv")
OUT_DIR = os.path.join("sweep_results", "mvo")


def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # same steps as naive_mvo.main()
    prices = pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True).sort_index()
    returns = prices.pct_change(fill_method=None)
    estimation_returns = returns.iloc[1:]

    for lam in LAMBDAS:
        weights, diagnostics = calculate_weights(estimation_returns, prices.index.min(), lam)

        weights_path = os.path.join(OUT_DIR, "weights_naive_mvo_lam{:g}.csv".format(lam))
        diag_path = os.path.join(OUT_DIR, "monthly_diagnostics_lam{:g}.csv".format(lam))
        weights.to_csv(weights_path)
        diagnostics.to_csv(diag_path, index=False)

        print("lambda {:g}: saved {}  (gross exposure {:.2f}x to {:.2f}x)".format(
            lam, weights_path,
            diagnostics["gross_exposure"].min(), diagnostics["gross_exposure"].max()))


if __name__ == "__main__":
    main()