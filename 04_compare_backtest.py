"""
STEP 4 — Forecast comparison and VaR backtesting

Compares GARCH vs LSTM forecast accuracy (RMSE, MAE) per stock, then
converts each model's volatility forecast into a VaR estimate and runs
the Kupiec (1995) proportion-of-failures test.

MAPE is deliberately NOT used here: the target being forecast (absolute
daily return, as a volatility proxy) sits near zero on calm trading days,
so MAPE's division by the actual value produces enormous, meaningless
percentages on those days. This is a well-documented limitation of MAPE
for volatility forecasting, not a bug in the models. MAE is reported
instead — same interpretability (average error, in return units), no
division, so no blow-up.

Run last: python 04_compare_backtest.py
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import norm, chi2

import config

Z_SCORE = norm.ppf(1 - (1 - config.VAR_CONFIDENCE))  # e.g. 2.326 for 99% VaR


def rmse(actual, forecast):
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def mae(actual, forecast):
    return float(np.mean(np.abs(actual - forecast)))


def kupiec_test(actual_returns, forecast_vol, confidence=config.VAR_CONFIDENCE):
    """
    Kupiec (1995) proportion-of-failures (POF) test.
    Returns the number of violations, the LR statistic, and whether the
    model is rejected at the 5% significance level (chi-square critical value 3.841).
    """
    p = 1 - confidence
    var_estimate = Z_SCORE * forecast_vol
    violations = (actual_returns < -var_estimate).astype(int)
    n = len(violations)
    x = int(violations.sum())

    if x == 0 or x == n:
        # LR statistic is undefined at the boundary; flag rather than compute
        return {"violations": x, "n_obs": n, "lr_statistic": np.nan, "rejected_at_5pct": None}

    pi_hat = x / n
    lr = -2 * (
        (n - x) * np.log(1 - p) + x * np.log(p)
        - ((n - x) * np.log(1 - pi_hat) + x * np.log(pi_hat))
    )
    rejected = lr > chi2.ppf(0.95, df=1)

    return {"violations": x, "n_obs": n, "lr_statistic": float(lr), "rejected_at_5pct": bool(rejected)}


def main():
    garch_files = sorted(glob.glob(os.path.join(config.OUTPUT_DIR, "garch_forecasts", "*.csv")))

    accuracy_rows = []
    kupiec_rows = []

    for garch_path in garch_files:
        ticker = os.path.splitext(os.path.basename(garch_path))[0]
        lstm_path = os.path.join(config.OUTPUT_DIR, "lstm_forecasts", f"{ticker}.csv")

        if not os.path.exists(lstm_path):
            print(f"[SKIPPED] {ticker}: no matching LSTM forecast found")
            continue

        garch_df = pd.read_csv(garch_path)
        lstm_df = pd.read_csv(lstm_path)

        merged = garch_df.merge(lstm_df, on=["date", "realized_abs_return", "actual_return"])

        accuracy_rows.append({
            "ticker": ticker,
            "garch_rmse": rmse(merged["realized_abs_return"], merged["garch_forecast_vol"]),
            "lstm_rmse": rmse(merged["realized_abs_return"], merged["lstm_forecast_vol"]),
            "garch_mae": mae(merged["realized_abs_return"], merged["garch_forecast_vol"]),
            "lstm_mae": mae(merged["realized_abs_return"], merged["lstm_forecast_vol"]),
        })

        garch_kupiec = kupiec_test(merged["actual_return"].values, merged["garch_forecast_vol"].values)
        lstm_kupiec = kupiec_test(merged["actual_return"].values, merged["lstm_forecast_vol"].values)

        kupiec_rows.append({"ticker": ticker, "model": "GARCH", **garch_kupiec})
        kupiec_rows.append({"ticker": ticker, "model": "LSTM", **lstm_kupiec})

        print(f"[OK] {ticker}: accuracy and Kupiec test computed")

    accuracy_df = pd.DataFrame(accuracy_rows)
    kupiec_df = pd.DataFrame(kupiec_rows)

    accuracy_path = os.path.join(config.OUTPUT_DIR, "forecast_accuracy_comparison.csv")
    kupiec_path = os.path.join(config.OUTPUT_DIR, "var_backtest_kupiec.csv")
    accuracy_df.to_csv(accuracy_path, index=False)
    kupiec_df.to_csv(kupiec_path, index=False)

    print(f"\nForecast accuracy comparison saved to {accuracy_path}")
    print(accuracy_df)
    print(f"\nVaR backtest (Kupiec) results saved to {kupiec_path}")
    print(kupiec_df)

    print("\n--- Summary across all stocks ---")
    print(f"Average GARCH RMSE: {accuracy_df['garch_rmse'].mean():.6f}")
    print(f"Average LSTM RMSE:  {accuracy_df['lstm_rmse'].mean():.6f}")
    print(f"Average GARCH MAE:  {accuracy_df['garch_mae'].mean():.6f}")
    print(f"Average LSTM MAE:   {accuracy_df['lstm_mae'].mean():.6f}")
    print(
        "\nNote: with a crisis window this short, the Kupiec test has limited "
        "statistical power — treat rejected_at_5pct as indicative, not conclusive, "
        "and say so explicitly in your Results/Limitations section."
    )


if __name__ == "__main__":
    main()
