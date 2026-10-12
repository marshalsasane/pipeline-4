"""
STEP 2 — GARCH(1,1) modelling

Fits GARCH(1,1) on the calibration window for every processed stock,
then forecasts volatility across the crisis window.
Also runs an ARCH-LM diagnostic on each fit so you have that check documented.

Run after 01_preprocess.py: python 02_garch_model.py
"""

import os
import glob
import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch

import config


def fit_and_forecast(df, ticker):
    df["date"] = pd.to_datetime(df["date"])

    calib = df[(df["date"] >= config.CALIBRATION_START) & (df["date"] <= config.CALIBRATION_END)]
    crisis = df[(df["date"] >= config.CRISIS_START) & (df["date"] <= config.CRISIS_END)]

    if len(calib) < 30 or len(crisis) == 0:
        raise ValueError(f"Not enough data in calibration ({len(calib)}) or crisis ({len(crisis)}) window")

    # Returns scaled to percent — standard practice, keeps the optimiser numerically stable
    returns_pct = calib["log_return"] * 100

    model = arch_model(returns_pct, vol="Garch", p=1, q=1, dist="normal")
    fitted = model.fit(disp="off")

    # ARCH-LM test on standardised residuals — should show no remaining ARCH effect if the fit is good
    arch_lm_pvalue = het_arch(fitted.resid.dropna())[1]

    # Forecast volatility across the crisis window horizon
    horizon = len(crisis)
    forecast = fitted.forecast(horizon=horizon, reindex=False)
    forecast_variance_pct2 = forecast.variance.values[-1]              # variance, in (percent)^2 units
    forecast_vol = np.sqrt(forecast_variance_pct2) / 100                # back to log-return scale

    result = crisis[["date"]].copy()
    result["garch_forecast_vol"] = forecast_vol
    result["realized_abs_return"] = crisis["log_return"].abs().values  # simple realized volatility proxy
    result["actual_return"] = crisis["log_return"].values

    return result, arch_lm_pvalue, fitted.params


def main():
    os.makedirs(os.path.join(config.OUTPUT_DIR, "garch_forecasts"), exist_ok=True)
    processed_files = sorted(glob.glob(os.path.join(config.PROCESSED_DIR, "*.csv")))

    diagnostics = []

    for path in processed_files:
        ticker = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path)

        try:
            result, arch_lm_pvalue, params = fit_and_forecast(df, ticker)
        except Exception as e:
            print(f"[SKIPPED] {ticker}: {e}")
            continue

        out_path = os.path.join(config.OUTPUT_DIR, "garch_forecasts", f"{ticker}.csv")
        result.to_csv(out_path, index=False)

        diagnostics.append({
            "ticker": ticker,
            "arch_lm_pvalue": arch_lm_pvalue,
            "no_remaining_arch_effect": arch_lm_pvalue > 0.05,
            "alpha": params.get("alpha[1]", np.nan),
            "beta": params.get("beta[1]", np.nan),
        })
        print(f"[OK] {ticker}: GARCH fitted, forecast saved -> {out_path}")

    diag_df = pd.DataFrame(diagnostics)
    diag_path = os.path.join(config.OUTPUT_DIR, "garch_diagnostics.csv")
    diag_df.to_csv(diag_path, index=False)
    print(f"\nGARCH diagnostics saved to {diag_path}")
    print(diag_df)


if __name__ == "__main__":
    main()
