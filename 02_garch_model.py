
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

    # Returns scaled to percent: standard practice, keeps the optimiser numerically stable
    calib_pct = (calib["log_return"].values) * 100
    full_pct = (pd.concat([calib, crisis])["log_return"].values) * 100

    # 1) Estimate parameters on the calibration window ONLY
    fitted = arch_model(calib_pct, vol="Garch", p=1, q=1, dist="normal").fit(disp="off")

    # 2) ARCH-LM on standardised residuals (raw residuals always show clustering,
    #    so testing them says nothing about whether the model captured it)
    arch_lm_pvalue = het_arch(pd.Series(fitted.std_resid).dropna())[1]

    # 3) Apply the fixed parameters to the full series. conditional_volatility at day t
    #    uses information up to t-1 only, i.e. a one-step-ahead forecast.
    filtered = arch_model(full_pct, vol="Garch", p=1, q=1, dist="normal").fix(fitted.params)
    forecast_vol = np.asarray(filtered.conditional_volatility)[len(calib):] / 100  # back to log-return scale

    result = crisis[["date"]].copy()
    result["garch_forecast_vol"] = forecast_vol                          # sigma forecast
    result["realized_abs_return"] = crisis["log_return"].abs().values    # |return|, converted to a sigma proxy in step 4
    result["actual_return"] = crisis["log_return"].values

    max_abs_return = float(calib["log_return"].abs().max())
    return result, arch_lm_pvalue, fitted.params, max_abs_return


def main():
    os.makedirs(os.path.join(config.OUTPUT_DIR, "garch_forecasts"), exist_ok=True)
    processed_files = sorted(glob.glob(os.path.join(config.PROCESSED_DIR, "*.csv")))

    diagnostics = []

    for path in processed_files:
        ticker = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path)

        try:
            result, arch_lm_pvalue, params, max_abs_return = fit_and_forecast(df, ticker)
        except Exception as e:
            print(f"[SKIPPED] {ticker}: {e}")
            continue

        out_path = os.path.join(config.OUTPUT_DIR, "garch_forecasts", f"{ticker}.csv")
        result.to_csv(out_path, index=False)

        alpha = params.get("alpha[1]", np.nan)
        beta = params.get("beta[1]", np.nan)
        diagnostics.append({
            "ticker": ticker,
            "arch_lm_pvalue": arch_lm_pvalue,
            "no_remaining_arch_effect": arch_lm_pvalue > 0.05,
            "alpha": alpha,
            "beta": beta,
            "persistence": alpha + beta,
            "max_abs_return_calib": max_abs_return,   # a very large value here points to a bad price in the data
        })
        print(f"[OK] {ticker}: GARCH fitted, forecast saved -> {out_path}")

    diag_df = pd.DataFrame(diagnostics)
    diag_path = os.path.join(config.OUTPUT_DIR, "garch_diagnostics.csv")
    diag_df.to_csv(diag_path, index=False)
    print(f"\nGARCH diagnostics saved to {diag_path}")
    print(diag_df)


if __name__ == "__main__":
    main()
