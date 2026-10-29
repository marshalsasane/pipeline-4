
import os
import glob
import numpy as np
import pandas as pd

import config

SECTORS = {
    # Draft ICB-style classification — VERIFY against your methodology chapter
    "AZN":  "Healthcare",
    "BA":   "Industrials",
    "BARC": "Financials",
    "CPG":  "Consumer Services",
    "EXPN": "Industrials",
    "HSBA": "Financials",
    "LGEN": "Financials",
    "NG":   "Utilities",
    "NXT":  "Consumer Discretionary",
    "RIO":  "Materials",
    "RR":   "Industrials",
    "SGRO": "Real Estate",
    "SHEL": "Energy",
    "TSCO": "Consumer Staples",
    "ULVR": "Consumer Staples",
}


def describe(series):
    return {
        "n_obs": int(series.count()),
        "mean": series.mean(),
        "std_dev": series.std(),
        "min": series.min(),
        "max": series.max(),
        "skewness": series.skew(),
        "kurtosis": series.kurtosis(),  # excess kurtosis
    }


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    rows = []
    for path in sorted(glob.glob(os.path.join(config.PROCESSED_DIR, "*.csv"))):
        # Processed files are named e.g. "Download Data - STOCK_UK_XLON_AZN.csv";
        # keep the full name for merging with the accuracy CSV, and derive a
        # short ticker for the sector lookup.
        full_name = os.path.splitext(os.path.basename(path))[0]
        ticker = full_name.removeprefix("Download Data - STOCK_UK_XLON_")

        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])

        calib_mask = (df["date"] >= config.CALIBRATION_START) & (df["date"] <= config.CALIBRATION_END)
        crisis_mask = (df["date"] >= config.CRISIS_START) & (df["date"] <= config.CRISIS_END)
        calib = df.loc[calib_mask, "log_return"].dropna()
        crisis = df.loc[crisis_mask, "log_return"].dropna()
        calib_dates = df.loc[calib_mask, "date"]
        crisis_dates = df.loc[crisis_mask, "date"]

        if len(calib) == 0 or len(crisis) == 0:
            print(f"[SKIPPED] {ticker}: empty calibration or crisis window")
            continue

        full_stats = describe(df["log_return"].dropna())
        calib_stats = describe(calib)
        crisis_stats = describe(crisis)

        rows.append({
            "ticker": full_name,          # matches forecast_accuracy_comparison.csv
            "short_ticker": ticker,       # e.g. AZN
            "sector": SECTORS.get(ticker, "VERIFY"),
            "calib_start": calib_dates.min().strftime("%Y-%m-%d"),
            "calib_end": calib_dates.max().strftime("%Y-%m-%d"),
            "crisis_start": crisis_dates.min().strftime("%Y-%m-%d"),
            "crisis_end": crisis_dates.max().strftime("%Y-%m-%d"),
            # Full sample (Table 4.2)
            **{f"full_{k}": v for k, v in full_stats.items()},
            # Calibration window
            **{f"calib_{k}": v for k, v in calib_stats.items()},
            # Crisis window
            **{f"crisis_{k}": v for k, v in crisis_stats.items()},
            # Ratio of crisis to calibration volatility (how much risk rose)
            "crisis_over_calib_vol": crisis_stats["std_dev"] / calib_stats["std_dev"],
        })

    desc = pd.DataFrame(rows)
    if desc.empty:
        print("[ERROR] No stocks processed — check that data/processed/ contains "
              "CSVs with 'date' and 'log_return' columns covering the configured windows.")
        return
    desc_path = os.path.join(config.OUTPUT_DIR, "descriptive_statistics.csv")
    desc.to_csv(desc_path, index=False)
    print(f"Descriptive statistics saved to {desc_path}")
    print(desc[["short_ticker", "sector", "calib_start", "calib_end",
               "full_n_obs", "calib_mean", "calib_std_dev",
               "crisis_std_dev", "crisis_over_calib_vol"]].to_string(index=False))

    # ---- Merge with forecast accuracy for research objective 2 ----
    acc_path = os.path.join(config.OUTPUT_DIR, "forecast_accuracy_comparison.csv")
    if os.path.exists(acc_path):
        acc = pd.read_csv(acc_path)
        merged = acc.merge(desc[["ticker", "short_ticker", "sector", "calib_std_dev"]],
                           on="ticker", how="left")
        # Correlation between pre-crisis volatility and forecast errors
        for col in ["garch_rmse", "lstm_rmse", "garch_mae", "lstm_mae"]:
            merged[f"corr_calibvol_{col}"] = merged["calib_std_dev"].corr(merged[col])
        out_path = os.path.join(config.OUTPUT_DIR, "precrisis_vol_vs_accuracy.csv")
        merged.to_csv(out_path, index=False)
        print(f"\nPre-crisis volatility vs accuracy saved to {out_path}")

        r_g = merged["calib_std_dev"].corr(merged["garch_rmse"])
        r_l = merged["calib_std_dev"].corr(merged["lstm_rmse"])
        print(f"Correlation (calibration vol vs GARCH RMSE): {r_g:.3f}")
        print(f"Correlation (calibration vol vs LSTM  RMSE): {r_l:.3f}")

        # High- vs low-volatility halves (exploratory)
        med = merged["calib_std_dev"].median()
        hi = merged[merged["calib_std_dev"] >= med]
        lo = merged[merged["calib_std_dev"] < med]
        print(f"\nHigh pre-crisis volatility group (n={len(hi)}): "
              f"GARCH RMSE {hi['garch_rmse'].mean():.5f}, LSTM RMSE {hi['lstm_rmse'].mean():.5f}")
        print(f"Low  pre-crisis volatility group (n={len(lo)}): "
              f"GARCH RMSE {lo['garch_rmse'].mean():.5f}, LSTM RMSE {lo['lstm_rmse'].mean():.5f}")
        print("GARCH RMSE advantage, high-vol group: "
              f"{(hi['lstm_rmse'] - hi['garch_rmse']).mean():+.5f}")
        print("GARCH RMSE advantage, low-vol group:  "
              f"{(lo['lstm_rmse'] - lo['garch_rmse']).mean():+.5f}")
    else:
        print(f"\n[NOTE] {acc_path} not found — run the main pipeline first "
              f"for the pre-crisis volatility comparison.")


if __name__ == "__main__":
    main()
