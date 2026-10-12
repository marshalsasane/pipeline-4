"""
STEP 1 — Preprocessing

Reads each ticker's TRAINING file from data/raw/train/ and its matching
TEST file from data/raw/test/, stitches them into one continuous series
(the test file's first return needs the training file's last price),
works out the price column automatically, computes log returns, runs the
ADF stationarity test, and saves a clean version of each stock to
data/processed/.

Run this first: python 01_preprocess.py
"""

import os
import glob
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

import config

PRICE_COLUMN_CANDIDATES = ["Adj Close", "Close", "close", "adj_close", "Price"]


def find_price_column(df):
    for col in PRICE_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find a price column. Columns found: {list(df.columns)}. "
        f"Add your column name to PRICE_COLUMN_CANDIDATES in this script."
    )


def load_one_file(csv_path):
    df = pd.read_csv(csv_path)

    # Find the date column (Yahoo Finance exports usually call it "Date")
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])

    price_col = find_price_column(df)
    df = df[[date_col, price_col]].rename(columns={date_col: "date", price_col: "price"})

    # Some exports write prices over 1,000 with a thousands separator,
    # e.g. "1,234.50", which pandas reads as TEXT rather than a number.
    # Strip commas (and any stray whitespace) before converting to numeric,
    # so this works whether or not the source file uses separators.
    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df.dropna(subset=["price"])
    return df


def load_and_clean(train_path, test_path):
    train_df = load_one_file(train_path)
    test_df = load_one_file(test_path)

    # Stitch training + test into one continuous series BEFORE computing
    # returns, so the test period's first return uses the real preceding
    # price from the training file rather than coming out as NaN.
    df = pd.concat([train_df, test_df], ignore_index=True)
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)

    df["log_return"] = np.log(df["price"] / df["price"].shift(1))
    df = df.dropna(subset=["log_return"]).reset_index(drop=True)

    return df


def run_adf_test(series):
    result = adfuller(series.dropna())
    return {
        "adf_statistic": result[0],
        "p_value": result[1],
        "is_stationary_at_5pct": result[1] < 0.05,
    }


def main():
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    train_files = {os.path.splitext(os.path.basename(p))[0]: p
                   for p in glob.glob(os.path.join(config.TRAIN_DIR, "*.csv"))}
    test_files = {os.path.splitext(os.path.basename(p))[0]: p
                  for p in glob.glob(os.path.join(config.TEST_DIR, "*.csv"))}

    if not train_files or not test_files:
        print(f"Add your training CSVs to {config.TRAIN_DIR} and test CSVs to {config.TEST_DIR} first.")
        return

    tickers = sorted(set(train_files) & set(test_files))
    missing_test = sorted(set(train_files) - set(test_files))
    missing_train = sorted(set(test_files) - set(train_files))
    if missing_test:
        print(f"[WARNING] No matching test file for: {missing_test} — these will be skipped")
    if missing_train:
        print(f"[WARNING] No matching training file for: {missing_train} — these will be skipped")

    adf_summary = []

    for ticker in tickers:
        try:
            df = load_and_clean(train_files[ticker], test_files[ticker])
        except Exception as e:
            print(f"[SKIPPED] {ticker}: {e}")
            continue

        adf_result = run_adf_test(df["log_return"])
        adf_result["ticker"] = ticker
        adf_result["n_observations"] = len(df)
        adf_summary.append(adf_result)

        out_path = os.path.join(config.PROCESSED_DIR, f"{ticker}.csv")
        df.to_csv(out_path, index=False)

        date_min = df["date"].min().strftime("%Y-%m-%d")
        date_max = df["date"].max().strftime("%Y-%m-%d")
        covers_crisis = date_max >= config.CRISIS_START
        flag = "" if covers_crisis else "  <-- does NOT reach the crisis window, check this stock's test file"
        print(f"[OK] {ticker}: {len(df)} rows, {date_min} to {date_max} -> {out_path}{flag}")

    summary_df = pd.DataFrame(adf_summary)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(config.OUTPUT_DIR, "adf_test_results.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nADF test summary saved to {summary_path}")
    print(summary_df[["ticker", "adf_statistic", "p_value", "is_stationary_at_5pct"]])


if __name__ == "__main__":
    main()
