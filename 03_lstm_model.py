"""
STEP 3 — LSTM modelling

Trains ONE fixed LSTM architecture (set in config.py) on the calibration
window for every stock, then produces one-step-ahead volatility forecasts
across the crisis window using the real lagged values each step (walk-forward),
so it's forecasting on the same footing as the GARCH model.

This is the slowest step in the pipeline — expect it to take a while
across all 15 stocks. Run after 02_garch_model.py: python 03_lstm_model.py
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

import config

tf.random.set_seed(42)
np.random.seed(42)


def make_windows(series, lookback):
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def build_model(lookback):
    model = Sequential([
        Input(shape=(lookback, 1)),
        LSTM(config.LSTM_UNITS),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def fit_and_forecast(df, ticker):
    df["date"] = pd.to_datetime(df["date"])

    calib = df[(df["date"] >= config.CALIBRATION_START) & (df["date"] <= config.CALIBRATION_END)]
    crisis = df[(df["date"] >= config.CRISIS_START) & (df["date"] <= config.CRISIS_END)]

    if len(calib) < config.LSTM_LOOKBACK + 20 or len(crisis) == 0:
        raise ValueError(f"Not enough data in calibration ({len(calib)}) or crisis ({len(crisis)}) window")

    # Volatility proxy: absolute log return (same proxy used for the GARCH comparison)
    vol_proxy_calib = calib["log_return"].abs().values

    scaler = MinMaxScaler()
    scaled_calib = scaler.fit_transform(vol_proxy_calib.reshape(-1, 1)).flatten()

    X_train, y_train = make_windows(scaled_calib, config.LSTM_LOOKBACK)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    model = build_model(config.LSTM_LOOKBACK)
    early_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        epochs=config.LSTM_EPOCHS,
        batch_size=config.LSTM_BATCH_SIZE,
        verbose=0,
        callbacks=[early_stop],
    )

    # Walk-forward forecast across the crisis window: the lookback window
    # always uses the REAL preceding values, matching how GARCH is evaluated.
    full_series = pd.concat([calib, crisis])["log_return"].abs().values
    full_scaled = scaler.transform(full_series.reshape(-1, 1)).flatten()

    crisis_start_idx = len(calib)
    forecasts_scaled = []
    for i in range(crisis_start_idx, len(full_scaled)):
        window = full_scaled[i - config.LSTM_LOOKBACK:i].reshape(1, config.LSTM_LOOKBACK, 1)
        pred = model.predict(window, verbose=0)[0, 0]
        forecasts_scaled.append(pred)

    forecasts = scaler.inverse_transform(np.array(forecasts_scaled).reshape(-1, 1)).flatten()

    result = crisis[["date"]].copy()
    result["lstm_forecast_vol"] = forecasts
    result["realized_abs_return"] = crisis["log_return"].abs().values
    result["actual_return"] = crisis["log_return"].values

    return result


def main():
    os.makedirs(os.path.join(config.OUTPUT_DIR, "lstm_forecasts"), exist_ok=True)
    processed_files = sorted(glob.glob(os.path.join(config.PROCESSED_DIR, "*.csv")))

    failures = []

    for path in processed_files:
        ticker = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path)

        try:
            result = fit_and_forecast(df, ticker)
        except Exception as e:
            print(f"[SKIPPED] {ticker}: {e}")
            failures.append(ticker)
            continue

        out_path = os.path.join(config.OUTPUT_DIR, "lstm_forecasts", f"{ticker}.csv")
        result.to_csv(out_path, index=False)
        print(f"[OK] {ticker}: LSTM trained, forecast saved -> {out_path}")

    if failures:
        print(f"\nStocks that failed and need a re-run or inspection: {failures}")
    else:
        print("\nAll stocks trained successfully.")


if __name__ == "__main__":
    main()
