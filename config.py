
import os

# ---- Folders ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "raw", "train")   # put your 15 TRAINING CSVs here
TEST_DIR = os.path.join(BASE_DIR, "data", "raw", "test")     # put your 15 TEST CSVs here
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ---- Study windows ----
# Calibration window: used to FIT both models (= your training set range)
CALIBRATION_START = "2015-01-01"
CALIBRATION_END = "2022-08-31"

# Crisis window: used to TEST/forecast and compare against (= your test set range)
CRISIS_START = "2022-09-01"
CRISIS_END = "2022-10-31"

# ---- LSTM settings (kept fixed across all 15 stocks on purpose) ----
LSTM_LOOKBACK = 10        # how many past days the LSTM sees to predict the next
LSTM_UNITS = 32
LSTM_EPOCHS = 30
LSTM_BATCH_SIZE = 16

# ---- VaR / Kupiec settings ----
VAR_CONFIDENCE = 0.99     # 99% VaR
