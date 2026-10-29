# GARCH vs LSTM Volatility Pipeline — Setup Guide

A step-by-step guide from installing Python through to running the full
automated pipeline on your 15 stock CSVs. Every script in this folder has
been test-run and works end to end.

---

## Part 1 — Install Python

1. **Download Python.** Go to [python.org/downloads](https://www.python.org/downloads/)
   and install the latest Python 3.11 or 3.12. On Windows, tick **"Add
   python.exe to PATH"** during installation — this is the step people
   most often miss.
2. **Confirm it installed.** Open a terminal (Command Prompt / PowerShell
   on Windows, Terminal on Mac) and run:
   ```
   python --version
   ```
   You should see something like `Python 3.12.x`. If it says "command
   not found", restart your terminal, and if that doesn't fix it,
   reinstall with the PATH box ticked.

## Part 2 — Set up this project

1. **Unzip this pipeline folder** somewhere easy to find, e.g. your
   Desktop or Documents folder.
2. **Open a terminal inside the folder.**
   - Windows: open the folder in File Explorer, click the address bar,
     type `cmd`, press Enter.
   - Mac: right-click the folder → "New Terminal at Folder" (or `cd` into
     it manually).
3. **Create a virtual environment** (keeps this project's packages
   separate from anything else on your machine):
   ```
   python -m venv venv
   ```
4. **Activate it:**
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

   Your terminal prompt should now show `(venv)` at the start of the line.
   You'll need to run this activate command every time you reopen a
   terminal to work on this project.
5. **Install the required packages:**
   ```
   pip install -r requirements.txt
   ```
   This installs pandas, numpy, scipy, scikit-learn, arch, statsmodels,
   and tensorflow. It can take a few minutes, tensorflow is a large
   package.

## Part 3 — Add your data

You have two separate files per stock: a **training set** and a **test
set**. The pipeline keeps these in two separate folders and stitches
each pair back together internally before computing returns (this
matters — the first return of the test period needs the previous day's
price from the training file, so they can't be processed separately).

1. Put your 15 **training** CSVs into `data/raw/train/`.
2. Put your 15 **test** CSVs into `data/raw/test/`.
3. **Use the identical filename for each ticker in both folders**, e.g.
   `TSCO.L.csv` in both `train/` and `test/`. The script matches files
   across the two folders by filename, so a mismatch means that stock
   gets skipped with a warning.
4. Each CSV needs a date column (any name, "Date" is assumed by default)
   and a price column named one of: `Adj Close`, `Close`, `close`,
   `adj_close`, or `Price`. This matches a standard Yahoo Finance export,
   so if that's where your data came from, you don't need to change
   anything. Prices that use a thousands separator (e.g. `"1,234.50"`)
   are handled automatically — no need to clean these yourself.

## Part 4 — Set your study windows

Open `config.py` in any text editor and check these two windows match
your training/test split:

```python
CALIBRATION_START = "2015-01-01"   # = your training set start
CALIBRATION_END = "2022-08-31"     # = your training set end
CRISIS_START = "2022-09-01"        # = your test set start
CRISIS_END = "2022-10-31"          # = your test set end
```

These are already set to match the dates you gave (01/01/2015-31/08/2022
training, 01/09/2022-31/10/2022 test), so you shouldn't need to change
this part unless your split dates differ from stock to stock.

## Part 5 — Run the pipeline

With your virtual environment activated, run each step in order, or run
them all at once.

**Run all four steps automatically:**
```
python run_pipeline.py
```

**Or run them one at a time** (useful if you want to check each
output before moving on, or if LSTM training needs to be restarted
without redoing the earlier steps):
```
python 01_preprocess.py
python 02_garch_model.py
python 03_lstm_model.py
python 04_compare_backtest.py
```

Step 3 (LSTM) is the slowest — expect roughly 10-30 seconds per stock
on a normal laptop, so a few minutes total for 15 stocks. If a
particular stock fails here, the script skips it and keeps going, then
lists which ones failed at the end so you can re-run just those.

## Part 6 — What you get

Everything lands in the `outputs/` folder:

| File | What it is |
|---|---|
| `adf_test_results.csv` | Stationarity test results for every stock |
| `garch_diagnostics.csv` | GARCH fit parameters and ARCH-LM diagnostic per stock |
| `garch_forecasts/{ticker}.csv` | Daily GARCH volatility forecast for the crisis window |
| `lstm_forecasts/{ticker}.csv` | Daily LSTM volatility forecast for the crisis window |
| `forecast_accuracy_comparison.csv` | RMSE and MAPE, GARCH vs LSTM, per stock |
| `var_backtest_kupiec.csv` | VaR violations and Kupiec test result per stock, per model |

These six files are what you'll pull into your Results & Discussion
section — the per-ticker detail backs up your tables, and the printed
summary at the end of step 4 gives you the headline averages.

## Troubleshooting

- **"Add your training CSVs to .../train and test CSVs to .../test"** —
  you haven't added your files yet, or they're in the wrong folder.
- **"No matching test file for: [...]"** or **"No matching training file
  for: [...]"** — the filename in one folder doesn't exactly match the
  other. Check spelling and capitalisation match between `train/` and
  `test/` for that ticker.
- **"Could not find a price column"** — open that CSV and check the exact
  column header, then add it to `PRICE_COLUMN_CANDIDATES` near the top
  of `01_preprocess.py`.
- **"unsupported operand type(s) for /: 'str' and 'str'"** — this means
  a price column still isn't converting to numbers. It's already handled
  for comma thousand-separators (e.g. `"11,034.00"`); if you still see
  this, open the CSV and check for a different formatting quirk, such as
  a currency symbol or a non-breaking space.
- **A stock skipped with "Not enough data in calibration/crisis window"**
  — that stock's CSV doesn't fully cover your configured date range.
  Check its date range against `config.py`.
- **tensorflow install fails** — make sure you're on Python 3.9-3.12;
  tensorflow doesn't yet support the very latest Python releases on all
  platforms.
