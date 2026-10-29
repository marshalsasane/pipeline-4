
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PREFIX = "Download Data - STOCK_UK_XLON_"
K = np.sqrt(np.pi / 2)
ALPHA = 0.05


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
def load_settings(root):
    cfg = {
        "out_dir": root / "outputs",
        "proc_dir": root / "data" / "processed",
        "cal_start": "2015-01-01",
        "cal_end": "2022-08-31",
    }
    sys.path.insert(0, str(root))
    try:
        import config  # noqa
        if hasattr(config, "OUTPUT_DIR"):
            cfg["out_dir"] = root / Path(str(config.OUTPUT_DIR))
        if hasattr(config, "PROCESSED_DIR"):
            cfg["proc_dir"] = root / Path(str(config.PROCESSED_DIR))
        if hasattr(config, "CALIBRATION_START"):
            cfg["cal_start"] = str(config.CALIBRATION_START)
        if hasattr(config, "CALIBRATION_END"):
            cfg["cal_end"] = str(config.CALIBRATION_END)
    except Exception:
        pass
    return cfg


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def dm_test(d, lags=0):
    """DM test on a loss differential series d (h = 1).

    With h = 1 there are no autocovariance terms in the long-run variance, so
    lags=0 uses the plain variance. lags>0 adds a Bartlett (Newey-West) correction.
    The Harvey-Leybourne-Newbold small-sample factor and a t(T-1) reference are applied.
    Returns (hln_statistic, p_value, T).
    """
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    T = len(d)
    if T < 3:
        return np.nan, np.nan, T
    dbar = d.mean()
    dc = d - dbar
    lrv = np.mean(dc ** 2)
    for k in range(1, lags + 1):
        gk = np.sum(dc[k:] * dc[:-k]) / T
        lrv += 2 * (1 - k / (lags + 1)) * gk
    if lrv <= 0:
        return np.nan, np.nan, T
    dm = dbar / np.sqrt(lrv / T)
    h = 1
    hln = dm * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    p = 2 * stats.t.sf(abs(hln), df=T - 1)
    return float(hln), float(p), T


def holm(pvals):
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def safe_wilcoxon(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 5 or np.all(x == 0):
        return np.nan
    try:
        return float(stats.wilcoxon(x).pvalue)
    except Exception:
        return np.nan


# --------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------
def short(name):
    return name.replace(PREFIX, "")


def calibration_volatility(path, start, end):
    """Standard deviation of daily log returns inside the calibration window."""
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    date_col = next((cols[c] for c in ("date", "datetime", "time") if c in cols), df.columns[0])
    ret_col = next((cols[c] for c in ("log_return", "log_returns", "logret", "return", "returns", "ret")
                    if c in cols), None)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(date_col)
    if ret_col is not None:
        r = pd.to_numeric(df[ret_col].astype(str).str.replace(",", ""), errors="coerce")
    else:
        pcol = next((cols[c] for c in ("close", "price", "adj close", "adj_close", "last") if c in cols), None)
        if pcol is None:
            raise ValueError(f"No return or price column found in {path.name}: {list(df.columns)}")
        px = pd.to_numeric(df[pcol].astype(str).str.replace(",", ""), errors="coerce")
        r = np.log(px).diff()
    mask = (df[date_col] >= pd.Timestamp(start)) & (df[date_col] <= pd.Timestamp(end))
    r = r[mask].dropna()
    return float(r.std(ddof=1)), int(len(r))


# --------------------------------------------------------------------------
# Part 1: DM tests
# --------------------------------------------------------------------------
def run_dm(garch_dir, lstm_dir):
    rows, sq_series, abs_series = [], {}, {}
    for gf in sorted(garch_dir.glob("*.csv")):
        lf = lstm_dir / gf.name
        if not lf.exists():
            print(f"  skipped {short(gf.stem)}: no LSTM file")
            continue
        g = pd.read_csv(gf, parse_dates=["date"])
        l = pd.read_csv(lf, parse_dates=["date"])
        m = g.merge(l, on="date", suffixes=("", "_l")).sort_values("date")
        if "realized_abs_return_l" in m and not np.allclose(
                m["realized_abs_return"], m["realized_abs_return_l"], equal_nan=True):
            print(f"  WARNING {short(gf.stem)}: realised series differ between GARCH and LSTM files")
        proxy = m["realized_abs_return"].values * K
        gs = m["garch_forecast_vol"].values
        ls = m["lstm_forecast_vol"].values * K
        eg, el = gs - proxy, ls - proxy
        d_sq = eg ** 2 - el ** 2
        d_abs = np.abs(eg) - np.abs(el)
        t = short(gf.stem)
        sq_series[t] = pd.Series(d_sq, index=m["date"].values)
        abs_series[t] = pd.Series(d_abs, index=m["date"].values)
        dm_sq, p_sq, T = dm_test(d_sq)
        dm_ab, p_ab, _ = dm_test(d_abs)
        rows.append({
            "ticker": t, "n_days": T,
            "garch_rmse": np.sqrt(np.mean(eg ** 2)), "lstm_rmse": np.sqrt(np.mean(el ** 2)),
            "garch_mae": np.mean(np.abs(eg)), "lstm_mae": np.mean(np.abs(el)),
            "dm_sq": dm_sq, "p_sq": p_sq, "dm_abs": dm_ab, "p_abs": p_ab,
        })
    res = pd.DataFrame(rows)
    res["p_sq_holm"] = holm(res["p_sq"].fillna(1.0))
    res["p_abs_holm"] = holm(res["p_abs"].fillna(1.0))

    def verdict(dm, p):
        if np.isnan(p) or p >= ALPHA:
            return "no sig. difference"
        return "LSTM better" if dm > 0 else "GARCH better"

    res["verdict_sq"] = [verdict(a, b) for a, b in zip(res["dm_sq"], res["p_sq"])]
    res["verdict_abs"] = [verdict(a, b) for a, b in zip(res["dm_abs"], res["p_abs"])]
    return res, pd.DataFrame(sq_series), pd.DataFrame(abs_series)


def pooled_dm(res, sq_panel, abs_panel):
    rows = []
    for label, panel, mean_col in (("Squared error", sq_panel, "sq"), ("Absolute error", abs_panel, "abs")):
        daily = panel.mean(axis=1)  # cross-sectional average differential per day
        for lags in (0, 3):
            dm, p, T = dm_test(daily.values, lags=lags)
            rows.append({"test": f"DM on daily cross-sectional mean, {label}, lag {lags}",
                         "statistic": dm, "p_value": p, "n": T,
                         "reading": "LSTM better" if dm > 0 else "GARCH better"})
        per_stock = panel.mean(axis=0)
        rows.append({"test": f"Wilcoxon across stocks, mean differential, {label}",
                     "statistic": np.nan, "p_value": safe_wilcoxon(per_stock.values), "n": len(per_stock),
                     "reading": f"LSTM lower loss in {(per_stock > 0).sum()} of {len(per_stock)} stocks"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Part 2: median split
# --------------------------------------------------------------------------
def median_split(res, proc_dir, cal_start, cal_end):
    vols = {}
    for t in res["ticker"]:
        cand = [proc_dir / f"{PREFIX}{t}.csv", proc_dir / f"{t}.csv"]
        path = next((c for c in cand if c.exists()), None)
        if path is None:
            print(f"  no processed file for {t}")
            continue
        v, n = calibration_volatility(path, cal_start, cal_end)
        vols[t] = v
    if len(vols) < len(res):
        raise SystemExit("Processed files missing for some tickers, so the median split cannot be built. "
                         "Check PROCESSED_DIR in config.py.")
    df = res.copy()
    df["pre_vol_daily"] = df["ticker"].map(vols)
    df["pre_vol_annual"] = df["pre_vol_daily"] * np.sqrt(252)
    med = df["pre_vol_daily"].median()
    df["group"] = np.where(df["pre_vol_daily"] > med, "High pre-crisis volatility", "Low pre-crisis volatility")
    df["rel_rmse_pct"] = (df["lstm_rmse"] / df["garch_rmse"] - 1) * 100
    df["rel_mae_pct"] = (df["lstm_mae"] / df["garch_mae"] - 1) * 100

    out = []
    for grp in ("Low pre-crisis volatility", "High pre-crisis volatility"):
        s = df[df["group"] == grp]
        out.append({
            "group": grp, "n_stocks": len(s), "tickers": ", ".join(s["ticker"]),
            "mean_pre_vol_daily": s["pre_vol_daily"].mean(),
            "mean_garch_rmse": s["garch_rmse"].mean(), "mean_lstm_rmse": s["lstm_rmse"].mean(),
            "mean_rel_rmse_pct": s["rel_rmse_pct"].mean(),
            "mean_garch_mae": s["garch_mae"].mean(), "mean_lstm_mae": s["lstm_mae"].mean(),
            "mean_rel_mae_pct": s["rel_mae_pct"].mean(),
            "lstm_wins_rmse": f"{(s['lstm_rmse'] < s['garch_rmse']).sum()}/{len(s)}",
            "lstm_wins_mae": f"{(s['lstm_mae'] < s['garch_mae']).sum()}/{len(s)}",
            "wilcoxon_p_rmse": safe_wilcoxon(s["garch_rmse"] - s["lstm_rmse"]),
            "dm_sig_lstm_better": int((s["verdict_sq"] == "LSTM better").sum()),
            "dm_sig_garch_better": int((s["verdict_sq"] == "GARCH better").sum()),
        })
    table = pd.DataFrame(out)

    hi = df[df["group"].str.startswith("High")]["rel_rmse_pct"]
    lo = df[df["group"].str.startswith("Low")]["rel_rmse_pct"]
    mw = stats.mannwhitneyu(hi, lo, alternative="two-sided")
    rho, p_rho = stats.spearmanr(df["pre_vol_daily"], df["rel_rmse_pct"])
    tests = pd.DataFrame([
        {"test": "Mann-Whitney U, relative RMSE difference, High vs Low group",
         "statistic": mw.statistic, "p_value": mw.pvalue},
        {"test": "Spearman rho, pre-crisis volatility vs relative RMSE difference (all stocks)",
         "statistic": rho, "p_value": p_rho},
    ])
    return df, table, tests, med


# --------------------------------------------------------------------------
# Interpretation text (numbers filled in from the results)
# --------------------------------------------------------------------------
def interpret(res, pooled, table, tests, med):
    n = len(res)
    L = []
    L.append("DIEBOLD-MARIANO TESTS")
    for key, name in (("sq", "squared error"), ("abs", "absolute error")):
        v = res[f"verdict_{key}"]
        nl, ng = int((v == "LSTM better").sum()), int((v == "GARCH better").sum())
        hl = int((res[f"p_{key}_holm"] < ALPHA).sum())
        L.append(f"- Under {name} loss, the LSTM is significantly more accurate than GARCH(1,1) for {nl} of {n} stocks "
                 f"and GARCH is significantly more accurate for {ng} (two-sided, 5% level, HLN-corrected, t with T-1 df). "
                 f"After Holm adjustment for {n} tests, {hl} stock(s) remain significant.")
    p0 = pooled.iloc[0]
    p2 = pooled.iloc[3]
    L.append(f"- On the daily cross-sectional average differential (which allows for common crisis shocks), the squared-error "
             f"DM statistic is {p0['statistic']:.2f} (p = {p0['p_value']:.3f}) and the absolute-error statistic is "
             f"{p2['statistic']:.2f} (p = {p2['p_value']:.3f}).")
    sig_any = (p0["p_value"] < ALPHA) or (p2["p_value"] < ALPHA)
    if sig_any:
        L.append("- At least one pooled test is significant, so there is some evidence of a systematic accuracy difference.")
    else:
        L.append("- Neither pooled test is significant, so the evidence does not support a systematic accuracy advantage "
                 "for either model over the crisis window.")
    L.append("- Caveat: each stock has about 42 daily observations, so DM tests have low power. A non-significant result "
             "means the data cannot separate the models, not that the models are equally accurate. Treat the findings as indicative.")
    L.append("")
    L.append("MEDIAN SPLIT (RESEARCH QUESTION 2)")
    lo = table[table["group"].str.startswith("Low")].iloc[0]
    hi = table[table["group"].str.startswith("High")].iloc[0]
    L.append(f"- Stocks were split at the median pre-crisis daily volatility ({med:.4f}) using calibration-window log returns. "
             f"The Low group has {int(lo['n_stocks'])} stocks and the High group {int(hi['n_stocks'])} (the median stock is placed in Low).")
    L.append(f"- Mean LSTM RMSE relative to GARCH: {lo['mean_rel_rmse_pct']:+.2f}% in the Low group and "
             f"{hi['mean_rel_rmse_pct']:+.2f}% in the High group (negative means LSTM is more accurate). "
             f"LSTM wins on RMSE for {lo['lstm_wins_rmse']} Low and {hi['lstm_wins_rmse']} High stocks.")
    mw_p, rho_row = tests.iloc[0]["p_value"], tests.iloc[1]
    if mw_p < ALPHA or rho_row["p_value"] < ALPHA:
        better = "High" if hi["mean_rel_rmse_pct"] < lo["mean_rel_rmse_pct"] else "Low"
        L.append(f"- The relative advantage differs between groups (Mann-Whitney p = {mw_p:.3f}; Spearman rho = "
                 f"{rho_row['statistic']:.2f}, p = {rho_row['p_value']:.3f}). The LSTM does comparatively better in the {better} "
                 f"volatility group.")
    else:
        L.append(f"- The difference between groups is not statistically reliable (Mann-Whitney p = {mw_p:.3f}; Spearman rho = "
                 f"{rho_row['statistic']:.2f}, p = {rho_row['p_value']:.3f}). Pre-crisis volatility does not "
                 f"measurably change the relative accuracy of the two models in this sample.")
    L.append("- Caveat: groups of 7 to 8 stocks give very little power, and the split uses one volatility measure only. "
             "Describe this as exploratory.")
    return "\n".join(L)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.cwd()))
    args = ap.parse_args()
    root = Path(args.root).resolve()
    cfg = load_settings(root)
    out = cfg["out_dir"]
    garch_dir, lstm_dir = out / "garch_forecasts", out / "lstm_forecasts"
    if not garch_dir.exists() or not lstm_dir.exists():
        raise SystemExit(f"Forecast folders not found under {out}. Run 02 and 03 first, or pass --root.")

    print("Running Diebold-Mariano tests ...")
    res, sq_panel, abs_panel = run_dm(garch_dir, lstm_dir)
    pooled = pooled_dm(res, sq_panel, abs_panel)
    print("Building median split ...")
    full, table, tests, med = median_split(res, cfg["proc_dir"], cfg["cal_start"], cfg["cal_end"])

    res_out = full[["ticker", "group", "pre_vol_daily", "n_days", "garch_rmse", "lstm_rmse", "garch_mae", "lstm_mae",
                    "dm_sq", "p_sq", "p_sq_holm", "verdict_sq", "dm_abs", "p_abs", "p_abs_holm", "verdict_abs"]]
    res_out.to_csv(out / "dm_tests_per_stock.csv", index=False)
    pooled.to_csv(out / "dm_tests_pooled.csv", index=False)
    table.to_csv(out / "median_split_table.csv", index=False)
    tests.to_csv(out / "median_split_tests.csv", index=False)
    text = interpret(res, pooled, table, tests, med)
    (out / "dm_median_split_interpretation.txt").write_text(text, encoding="utf-8")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\n=== DM tests per stock (positive DM = LSTM more accurate) ===")
    print(res_out.round(4).to_string(index=False))
    print("\n=== Pooled tests ===")
    print(pooled.round(4).to_string(index=False))
    print("\n=== Median split ===")
    print(table.drop(columns=["tickers"]).round(4).T.to_string())
    print("\n=== Between-group tests ===")
    print(tests.round(4).to_string(index=False))
    print("\n" + text)
    print(f"\nSaved 5 files to {out}")


if __name__ == "__main__":
    main()
