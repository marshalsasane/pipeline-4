
import glob
import os
import numpy as np
import pandas as pd
from arch import arch_model

import config


def filtered_sigma(returns_pct, params):
    model = arch_model(returns_pct, vol="Garch", p=1, q=1, dist="normal").fix(params)
    return np.asarray(model.conditional_volatility)


def manual_sigma(returns_pct, params, tau=75):
    e = returns_pct - params["mu"]
    w = 0.94 ** np.arange(tau)
    backcast = (w * e[:tau] ** 2).sum() / w.sum()
    s2 = np.empty(len(returns_pct))
    s2[0] = params["omega"] + params["alpha[1]"] * backcast + params["beta[1]"] * backcast
    for t in range(1, len(returns_pct)):
        s2[t] = params["omega"] + params["alpha[1]"] * e[t - 1] ** 2 + params["beta[1]"] * s2[t - 1]
    return np.sqrt(s2)


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(config.PROCESSED_DIR, "*.csv"))):
        ticker = os.path.splitext(os.path.basename(path))[0].replace("Download Data - STOCK_UK_XLON_", "")
        df = pd.read_csv(path, parse_dates=["date"])
        calib = df[(df["date"] >= config.CALIBRATION_START) & (df["date"] <= config.CALIBRATION_END)]
        crisis = df[(df["date"] >= config.CRISIS_START) & (df["date"] <= config.CRISIS_END)]
        n = len(calib)
        r = pd.concat([calib, crisis])["log_return"].values * 100
        params = arch_model(calib["log_return"].values * 100, vol="Garch", p=1, q=1, dist="normal").fit(disp="off").params

        s_lib = filtered_sigma(r, params)
        s_man = manual_sigma(r, params)
        rel_ref = float(np.max(np.abs(s_lib[n:] - s_man[n:]) / s_man[n:]))

        k = n + 10
        r_alt = r.copy()
        r_alt[k:] = r_alt[k:] * 5 + 7
        s_alt = filtered_sigma(r_alt, params)
        rel_pert = float(np.max(np.abs(s_lib[n:k + 1] - s_alt[n:k + 1]) / s_lib[n:k + 1]))
        moved_after = bool(abs(s_lib[k + 1] - s_alt[k + 1]) > 1e-6)   # sanity: later days should respond

        rows.append({"ticker": ticker, "max_rel_diff_vs_past_only_recursion": rel_ref,
                     "max_rel_change_when_future_altered": rel_pert, "later_days_respond": moved_after})

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    ok = (out["max_rel_diff_vs_past_only_recursion"] < 1e-9).all() and (out["max_rel_change_when_future_altered"] < 1e-9).all()
    print("\nAll crisis-day GARCH forecasts are past-only:", bool(ok))


if __name__ == "__main__":
    main()
