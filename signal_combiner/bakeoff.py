"""
Linear-vs-XGBoost signal-combination bake-off.

Question under test: does a more flexible model combine the five factor signals
(value, momentum, quality, low_vol, size) into a better cross-sectional return
forecast than the naive equal-weight composite the factor sleeve already trades?

Four combiners, identical features, identical evaluation:
  - EW         : equal-weight composite (incumbent) — mean of the 5 z-scores, no fit
  - OLS        : pooled linear regression
  - Ridge      : L2-regularized linear (RidgeCV picks alpha on the train fold only)
  - XGBoost    : gradient-boosted trees (shallow, subsampled, L2 — a fair but
                 regularized shot at nonlinear factor interactions)

Evaluation — PURGED + EMBARGOED expanding walk-forward (López de Prado):
  For each test month t (after min_train_months of history), train on all months
  up to t minus an embargo buffer, so a training label (a forward return realized
  over [m, m+1]) cannot overlap the test month. Predict the t cross-section, then:
    - rank IC : Spearman(prediction, realized fwd return) for month t
    - L/S     : long top-decile / short bottom-decile by prediction, equal weight
  Results split dev (<2025) vs true OOS (>=2025), the same split as every sleeve.

The honest prior: at equity-factor signal-to-noise, variance dominates, so the
regularized linear model should roughly tie the composite and XGBoost should NOT
beat it out of sample. Demonstrating that cleanly is the result.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, RidgeCV
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_panel import build_panel  # noqa: E402

warnings.filterwarnings("ignore")
CFG = yaml.safe_load(open(Path(__file__).resolve().parent / "config.yaml"))["bakeoff"]
FEATS = CFG["features"]
ANN = 12


def _winsorize(s: pd.Series, p: float) -> pd.Series:
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)


def _prep(panel: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectionally demean + winsorize the target; mark the EW composite."""
    df = panel.copy()
    df["ew"] = df[FEATS].mean(axis=1)                       # incumbent, skipna mean
    # target: cross-sectionally demeaned, winsorized forward return
    df["y"] = df.groupby("date")["fwd_ret"].transform(
        lambda s: _winsorize(s - s.mean(), CFG["winsor"]))
    df[FEATS] = df[FEATS].fillna(0.0)                       # missing factor -> neutral
    return df


def _ls_return(pred: np.ndarray, fwd: np.ndarray, q: float) -> float:
    """Equal-weight long-short return: top-q minus bottom-q by prediction."""
    n = len(pred)
    k = max(int(n * q), 1)
    order = np.argsort(pred)
    short = fwd[order[:k]].mean()
    long_ = fwd[order[-k:]].mean()
    return float(long_ - short)


def run_bakeoff():
    panel = _prep(build_panel())
    months = np.array(sorted(panel["date"].unique()))
    by_month = {m: panel[panel["date"] == m] for m in months}
    min_train, emb, q = CFG["min_train_months"], CFG["embargo_months"], CFG["decile"]
    oos_ts = pd.Timestamp(CFG["oos_start"])

    models = ["ew", "ols", "ridge", "xgb"]
    rec = {m: [] for m in models}    # list of (date, ic, ls_ret)

    for i in range(min_train + emb, len(months)):
        test_m = months[i]
        train_months = months[: i - emb]                   # purge: drop embargo buffer
        if len(train_months) < min_train:
            continue
        tr = panel[panel["date"].isin(train_months)]
        te = by_month[test_m]
        Xtr, ytr = tr[FEATS].values, tr["y"].values
        Xte = te[FEATS].values
        fwd = te["fwd_ret"].values

        preds = {"ew": te["ew"].values}
        preds["ols"] = LinearRegression().fit(Xtr, ytr).predict(Xte)
        preds["ridge"] = RidgeCV(alphas=[0.1, 1, 10, 100, 1000]).fit(Xtr, ytr).predict(Xte)
        preds["xgb"] = XGBRegressor(
            **CFG["xgb"], objective="reg:squarederror", n_jobs=4, verbosity=0
        ).fit(Xtr, ytr).predict(Xte)

        for m in models:
            ic = spearmanr(preds[m], fwd).correlation
            rec[m].append((test_m, ic, _ls_return(preds[m], fwd, q)))

    # ── Aggregate, split dev vs OOS ──
    def stats(df: pd.DataFrame, mask) -> dict:
        d = df[mask]
        ic, ls = d["ic"].dropna(), d["ls"]
        n = len(ic)
        return {
            "n": n,
            "ic": ic.mean(),
            "ic_t": ic.mean() / (ic.std() / np.sqrt(n)) if n > 1 else np.nan,
            "ls_sharpe": ls.mean() / ls.std() * np.sqrt(ANN) if ls.std() > 0 else 0.0,
            "ls_total": (1 + ls).prod() - 1,
        }

    print(f"\n{'='*86}")
    print("Linear-vs-XGBoost bake-off — purged/embargoed walk-forward")
    print(f"features={FEATS}  min_train={min_train}mo  embargo={emb}mo  "
          f"L/S={q:.0%}  split@{oos_ts.date()}")
    print(f"{'='*86}")
    hdr = f"{'model':<8} {'segment':<5} {'n':>4} {'meanIC':>8} {'IC t':>7} {'L/S Sharpe':>11} {'L/S total':>10}"
    table = {}
    for m in models:
        df = pd.DataFrame(rec[m], columns=["date", "ic", "ls"])
        table[m] = {"dev": stats(df, df["date"] < oos_ts),
                    "oos": stats(df, df["date"] >= oos_ts)}
    for seg in ["dev", "oos"]:
        print(f"\n── {seg.upper()} ──")
        print(hdr)
        for m in models:
            s = table[m][seg]
            print(f"{m:<8} {seg:<5} {s['n']:>4} {s['ic']:>+8.4f} {s['ic_t']:>+7.2f} "
                  f"{s['ls_sharpe']:>+11.2f} {s['ls_total']:>+10.1%}")

    # ── Verdict ──
    print(f"\n{'─'*86}")
    base = table["ew"]
    for seg in ["dev", "oos"]:
        best = max(models, key=lambda m: table[m][seg]["ls_sharpe"])
        lift = table[best][seg]["ls_sharpe"] - base[seg]["ls_sharpe"]
        print(f"{seg.upper():<4} best L/S Sharpe: {best} ({table[best][seg]['ls_sharpe']:+.2f})  "
              f"vs EW composite ({base[seg]['ls_sharpe']:+.2f})  "
              f"→ lift {lift:+.2f}")
    xgb_oos = table["xgb"]["oos"]["ls_sharpe"]
    ew_oos = table["ew"]["oos"]["ls_sharpe"]
    print(f"\nKey test — XGBoost vs EW composite, true OOS: "
          f"{xgb_oos:+.2f} vs {ew_oos:+.2f}  "
          f"({'XGB wins' if xgb_oos > ew_oos else 'XGB does NOT beat the naive composite'})")

    out = Path(__file__).resolve().parent / "results" / "bakeoff.csv"
    pd.concat({m: pd.DataFrame(rec[m], columns=["date", "ic", "ls"]).set_index("date")
               for m in models}, axis=1).to_csv(out)
    print(f"\nPer-month detail → {out}")


if __name__ == "__main__":
    run_bakeoff()
