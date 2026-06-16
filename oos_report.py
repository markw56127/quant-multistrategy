"""
True out-of-sample report: development period vs 2025+ for each sleeve and
for the combined risk-parity book.

Usage (after rerunning factor_model and earnings_model with the extended
end_date in their configs):

    python oos_report.py                      # split at 2025-01-01
    python oos_report.py --split 2022-01-01   # custom split (mechanics check)

Methodology notes:
  - Monthly net returns derived from each sleeve's capital curve.
  - Combined book = expanding-window inverse-vol risk parity. Weights at
    month t use only months < t, so the OOS segment of the combined book is
    uncontaminated: the weights entering 2025 were estimated on development
    data only and update as OOS months accrue, exactly as live trading would.
  - READ THE ERROR BARS: ~17 OOS months can only distinguish "roughly
    behaving as expected" from "clearly broken". The standard error of an
    annualized Sharpe over T years is roughly sqrt(1/T): for 1.4 years
    that's ±0.85. Do not celebrate or panic over the point estimate.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
STRATS = {
    "factor_vq": ROOT / "factor_model"   / "results" / "backtest.csv",
    "pead":      ROOT / "earnings_model" / "results" / "backtest.csv",
}
ANN = 12


def monthly_net(path: Path) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    eq = df["capital"].resample("ME").last().ffill()
    return eq.pct_change().dropna()


def seg_stats(r: pd.Series) -> dict:
    if len(r) < 2 or r.std() == 0:
        return {"months": len(r), "total": np.nan, "sharpe": np.nan,
                "vol": np.nan, "maxdd": np.nan}
    eq = (1 + r).cumprod()
    return {
        "months": len(r),
        "total":  eq.iloc[-1] - 1,
        "sharpe": r.mean() / r.std() * np.sqrt(ANN),
        "vol":    r.std() * np.sqrt(ANN),
        "maxdd":  (eq / eq.cummax() - 1).min(),
    }


def fmt(s: dict) -> str:
    if np.isnan(s.get("sharpe", np.nan)):
        return f"{s['months']:>3} mo   (insufficient data)"
    return (f"{s['months']:>3} mo   total={s['total']:>+7.1%}   "
            f"Sharpe={s['sharpe']:>+5.2f}   vol={s['vol']:>5.1%}   "
            f"maxDD={s['maxdd']:>+6.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="2025-01-01",
                    help="OOS start date (default: 2025-01-01)")
    args = ap.parse_args()
    split = pd.Timestamp(args.split)

    rets = {}
    for name, path in STRATS.items():
        if not path.exists():
            print(f"!! {name}: {path} missing — run that model first")
            continue
        rets[name] = monthly_net(path)
    if not rets:
        return
    panel = pd.DataFrame(rets).dropna()

    last = panel.index.max()
    n_oos = int((panel.index >= split).sum())
    print(f"Data: {panel.index.min().date()} → {last.date()}   "
          f"split @ {split.date()}   ({n_oos} OOS months)")
    if n_oos == 0:
        print("\n!! No OOS months found. Re-run the models AFTER extending "
              "end_date in their configs (factor_model & earnings_model are "
              "set to 2026-06-30 now), then run this report again.")
        return
    if n_oos < 6:
        print("!! Very short OOS window — treat everything below as anecdote.")

    print(f"\n{'─'*72}\nPer sleeve (net, monthly grid)\n{'─'*72}")
    for name in panel.columns:
        dev = panel.loc[panel.index <  split, name]
        oos = panel.loc[panel.index >= split, name]
        print(f"{name}")
        print(f"   dev  {fmt(seg_stats(dev))}")
        print(f"   OOS  {fmt(seg_stats(oos))}")

    # Combined: expanding inverse-vol risk parity, strictly past-only weights
    vol = panel.expanding(min_periods=12).std().shift(1)
    w = (1.0 / vol).div((1.0 / vol).sum(axis=1), axis=0)
    w = w.fillna(1.0 / panel.shape[1])
    book = (panel * w).sum(axis=1)

    print(f"\n{'─'*72}\nCombined book (expanding risk parity, past-only weights)\n{'─'*72}")
    print(f"   dev  {fmt(seg_stats(book[book.index <  split]))}")
    print(f"   OOS  {fmt(seg_stats(book[book.index >= split]))}")
    w_now = w.iloc[-1].round(2).to_dict()
    print(f"   current weights: {w_now}")

    yrs = n_oos / 12
    se = np.sqrt(1.0 / max(yrs, 1e-9))
    print(f"\nError-bar reminder: with {n_oos} OOS months, the Sharpe point "
          f"estimate has SE ≈ ±{se:.2f}.\nThe question this answers is "
          f"'is the book roughly behaving?' — not 'is the Sharpe 0.9 or 0.7?'")


if __name__ == "__main__":
    main()
