"""
The pre-registered screen (README.md): C1–C5 by Fama-MacBeth, C6 by event study, one
Holm–Bonferroni family at a 5% familywise error rate, plus an incrementality condition.
DEV ONLY: a rebalance counts if its forward-return window ends before 2025-01-01.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
from characteristics import CANDS, OUT as CHARS, PANEL, PRICES  # noqa: E402
import events  # noqa: E402

DEV_END = pd.Timestamp("2025-01-01")
SIGN = {"asset_growth": -1, "net_issuance": -1, "accruals": -1, "rd_intensity": +1, "st_reversal": -1}
EXISTING = ["value", "momentum", "quality", "low_vol", "size"]
FWER, TC = 0.05, 0.001
SUB = {"2016-19": ("2016-01-01", "2020-01-01"), "2020-24": ("2020-01-01", "2025-01-01")}


def dev_panel():
    panel = pd.read_parquet(PANEL)
    days = pd.read_parquet(PRICES).index
    pos = days.get_indexer(panel.date)
    end = pd.Series(pd.NaT, index=panel.index)
    ok = (pos >= 0) & (pos + 21 < len(days))
    end[ok] = days[pos[ok] + 21]
    panel = panel[end < DEV_END].copy()
    ch = pd.read_parquet(CHARS)[["date", "ticker"] + CANDS]
    return panel.merge(ch, on=["date", "ticker"], how="left")


def nw(x, lags=3):
    x = np.asarray(pd.Series(x).dropna())
    r = sm.OLS(x, np.ones((len(x), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(r.params[0]), float(r.tvalues[0]), len(x)


def fm_slopes(p, c, controls=()):
    out = {}
    for d, g in p.groupby("date"):
        g = g.dropna(subset=[c])
        if len(g) < 50:
            continue
        X = sm.add_constant(g[[c, *controls]].fillna(0.0))
        out[d] = sm.OLS(g["fwd_ret"], X).fit().params[c]
    return pd.Series(out)


def ls_book(p, c, sign, q=0.2):
    rows, prev = [], pd.Series(dtype=float)
    for d, g in p.groupby("date"):
        s = (sign * g.set_index("ticker")[c]).dropna()
        if len(s) < 50:
            continue
        k = int(len(s) * q)
        w = pd.Series(0.0, index=s.index)
        w[s.nlargest(k).index] = 0.5 / k
        w[s.nsmallest(k).index] = -0.5 / k
        r = (w * g.set_index("ticker")["fwd_ret"].reindex(w.index)).sum()
        turn = (w.reindex(w.index.union(prev.index), fill_value=0)
                - prev.reindex(w.index.union(prev.index), fill_value=0)).abs().sum()
        rows.append((d, r, r - TC * turn))
        prev = w
    df = pd.DataFrame(rows, columns=["date", "gross", "net"]).set_index("date")
    sr = lambda x: x.mean() / x.std() * np.sqrt(12)  # noqa: E731
    return sr(df.gross), sr(df.net)


def holm(pvals: dict, alpha=FWER):
    order = sorted(pvals, key=pvals.get)
    m, passed, still = len(order), {}, True
    for k, name in enumerate(order):
        thr = alpha / (m - k)
        still = still and pvals[name] <= thr
        passed[name] = (still, thr)
    return passed


def main():
    p = dev_panel()
    print(f"DEV: {p.date.nunique()} rebalances [{p.date.min().date()} → {p.date.max().date()}]\n")
    res, pvals = {}, {}
    for c in CANDS:
        s = SIGN[c]
        m, t, n = nw(fm_slopes(p, c))
        mj, tj, _ = nw(fm_slopes(p, c, EXISTING))
        ic = p.groupby("date").apply(lambda g: g[c].corr(g["fwd_ret"], method="spearman")).mean()
        gs, ns = ls_book(p, c, s)
        sub = {k: nw(fm_slopes(p[(p.date >= a) & (p.date < b)], c))[1] for k, (a, b) in SUB.items()}
        res[c] = {"sign": s, "prem_ann": m * 12, "t": t, "n": n, "joint_t": tj, "ic": ic,
                  "ls_gross": gs, "ls_net": ns, **{f"t_{k}": v for k, v in sub.items()},
                  "p": float(stats.t.sf(s * t, n - 1)), "p_joint": float(stats.t.sf(s * tj, n - 1))}
        pvals[c] = res[c]["p"]
    ev = events.run()
    res["index_deletion"] = {"sign": +1, "prem_ann": ev[21]["mean_car"], "t": ev[21]["t"], "n": ev[21]["n_months"],
                             "p": ev[21]["p_one_sided"], "joint_t": ev[63]["t"], "p_joint": ev[63]["p_one_sided"],
                             "n_events": ev[21]["n_events"], "car63": ev[63]["mean_car"]}
    pvals["index_deletion"] = res["index_deletion"]["p"]
    h = holm(pvals)

    print(f"{'candidate':<16}{'sign':>5}{'prem/yr':>9}{'t':>7}{'p(1s)':>8}{'Holm thr':>9}{'Holm':>6}"
          f"{'joint t':>9}{'IC':>8}{'LS gross':>9}{'LS net':>8}{'t 16-19':>8}{'t 20-24':>8}{'PASS':>6}")
    for c, r in res.items():
        hp, thr = h[c]
        inc = r["p_joint"] < 0.05
        ok = hp and inc
        r.update({"holm_pass": hp, "holm_thr": thr, "incremental": inc, "PASS": ok})
        if c == "index_deletion":
            print(f"{c:<16}{r['sign']:>+5}{r['prem_ann']:>+9.2%}{r['t']:>+7.2f}{r['p']:>8.4f}{thr:>9.4f}"
                  f"{'yes' if hp else 'no':>6}{r['joint_t']:>+9.2f}{'':>8}{'CAR21 (n=' + str(r['n_events']) + ')':>17}"
                  f"{'':>16}{'YES' if ok else 'no':>6}")
            continue
        print(f"{c:<16}{r['sign']:>+5}{r['prem_ann']:>+9.2%}{r['t']:>+7.2f}{r['p']:>8.4f}{thr:>9.4f}"
              f"{'yes' if hp else 'no':>6}{r['joint_t']:>+9.2f}{r['ic']:>+8.4f}{r['ls_gross']:>+9.2f}{r['ls_net']:>+8.2f}"
              f"{r['t_2016-19']:>+8.2f}{r['t_2020-24']:>+8.2f}{'YES' if ok else 'no':>6}")
    print("\n(prem/yr: FM slope ×12 per 1σ, or mean CAR for index_deletion. joint t: slope with the 5 existing "
          "factors,\n or CAR[+1,+63] t for index_deletion. HLZ hurdle |t|>3 reported by the t column.)")

    corr = p.groupby("date")[CANDS + EXISTING].corr().groupby(level=1).mean().loc[CANDS, CANDS + EXISTING]
    print("\nmean cross-sectional correlation:\n" + corr.round(2).to_string())
    pd.DataFrame(res).T.to_csv(ROOT / "results" / "screen_dev.csv")


if __name__ == "__main__":
    (ROOT / "results").mkdir(exist_ok=True)
    main()
