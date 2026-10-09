"""
Daily implied-volatility surfaces from the banked option chains.

Per chain and expiry:
  1. QUOTE FILTER   bid > 0, ask > bid, relative spread < 50%, mid ≥ 0.05, at least 5 days to expiry
  2. FORWARD        put-call parity, C − P = D·F − D·K: robust OLS (3-MAD outlier pass) of (C−P)
                    on K over strikes within ±5% of spot gives the discount factor D and forward F,
                    absorbing dividends, borrow and rates with no assumptions. Wider bands pull in
                    stale deep quotes (2026-12-31 gave D = 0.93 at ±20% vs 0.989 at ±5%). Fallback
                    (D from a 4% rate, F from spot) if the implied rate is outside 0–10%
  3. OWN IV         Black-76 on the forward from OTM mids (puts K < F, calls K ≥ F). Yahoo's
                    impliedVolatility column is not used: 8% of SPX contracts carry IV < 1%
  4. NO-ARBITRAGE   call prices (OTM puts converted by parity) must be monotone and convex in K;
                    total implied variance must not decrease with T at fixed log-moneyness
  5. SVI            raw SVI per expiry on total variance w(k) = σ²T, k = ln(K/F):
                        w(k) = a + b·(ρ·(k − m) + √((k − m)² + s²))
                    (Gatheral 2004), fitted by bounded least squares with multiple starts, plus
                    Gatheral's butterfly-density check g(k) ≥ 0 on the fitted smile
  6. SUMMARY        at 30 and 90 days, interpolating total variance linearly in T at fixed k, and
                    only inside each expiry's quoted k-range (else NaN): ATM vol;
                    skew = IV(k=−sd) − IV(k=+sd) and curvature = ½[IV(−sd)+IV(+sd)] − ATM,
                    where sd = σ_ATM·√T (±1 standard deviation, ≈ 16-delta put vs call);
                    term slope = ATM90 − ATM30

SPX options are European. SPY/QQQ/IWM and single stocks are American, so using only OTM
options (where early exercise is worth little) keeps Black-76 a close approximation.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
SNAP = ROOT.parent / "options_data" / "snapshots"
MIN_DAYS, MAX_REL_SPREAD, MIN_MID = 5, 0.5, 0.05


# ─────────────────────────────────────────────────────────── Black-76

def b76(F, K, T, D, vol, call):
    sv = vol * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sv ** 2) / sv
    d2 = d1 - sv
    return D * (F * norm.cdf(d1) - K * norm.cdf(d2)) if call else D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def implied_vol(price, F, K, T, D, call):
    intrinsic = D * max(F - K, 0) if call else D * max(K - F, 0)
    if price <= intrinsic + 1e-10:
        return np.nan
    f = lambda v: b76(F, K, T, D, v, call) - price  # noqa: E731
    try:
        return brentq(f, 1e-4, 5.0, xtol=1e-8)
    except ValueError:
        return np.nan


# ─────────────────────────────────────────────────────────── per-expiry processing

def clean(chain: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    d = chain.copy()
    d["mid"] = (d.bid + d.ask) / 2
    ok = (d.bid > 0) & (d.ask > d.bid) & ((d.ask - d.bid) / d["mid"] < MAX_REL_SPREAD) & (d["mid"] >= MIN_MID)
    d = d[ok].copy()
    # expiry at 16:00 ET ≈ 20:00 UTC; T in years from the actual fetch time
    d["T"] = ((pd.to_datetime(d.expiry) + pd.Timedelta(hours=20)).dt.tz_localize("UTC") - asof).dt.total_seconds() / (365.25 * 86400)
    return d[d["T"] >= MIN_DAYS / 365.25]


def implied_forward(g: pd.DataFrame, spot: float, T: float, rate: float):
    """OLS of C − P on K across near-the-money strikes quoted on both sides."""
    c = g[g.type == "call"].set_index("strike")["mid"]
    p = g[g.type == "put"].set_index("strike")["mid"]
    both = c.index.intersection(p.index)
    both = both[(both > 0.95 * spot) & (both < 1.05 * spot)]   # liquid strikes only: deep quotes are often stale
    if len(both) >= 4:
        y, K = (c[both] - p[both]).values, both.values.astype(float)
        slope, icpt = np.polyfit(K, y, 1)
        r = y - (slope * K + icpt)
        mad = np.median(np.abs(r - np.median(r))) + 1e-9
        keep = np.abs(r - np.median(r)) <= 3 * 1.4826 * mad
        if keep.sum() >= 4:                     # stale/bad quotes are common in Yahoo chains
            slope, icpt = np.polyfit(K[keep], y[keep], 1)
        D = -slope
        if np.exp(-0.10 * T) <= D <= 1.0005:           # implied rate must be within 0–10%
            return icpt / D, min(D, 1.0), "parity"
        # D implausible (short T amplifies quote noise): fix D, keep the forward from parity
        D = np.exp(-rate * T)
        Kk, yk = (K[keep], y[keep]) if keep.sum() >= 4 else (K, y)
        return float(np.median(yk / D + Kk)), D, "parity_fixedD"
    D = np.exp(-rate * T)
    return spot / D, D, "fallback"


def expiry_ivs(g: pd.DataFrame, spot: float, rate: float):
    T = g["T"].iloc[0]
    F, D, how = implied_forward(g, spot, T, rate)
    otm = g[((g.type == "put") & (g.strike < F)) | ((g.type == "call") & (g.strike >= F))].copy()
    otm["iv"] = [implied_vol(m, F, k, T, D, t == "call") for m, k, t in zip(otm["mid"], otm.strike, otm.type)]
    otm = otm.dropna(subset=["iv"])
    otm["k"] = np.log(otm.strike / F)
    otm["w"] = otm.iv ** 2 * T
    return otm.sort_values("k"), F, D, how


def arbitrage_checks(otm: pd.DataFrame, F: float, D: float):
    """Convert OTM puts to calls by parity, then count monotonicity and convexity violations
    that are LARGER than the bid/ask spreads involved. Mid-quote noise alone does not count."""
    K = otm.strike.values.astype(float)
    C = np.where(otm.type == "call", otm["mid"], otm["mid"] + D * (F - K))
    hs = ((otm.ask - otm.bid) / 2).values if {"ask", "bid"} <= set(otm.columns) else np.zeros(len(K))
    o = np.argsort(K)
    K, C, hs = K[o], C[o], hs[o]
    mono = int((np.diff(C) > hs[:-1] + hs[1:] + 1e-9).sum())
    dK = np.diff(K)
    slopes = np.diff(C) / dK
    tol = (hs[:-2] + 2 * hs[1:-1] + hs[2:]) / np.minimum(dK[:-1], dK[1:])
    conv = int((np.diff(slopes) < -tol - 1e-9).sum())
    return mono, conv


# ─────────────────────────────────────────────────────────── SVI

def svi_w(k, a, b, rho, m, s):
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s ** 2))


def fit_svi(k, w, T=None):
    """Fit raw SVI. With T, residuals are in implied-vol units with a soft-L1 loss (f_scale = 1
    vol point), so isolated stale quotes cannot drag the smile. Without T, plain least squares
    on total variance."""
    iv = np.sqrt(np.maximum(w, 0) / T) if T else None
    def resid(p):
        a, b, rho, m, s = p
        wf = svi_w(k, *p)
        r = (np.sqrt(np.maximum(wf, 1e-12) / T) - iv) if T else (wf - w)
        floor = a + b * s * np.sqrt(1 - rho ** 2)            # min of w(k): must be ≥ 0
        return np.append(r, 100.0 * max(0.0, -floor))
    best = None
    for rho0 in (-0.7, -0.3, 0.0):
        for s0 in (0.05, 0.2):
            x0 = [max(w.min() * 0.5, 1e-5), 0.1, rho0, 0.0, s0]
            r = least_squares(resid, x0, bounds=([-1, 0, -0.999, -1, 1e-4], [1, 5, 0.999, 1, 2]),
                              loss="soft_l1" if T else "linear", f_scale=0.01, max_nfev=5000)
            a, b, rho, m, s = r.x
            if a + b * s * np.sqrt(1 - rho ** 2) < -1e-8:                   # w ≥ 0 everywhere
                continue
            if best is None or r.cost < best.cost:
                best = r
    return None if best is None else best.x


def svi_butterfly_ok(p, kgrid=np.linspace(-1, 1, 401)):
    """Gatheral's density condition g(k) ≥ 0 (no butterfly arbitrage) on a grid. Callers pass
    the quoted k-range: checking SVI's extrapolation far outside the data is meaningless."""
    a, b, rho, m, s = p
    x = kgrid - m
    w = svi_w(kgrid, *p)
    w1 = b * (rho + x / np.sqrt(x ** 2 + s ** 2))
    w2 = b * s ** 2 / (x ** 2 + s ** 2) ** 1.5
    g = (1 - kgrid * w1 / (2 * w)) ** 2 - w1 ** 2 / 4 * (1 / w + 0.25) + w2 / 2
    return bool((g >= -1e-8).all())


# ─────────────────────────────────────────────────────────── a whole surface

@dataclass
class Surface:
    underlying: str
    date: str
    spot: float
    capture: str                    # "regular" (during that day's session) or "after_hours"
    expiries: pd.DataFrame          # one row per expiry: T, F, D, forward method, SVI params, rmse, checks
    points: pd.DataFrame            # all OTM points with own IV

    def total_var(self, k, T, tol=0.01):
        """Linear interpolation of SVI total variance in T at fixed k (flat-forward beyond the ends).
        NaN unless k lies inside the QUOTED range of every expiry used: never read SVI's
        extrapolated wings (AT&T's coarse strikes gave a 95-vol-point 'skew' that way)."""
        e = self.expiries.dropna(subset=["a"]).sort_values("T").reset_index(drop=True)
        Ts = e["T"].values
        if T <= Ts[0]:
            use, wts = [0], [T / Ts[0]]
        elif T >= Ts[-1]:
            use, wts = [len(Ts) - 1], [T / Ts[-1]]
        else:
            j = int(np.searchsorted(Ts, T))
            x = (T - Ts[j - 1]) / (Ts[j] - Ts[j - 1])
            use, wts = [j - 1, j], [1 - x, x]
        total = 0.0
        for i, wt in zip(use, wts):
            r = e.iloc[i]
            if not (r.kmin - tol <= k <= r.kmax + tol):
                return np.nan
            total += wt * svi_w(k, *r[["a", "b", "rho", "m", "s"]].values.astype(float))
        return total

    def iv(self, k, days):
        T = days / 365.25
        w = self.total_var(k, T)
        return float(np.sqrt(max(w, 0) / T)) if np.isfinite(w) else np.nan

    def summary(self):
        out = {"date": self.date, "underlying": self.underlying, "spot": self.spot, "capture": self.capture}
        for days in (30, 90):
            # volatility-scaled moneyness: ±1 sd of the expected move, k = ±σ_ATM·√T
            # (≈ 16-delta put vs call), comparable across names and horizons, inside the quotes
            atm = self.iv(0.0, days)
            sd = atm * np.sqrt(days / 365.25) if np.isfinite(atm) else np.nan
            dn, up = (self.iv(-sd, days), self.iv(sd, days)) if np.isfinite(sd) else (np.nan, np.nan)
            out.update({f"atm{days}": atm, f"skew{days}": dn - up, f"curv{days}": 0.5 * (dn + up) - atm})
        out["term_slope"] = out["atm90"] - out["atm30"]
        e = self.expiries
        out.update({"n_expiries": int(e.a.notna().sum()), "n_points": len(self.points),
                    "fwd_fixedD_share": float((e.fwd_method == "parity_fixedD").mean()),
                    "fit_rmse_volpts": float(np.nanmedian(e.rmse_vol) * 100),
                    "butterfly_viol": int((~e.svi_ok.astype(bool)).sum()),
                    "mono_viol": int(e.mono_viol.sum()), "convex_viol": int(e.convex_viol.sum()),
                    "calendar_viol": int(self.calendar_violations()),
                    "fwd_parity_share": float(e.fwd_method.str.startswith("parity").mean())})
        return out

    def calendar_violations(self):
        """Total variance must not fall with T at fixed k, checked where every expiry has quotes."""
        e = self.expiries.dropna(subset=["a"]).sort_values("T")
        lo, hi = max(e.kmin.max(), -0.10), min(e.kmax.min(), 0.10)
        if hi <= lo:
            return 0
        ks = np.linspace(lo, hi, 9)
        W = np.array([[svi_w(k, *r[["a", "b", "rho", "m", "s"]].values.astype(float)) for k in ks]
                      for _, r in e.iterrows()])
        return int((np.diff(W, axis=0) < -1e-6).sum()) if len(W) > 1 else 0


def build_surface(path: Path, underlying: str, date: str, rate: float, max_days: int = 400) -> Surface:
    raw = pd.read_parquet(path)
    asof = pd.Timestamp(raw.fetched_utc.iloc[0])
    et = asof.tz_convert("America/New_York")
    regular = et.date().isoformat() == date and (9, 30) <= (et.hour, et.minute) <= (16, 15) and et.weekday() < 5
    spot = float(raw.spot.dropna().iloc[0]) if raw.spot.notna().any() else np.nan
    root = raw.contractSymbol.str.extract(r"^([A-Z]+)\d")[0]
    if underlying == "SPX":                     # AM-settled SPX and PM-settled SPXW share third-Friday dates
        has_w = raw.assign(r=root).groupby("expiry").r.transform(lambda r: (r == "SPXW").any())
        raw = raw[~has_w | (root == "SPXW")]
    d = clean(raw, asof)
    d = d[d["T"] <= max_days / 365.25]
    rows, pts = [], []
    for exp, g in d.groupby("expiry"):
        if np.isnan(spot):
            continue
        otm, F, D, how = expiry_ivs(g, spot, rate)
        T = g["T"].iloc[0]
        mono, conv = arbitrage_checks(otm, F, D) if len(otm) >= 3 else (0, 0)
        rec = {"expiry": exp, "T": T, "F": F, "D": D, "fwd_method": how, "n": len(otm),
               "mono_viol": mono, "convex_viol": conv, "a": np.nan, "b": np.nan, "rho": np.nan, "m": np.nan,
               "s": np.nan, "rmse_vol": np.nan, "svi_ok": True}
        sel = otm[(otm.k > -0.6) & (otm.k < 0.4)]
        if len(sel) >= 6:
            p = fit_svi(sel.k.values, sel.w.values, T)
            if p is not None:
                fit_iv = np.sqrt(np.maximum(svi_w(sel.k.values, *p), 0) / T)
                rec.update(dict(zip(["a", "b", "rho", "m", "s"], p)),
                           rmse_vol=float(np.sqrt(np.mean((fit_iv - sel.iv.values) ** 2))),
                           svi_ok=svi_butterfly_ok(p, np.linspace(sel.k.min(), sel.k.max(), 201)),
                           kmin=float(sel.k.min()), kmax=float(sel.k.max()))
        rows.append(rec)
        pts.append(otm.assign(expiry=exp))
    return Surface(underlying, date, spot, "regular" if regular else "after_hours", pd.DataFrame(rows),
                   pd.concat(pts, ignore_index=True) if pts else pd.DataFrame())
