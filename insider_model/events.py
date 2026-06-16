"""
Insider purchase events from the SEC Form 3/4/5 structured data sets.

Source (verified 2026-06):
  https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{year}q{q}_form345.zip

Each quarterly ZIP contains flattened TSVs; we use:
  SUBMISSION.tsv      ACCESSION_NUMBER, FILING_DATE, ISSUERTRADINGSYMBOL, DOCUMENT_TYPE
  NONDERIV_TRANS.tsv  ACCESSION_NUMBER, TRANS_DATE, TRANS_CODE, TRANS_SHARES,
                      TRANS_PRICEPERSHARE, TRANS_ACQUIRED_DISP_CD
  REPORTINGOWNER.tsv  ACCESSION_NUMBER, RPTOWNERCIK, (IS_DIRECTOR / IS_OFFICER /
                      IS_TENPERCENTOWNER flags — column names vary slightly
                      across vintages, handled defensively)

Output: one row per (accession, owner) open-market purchase —
  [ticker, filed, trans_date, owner_cik, shares, price, value, is_officer_director]

Only TRANS_CODE == 'P' (open-market purchase) and TRANS_ACQUIRED_DISP_CD == 'A'
are kept: Lakonishok & Lee (2001) — buys are informative, sells are noise.

The event timestamp consumers should use is `filed` (the SEC filing date):
that is when the information became public. run.py requires filed < rebalance
date — same lesson as the PEAD model's ann_date fix (2026-06).
"""

import io
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from loguru import logger

_URL = ("https://www.sec.gov/files/structureddata/data/"
        "insider-transactions-data-sets/{year}q{q}_form345.zip")
# SEC fair-access policy: identify yourself, stay under 10 req/s (we do ~1/2s)
_HEADERS = {"User-Agent": "markwang426@gmail.com personal quant research"}
_DELAY = 2.0


def _read_tsv(zf: zipfile.ZipFile, name: str, usecols: List[str]) -> Optional[pd.DataFrame]:
    """Read one TSV from the zip, tolerant to column-name drift across vintages."""
    member = next((n for n in zf.namelist() if n.upper().endswith(name.upper())), None)
    if member is None:
        return None
    with zf.open(member) as f:
        # Peek header to resolve actual column names case-insensitively
        header = f.readline().decode("utf-8", errors="replace").rstrip("\n").split("\t")
    lookup = {c.upper().strip(): c for c in header}
    resolved = [lookup[c] for c in usecols if c in lookup]
    with zf.open(member) as f:
        df = pd.read_csv(
            f, sep="\t", usecols=resolved, dtype=str,
            na_values=["", "NULL"], keep_default_na=True,
            on_bad_lines="skip", low_memory=False,
        )
    df.columns = [c.upper().strip() for c in df.columns]
    return df


def _parse_dates(s: pd.Series) -> pd.Series:
    """SEC structured data uses DD-MON-YYYY (e.g. 28-FEB-2024); fall back to generic."""
    out = pd.to_datetime(s, format="%d-%b-%Y", errors="coerce")
    if out.isna().mean() > 0.5:
        out = pd.to_datetime(s, errors="coerce")
    return out


def _fetch_with_retries(url: str, attempts: int = 5) -> Optional[bytes]:
    """
    Download with exponential backoff. SEC drops connections under load
    (BrokenPipe / ChunkedEncodingError) — observed in practice after ~7 rapid
    requests; a short pause resolves it.
    """
    backoff = 5.0
    for attempt in range(1, attempts + 1):
        try:
            with requests.Session() as s:
                r = s.get(url, headers=_HEADERS, timeout=120)
            if r.status_code == 200:
                return r.content
            if r.status_code in (403, 429, 503):
                logger.warning(f"  HTTP {r.status_code} (throttled), attempt {attempt}/{attempts} "
                               f"— sleeping {backoff:.0f}s")
            else:
                logger.warning(f"  HTTP {r.status_code}, attempt {attempt}/{attempts}")
        except (requests.RequestException, ConnectionError, OSError) as e:
            logger.warning(f"  Connection error ({type(e).__name__}), attempt {attempt}/{attempts} "
                           f"— sleeping {backoff:.0f}s")
        if attempt < attempts:
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
    return None


def _quarter_events(year: int, q: int, cache_dir: Path) -> pd.DataFrame:
    """Download (or load cached) one quarter and extract purchase events."""
    pq = cache_dir / f"form345_{year}q{q}_purchases.parquet"
    if pq.exists():
        return pd.read_parquet(pq)

    url = _URL.format(year=year, q=q)
    logger.info(f"Fetching {url}")
    content = _fetch_with_retries(url)
    if content is None:
        # Raise rather than silently skip: a missing quarter would leave a
        # hole in the signal. Per-quarter caching makes re-running cheap.
        raise RuntimeError(f"Failed to download {url} after retries — re-run to resume")
    zf = zipfile.ZipFile(io.BytesIO(content))

    sub = _read_tsv(zf, "SUBMISSION.tsv",
                    ["ACCESSION_NUMBER", "FILING_DATE", "ISSUERTRADINGSYMBOL", "DOCUMENT_TYPE"])
    trans = _read_tsv(zf, "NONDERIV_TRANS.tsv",
                      ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
                       "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD"])
    owner = _read_tsv(zf, "REPORTINGOWNER.tsv",
                      ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP",
                       "IS_DIRECTOR", "IS_OFFICER", "IS_TENPERCENTOWNER", "IS_OTHER"])
    if sub is None or trans is None:
        logger.warning(f"  {year}q{q}: missing expected TSVs — skipping")
        return pd.DataFrame()

    # Open-market purchases only
    t = trans[(trans.get("TRANS_CODE", "").astype(str).str.upper() == "P")]
    if "TRANS_ACQUIRED_DISP_CD" in t.columns:
        t = t[t["TRANS_ACQUIRED_DISP_CD"].astype(str).str.upper().isin(["A", "NAN"])]
    if t.empty:
        return pd.DataFrame()

    t = t.copy()
    t["shares"] = pd.to_numeric(t["TRANS_SHARES"], errors="coerce")
    t["price"]  = pd.to_numeric(t.get("TRANS_PRICEPERSHARE"), errors="coerce")
    t["trans_date"] = _parse_dates(t["TRANS_DATE"])
    t = t.dropna(subset=["shares", "trans_date"])
    t["value"] = (t["shares"] * t["price"]).fillna(0.0)
    agg = t.groupby("ACCESSION_NUMBER").agg(
        shares=("shares", "sum"), value=("value", "sum"),
        price=("price", "mean"), trans_date=("trans_date", "min"),
    ).reset_index()

    sub = sub.copy()
    sub["filed"]  = _parse_dates(sub["FILING_DATE"])
    sub["ticker"] = sub["ISSUERTRADINGSYMBOL"].astype(str).str.upper().str.strip()
    merged = agg.merge(
        sub[["ACCESSION_NUMBER", "filed", "ticker"]], on="ACCESSION_NUMBER", how="inner"
    ).dropna(subset=["filed", "ticker"])

    # Owner identity → distinct-buyer counting + officer/director flag
    if owner is not None and "RPTOWNERCIK" in owner.columns:
        ow = owner.copy()
        is_od = pd.Series(False, index=ow.index)
        for col in ("IS_DIRECTOR", "IS_OFFICER"):
            if col in ow.columns:
                is_od |= ow[col].astype(str).str.strip().isin(["1", "true", "True", "Y"])
        if "RPTOWNER_RELATIONSHIP" in ow.columns:   # older vintages: text field
            rel = ow["RPTOWNER_RELATIONSHIP"].astype(str).str.upper()
            is_od |= rel.str.contains("DIRECTOR|OFFICER", regex=True, na=False)
        ow["is_officer_director"] = is_od
        ow_agg = ow.groupby("ACCESSION_NUMBER").agg(
            owner_cik=("RPTOWNERCIK", "first"),
            is_officer_director=("is_officer_director", "any"),
        ).reset_index()
        merged = merged.merge(ow_agg, on="ACCESSION_NUMBER", how="left")
    else:
        merged["owner_cik"] = merged["ACCESSION_NUMBER"]   # fallback: accession≈owner
        merged["is_officer_director"] = True

    merged["owner_cik"] = merged["owner_cik"].fillna(merged["ACCESSION_NUMBER"])
    merged["is_officer_director"] = merged["is_officer_director"].fillna(False)
    out = merged[["ticker", "filed", "trans_date", "owner_cik",
                  "shares", "price", "value", "is_officer_director"]]

    cache_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(pq)
    logger.info(f"  {year}q{q}: {len(out)} purchase filings cached")
    return out


def flag_routine_buyers(events: pd.DataFrame) -> pd.DataFrame:
    """
    Cohen, Malloy & Pomorski (2012) "Decoding Inside Information":
    insiders who trade in the SAME CALENDAR MONTH year after year are
    "routine" traders (scheduled diversification, vesting calendars) and
    their trades carry no predictive content. Opportunistic (non-routine)
    trades carry nearly all the alpha.

    A purchase by owner o filed in month m of year y is flagged routine if
    the same owner also filed purchases in month m of BOTH y-1 and y-2
    (i.e. the third consecutive year of same-month buying).

    Point-in-time safe: the classification of a trade at time t only looks
    BACKWARD at that owner's prior filings.

    Adds a boolean column `is_routine`.
    """
    ev = events.copy()
    ev["_y"] = ev["filed"].dt.year
    ev["_m"] = ev["filed"].dt.month
    seen = set(map(tuple, ev[["owner_cik", "_m", "_y"]].drop_duplicates().values))
    ev["is_routine"] = [
        (o, m, y - 1) in seen and (o, m, y - 2) in seen
        for o, m, y in ev[["owner_cik", "_m", "_y"]].values
    ]
    n_routine = int(ev["is_routine"].sum())
    logger.info(f"Routine-buyer filter: {n_routine}/{len(ev)} purchases flagged routine "
                f"({n_routine / max(len(ev), 1):.1%})")
    return ev.drop(columns=["_y", "_m"])


def build_purchase_events(
    start_year: int,
    end_year: int,
    cache_dir: str = "cache",
) -> pd.DataFrame:
    """
    All open-market insider purchases, start_year Q1 → end_year Q4.
    Returns DataFrame [ticker, filed, trans_date, owner_cik, shares, price,
    value, is_officer_director], sorted by filed date. Quarterly downloads are
    cached individually, plus one combined parquet.
    """
    cdir = Path(cache_dir)
    combined = cdir / f"insider_purchases_{start_year}_{end_year}.parquet"
    if combined.exists():
        logger.info(f"Loading cached insider purchases from {combined}")
        return pd.read_parquet(combined)

    frames = []
    for year in range(start_year, end_year + 1):
        for q in (1, 2, 3, 4):
            df = _quarter_events(year, q, cdir)
            if not df.empty:
                frames.append(df)
            time.sleep(_DELAY)

    if not frames:
        raise RuntimeError("No insider data retrieved — check network / SEC availability")
    events = pd.concat(frames, ignore_index=True).sort_values("filed").reset_index(drop=True)
    events.to_parquet(combined)
    logger.info(f"Cached {len(events)} insider purchases "
                f"({events.ticker.nunique()} tickers) → {combined}")
    return events
