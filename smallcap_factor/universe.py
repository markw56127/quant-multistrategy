"""
S&P 600 SmallCap universe — constituent list with GICS sectors.

SURVIVORSHIP CAVEAT (read this): unlike the S&P 500 sleeve, there is no free
point-in-time membership dataset for the S&P 600 (the fja05680 history is large-cap
only). We therefore use TODAY's constituents, which conditions on survival — and
small-caps delist far more often than large-caps, so this bias is *larger* here than
anywhere else in the repo. Mitigations: (1) prices are fetched survivorship-free-style
(partial history retained, so a name that delisted recently still contributes until it
stops trading); (2) we judge the experiment on the cross-sectional IC and the long-short
spread, which are ranking statistics and far more robust to a level-survivorship bias
than absolute total return. Absolute returns here are optimistic — do not headline them.
"""

from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from loguru import logger

_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"


def fetch_sp600_universe(cache_dir: Optional[str] = None) -> pd.DataFrame:
    """DataFrame with columns Symbol, GICS Sector (current S&P 600 constituents)."""
    if cache_dir:
        p = Path(cache_dir) / "sp600_universe.csv"
        if p.exists():
            logger.info(f"Loading cached S&P 600 universe from {p}")
            return pd.read_csv(p)

    logger.info("Fetching S&P 600 constituents from Wikipedia...")
    r = requests.get(_WIKI_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    tables = pd.read_html(StringIO(r.text))
    df = next(t for t in tables
              if any("Symbol" in str(c) for c in t.columns)
              and any("GICS Sector" in str(c) for c in t.columns))
    df = df[["Symbol", "GICS Sector"]].copy()
    df["Symbol"] = df["Symbol"].astype(str).str.replace(".", "-", regex=False).str.strip()
    df = df.drop_duplicates("Symbol").reset_index(drop=True)
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(cache_dir) / "sp600_universe.csv", index=False)
    logger.info(f"S&P 600 universe: {len(df)} constituents")
    return df


if __name__ == "__main__":
    print(fetch_sp600_universe(cache_dir="cache").head())
