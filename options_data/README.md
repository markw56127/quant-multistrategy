# options_data — daily options-chain snapshot collector

**Status:** INFRASTRUCTURE (live since 2026-07-06) — no model consumes this
yet, by design. Historical options data is not free (OptionMetrics / ORATS /
CBOE DataShop), so we bank our own: each daily run adds a point-in-time
snapshot nobody can question later. Today's collection is next year's
dataset; a future surface-fitting or vol-risk-premium project starts with
real data instead of a paywall.

## What it collects (one run per trading day)

- `snapshots/{YYYY-MM-DD}/{ticker}.parquet` — full chains (all listed
  expiries, calls+puts: strike, bid/ask/last, volume, OI, implied vol) for
  **SPY, QQQ, IWM, ^SPX, ^VIX**, with underlying spot and a UTC fetch
  timestamp on every row. ~43k contracts / ~1.8 MB per day.
- `snapshots/vix_term_structure.csv` — appended daily ^VIX9D / ^VIX / ^VIX3M
  closes (the always-useful companion series).

Idempotent (same-day reruns skipped; `--force` refreshes). Yahoo chains are
15-minute-delayed retail data — fine for surface history, not for
market-making. Best run in the last hour of the session; any consistent time
works, the timestamp records the truth.

## Schedule it

```
crontab -e
30 15 * * 1-5 cd /Users/markwang/Desktop/trading_model/options_data && /usr/bin/env python collect_snapshots.py >> collect.log 2>&1
```

(Adjust 15:30 to your local offset from ET. A missed day is a permanent hole
— this is the one part of the repo where uptime matters.)

## Back it up

`snapshots/` is **git-ignored** (grows ~450 MB/yr) and is the only
non-regenerable data in the repo — everything else re-downloads. Sync it to
cloud storage or an external disk periodically.
