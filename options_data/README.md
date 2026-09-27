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

## Schedule it: INSTALLED 2026-09-26 (launchd)

**Coverage gap:** between 2026-07-06 and 2026-09-26 the collector was never scheduled, so
only one day was banked. That window is permanently lost. Collection resumed 2026-09-26.

The job runs weekdays at **12:30 PT (15:30 ET)** via launchd, not cron: cron on macOS can't
read `~/Desktop` without Full Disk Access, fails silently, and skips runs while the Mac
sleeps. launchd runs a missed slot when the Mac wakes. The job definition is kept at
`com.markwang.options-snapshots.plist`.

```bash
# install / reload
cp options_data/com.markwang.options-snapshots.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.markwang.options-snapshots.plist
# status (runs, last exit code)
launchctl print gui/$(id -u)/com.markwang.options-snapshots | grep -E "runs|last exit"
# remove
launchctl bootout gui/$(id -u)/com.markwang.options-snapshots
```

Log: `options_data/collect.log`. **Check it weekly for gaps.** A day the Mac is off
entirely is still lost; the snapshot folder date plus the UTC timestamp on every row
record what was actually captured. The 2026-09-26 (Saturday) folder was the install
test and holds Friday's closing chains.

## Back it up

`snapshots/` is **git-ignored** (grows ~450 MB/yr) and is the only
non-regenerable data in the repo — everything else re-downloads. Sync it to
cloud storage or an external disk periodically.
