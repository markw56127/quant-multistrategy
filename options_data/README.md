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

## 1-minute bars (added 2026-09-30)

`collect_intraday.py` banks 1-minute OHLCV bars for **41 ETFs**: broad equity, rates and
credit, commodities, sector SPDRs, **leveraged/inverse ETFs** (TQQQ/SQQQ, UPRO/SPXU,
SSO/SDS, TMF/TMV, SOXL/SOXS) and vol ETPs. Yahoo keeps 1-minute history for only about
8 trading days, so this data is lost unless banked. Its purpose is the intraday
forced-flow studies daily data can't reach: leveraged-ETF rebalancing into the close, and
*when* the (crowded) month-end flows now hit.

- Output: `intraday/{YYYY-MM-DD}.parquet`, long format (ts_utc, ticker, OHLCV), about
  0.35 MB/day. Git-ignored, so back it up along with `snapshots/`.
- **Self-healing:** each run re-reads the last 7 days and writes any completed day not on
  disk, so missed runs are filled automatically if the gap is under about a week. Today
  is written only after 16:00 ET.
- Backfilled at install: 2026-09-22 → 2026-09-30.
- Schedule: launchd `com.markwang.intraday-bars`, weekdays **13:30 PT (16:30 ET, after the
  close)**. Log: `collect_intraday.log`. Job definition: `com.markwang.intraday-bars.plist`.
  Install, status and remove work like the snapshot job, with the label swapped.

## Single-stock options, analyst estimates, ETF assets (added 2026-09-30)

`collect_extras.py` banks three more feeds whose history isn't free later:

| part | what | output | schedule |
|---|---|---|---|
| `stock_options` | chains for **50 liquid single stocks**, expiries ≤ 6 months (same schema as the index chains) | `snapshots/{date}/stocks/{TICKER}.parquet` (~1.9 MB/day) | `com.markwang.stock-options`, weekdays **12:40 PT**, pre-close |
| `estimates` | **all current S&P 500 members** (list refreshed from Wikipedia each run): EPS and revenue consensus, EPS trend with its 7/30/60/90-day-ago values, revisions, recommendation counts, price targets, next earnings date. Point-in-time history is otherwise a paid product (I/B/E/S) | `estimates/{date}.parquet`, long format (~0.3 MB/day) | `com.markwang.eod-extras`, weekdays **14:00 PT** |
| `etf_aum` | total assets, shares outstanding, NAV and price for **61 ETFs**. ΔAUM net of returns gives fund flows | `etf_aum/{date}.csv`, one file per day; **tracked in git** | same job, 14:00 PT |

What each enables: cross-sectional option signals (skew, call–put IV spread) for stock
selection; **earnings-revision** signals, the anomaly the free-data screen couldn't test;
and an ETF-flow price-pressure study. Every part retries with backoff, is isolated from
the others, and is idempotent per date (`--force` rewrites). Option expiries also retry
3× on Yahoo timeouts. Log: `collect_extras.log`.

First collection: 2026-09-30, gathered after the close at install time. Coverage: 503/503
S&P 500 names, 50/50 stocks, 61/61 ETFs with total assets (37 also report shares
outstanding).

### The full daily schedule (weekdays, Pacific), revised 2026-10-07

| first run | retries | job | data |
|---|---|---|---|
| 12:30 | 15:00, 18:00 | `com.markwang.options-snapshots` | index/ETF chains (SPY, QQQ, IWM, SPX, VIX), VIX term structure |
| 12:40 | 15:10, 18:10 | `com.markwang.stock-options` | 50 single-stock chains |
| 13:30 | 18:20 | `com.markwang.intraday-bars` | 1-minute bars, 41 ETFs (also self-heals gaps under 7 days) |
| 14:00 | 16:00, 19:00 | `com.markwang.eod-extras` | S&P 500 analyst estimates, ETF assets |
| 19:30 | — | `com.markwang.collection-check` | **macOS notification if any feed is missing today** |

- Every job runs under `caffeinate -i -s`, so a run that starts during a brief dark wake
  holds the Mac awake until it finishes. Collectors are idempotent, so retries only fill
  gaps. A retry capture is post-close, and every row keeps its true `fetched_utc`.
- Logs: `~/Library/Logs/trading_model/*.log`. They were moved out of `~/Desktop` because
  launchd silently lost write access to the Desktop log files after 2026-10-02.
- **Missed a day?** Run `python options_data/catch_up.py` the same evening (or before the
  next open). Chains, estimates and ETF assets are only available as *current* snapshots,
  so a day not captured before the next session opens is gone for good. 1-minute bars and
  the VIX daily series can be backfilled for about a week.
- Status of all jobs: `for L in options-snapshots stock-options intraday-bars eod-extras collection-check; do launchctl print gui/$(id -u)/com.markwang.$L | grep -E "runs|last exit"; done`

### iCloud: why every collector is write-only (2026-10-08)

`~/Desktop` is synced to **iCloud Drive with "Optimize Mac Storage"**, so macOS offloads
older files to the cloud and leaves placeholders on disk. A foreground program reading a
placeholder triggers a download automatically. A **background (launchd) job gets
`[Errno 11] Resource deadlock avoided`** instead. That broke the ETF-assets append on
2026-10-08 (filled the same evening), and it is the likely cause of the Desktop log
failures after 10-02.

- The upside: **all collected data is already backed up to iCloud.**
- The rule: **collectors never read back earlier files.** Every feed writes a new file per
  day (`snapshots/{date}/`, `intraday/{date}.parquet`, `estimates/{date}.parquet`,
  `etf_aum/{date}.csv`, `snapshots/vix_term_structure/{date}.csv`). Idempotency and the
  nightly check use `.exists()`, which reads only metadata and works on placeholders. Logs
  live in `~/Library/Logs`, outside iCloud.
- Reading for research happens in the foreground, where macOS downloads on demand. Use
  `collect_snapshots.load_vix_term_structure()` to combine the VIX files with the legacy
  CSV.
- Legacy files kept: `snapshots/vix_term_structure.csv` (through 2026-10-07) and
  `etf_aum_legacy_until_2026-10-07.csv` (already split into `etf_aum/`).

### Gap log

| dates | what was lost | cause |
|---|---|---|
| 2026-07-07 → 09-25 | everything | collector never scheduled |
| 2026-10-05 (Mon) | index + stock chains, ETF assets | Mac asleep (dark wakes only); jobs froze mid-run |
| 2026-10-08 (Thu) | nothing lost: ETF assets failed at 14:00 and 16:00 (iCloud placeholder), filled at 18:14 | iCloud Optimize Storage |
| 2026-10-06 (Tue) | *recovered* 2026-10-07 01:00 PT from Tuesday's close (see `snapshots/2026-10-06/RECOVERED.txt`) | same |

## Back it up

`snapshots/`, `intraday/` and `estimates/` are **git-ignored** (grows ~450 MB/yr) and is the only
non-regenerable data in the repo — everything else re-downloads. Sync it to
cloud storage or an external disk periodically.
