# Alternative Data Model

**Status:** Idea / parked (requires paid data or heavy scraping)

## Concept

Price-based and fundamental factors are so widely known they're largely
arbitraged away. The genuine edge at large quant shops comes from data that
isn't in everyone's models yet. This is where Two Sigma, Citadel, and
WorldQuant actually differentiate.

## Free / low-cost signals worth exploring

- **Google Trends** — search interest as a demand proxy (free API)
- **Wikipedia pageviews** — attention/interest in a company (free API)
- **Reddit / StockTwits sentiment** — retail attention (free-ish, scraping)
- **SEC EDGAR full-text search** — 8-K filings, insider Form 4 transactions
  (free; insider buying clusters are a documented signal)
- **GitHub activity** — for tech companies, commit/star velocity as a product
  momentum proxy
- **App Store rankings** — free-tier scraping for consumer app companies
- **Job postings** — hiring velocity from company career pages (scraping)

## Expensive signals (the real institutional edge)

- Credit card transaction panels ($50k-$500k/yr)
- Satellite imagery (parking lots, oil tanks, crop yields)
- Web traffic (SimilarWeb)
- Email receipt data

## Honest note

This is the highest-ceiling but highest-effort direction. Most free alt-data
signals are weak individually and need careful combination. Insider
transactions (Form 4 via EDGAR) is probably the best free starting point —
it's structured, timestamped, and has documented predictive power
(Lakonishok & Lee 2001).

## Best free starting point

SEC EDGAR Form 4 (insider transactions). We already have the EDGAR fetching
infrastructure. Cluster insider *buys* (not sells — sells are noisy, driven by
diversification/liquidity) as a conviction signal.
