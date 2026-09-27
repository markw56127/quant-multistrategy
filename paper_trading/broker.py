"""
Minimal Alpaca REST client: PAPER ONLY.

Keys come from the environment or from paper_trading/.env (git-ignored):
    APCA_API_KEY_ID=...
    APCA_API_SECRET_KEY=...

The base URL is hard-pinned to the paper endpoint, and any other value raises. This module
cannot place live orders.
"""

import os
from pathlib import Path

import requests

PAPER_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"
ENV_FILE = Path(__file__).resolve().parent / ".env"


def _load_env():
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def have_keys() -> bool:
    _load_env()
    return bool(os.environ.get("APCA_API_KEY_ID") and os.environ.get("APCA_API_SECRET_KEY"))


class PaperBroker:
    def __init__(self, base_url: str = PAPER_URL, timeout: float = 15):
        if base_url.rstrip("/") != PAPER_URL:
            raise ValueError(f"refusing non-paper endpoint {base_url!r}; this client is paper-only")
        _load_env()
        key, secret = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
        if not (key and secret):
            raise RuntimeError(f"no Alpaca paper keys: set APCA_API_KEY_ID / APCA_API_SECRET_KEY "
                               f"in the environment or in {ENV_FILE}")
        self.base, self.timeout = base_url.rstrip("/"), timeout
        self.s = requests.Session()
        self.s.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})

    def _req(self, method, path, base=None, **kw):
        r = self.s.request(method, (base or self.base) + path, timeout=self.timeout, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"Alpaca {method} {path} → {r.status_code}: {r.text[:300]}")
        return r.json() if r.text else None

    # account / market state
    def account(self):
        return self._req("GET", "/v2/account")

    def clock(self):
        return self._req("GET", "/v2/clock")

    def positions(self) -> dict:
        return {p["symbol"]: float(p["qty"]) for p in self._req("GET", "/v2/positions")}

    def latest_trade(self, symbol: str) -> float:
        d = self._req("GET", f"/v2/stocks/{symbol}/trades/latest", base=DATA_URL, params={"feed": "iex"})
        return float(d["trade"]["p"])

    def portfolio_history(self, period="1A", timeframe="1D"):
        return self._req("GET", "/v2/account/portfolio/history", params={"period": period, "timeframe": timeframe})

    # orders
    def submit_market(self, symbol: str, qty: int, side: str, client_order_id: str):
        body = {"symbol": symbol, "qty": str(int(qty)), "side": side, "type": "market",
                "time_in_force": "day", "client_order_id": client_order_id}
        return self._req("POST", "/v2/orders", json=body)

    def order_by_client_id(self, client_order_id: str):
        return self._req("GET", "/v2/orders:by_client_order_id", params={"client_order_id": client_order_id})
