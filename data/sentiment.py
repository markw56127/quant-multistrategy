import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from loguru import logger

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from transformers import pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class SentimentEngine:
    """
    Converts unstructured financial text (news, earnings calls) into a
    scalar sentiment score in [-1, 1] representing market sentiment.

    Provider hierarchy:
      1. Claude (via Anthropic API) – richest financial reasoning
      2. FinBERT (local HuggingFace) – offline fallback
      3. Simple keyword heuristic – last resort
    """

    BULLISH_KEYWORDS = ["beat", "surge", "record", "strong", "growth", "upside", "outperform", "raised"]
    BEARISH_KEYWORDS = ["miss", "decline", "loss", "weak", "cut", "downside", "underperform", "warning"]

    def __init__(
        self,
        provider: str = "claude",
        claude_model: str = "claude-sonnet-4-6",
        news_api_key: Optional[str] = None,
        sentiment_decay: float = 0.85,
        batch_size: int = 10,
    ):
        self.provider = provider
        self.claude_model = claude_model
        self.news_api_key = news_api_key or os.getenv("NEWSAPI_KEY")
        self.sentiment_decay = sentiment_decay
        self.batch_size = batch_size

        self._client = None
        self._finbert = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Provider initialization
    # ------------------------------------------------------------------

    def _init_provider(self):
        if self.provider == "claude":
            if not ANTHROPIC_AVAILABLE:
                logger.warning("anthropic package not installed. Falling back to FinBERT.")
                self.provider = "finbert"
            else:
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    logger.warning("ANTHROPIC_API_KEY not set. Falling back to FinBERT.")
                    self.provider = "finbert"
                else:
                    self._client = anthropic.Anthropic(api_key=api_key)
                    logger.info("Sentiment engine initialized with Claude")

        if self.provider == "finbert":
            if not TRANSFORMERS_AVAILABLE:
                logger.warning("transformers not installed. Falling back to keyword heuristic.")
                self.provider = "keyword"
            else:
                try:
                    self._finbert = hf_pipeline(
                        "text-classification",
                        model="ProsusAI/finbert",
                        top_k=None,
                        device="cpu",
                    )
                    logger.info("Sentiment engine initialized with FinBERT")
                except Exception as e:
                    logger.warning(f"FinBERT load failed: {e}. Using keyword heuristic.")
                    self.provider = "keyword"

        if self.provider == "keyword":
            logger.info("Sentiment engine initialized with keyword heuristic")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _ensure_initialized(self):
        if not self._initialized:
            self._initialized = True
            self._init_provider()

    def score_text(self, text: str) -> float:
        """Score a single piece of text → scalar in [-1, 1]."""
        if not text or not text.strip():
            return 0.0
        self._ensure_initialized()
        if self.provider == "claude":
            return self._score_claude(text)
        if self.provider == "finbert":
            return self._score_finbert(text)
        return self._score_keyword(text)

    def score_batch(self, texts: List[str]) -> List[float]:
        """Score a batch of texts efficiently."""
        self._ensure_initialized()
        scores = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            if self.provider == "claude":
                scores.extend(self._score_claude_batch(chunk))
            elif self.provider == "finbert":
                scores.extend(self._score_finbert_batch(chunk))
            else:
                scores.extend([self._score_keyword(t) for t in chunk])
        return scores

    def fetch_and_score_news(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        company_name: Optional[str] = None,
    ) -> pd.Series:
        """
        Fetches news headlines for a ticker and returns a daily sentiment time series.
        Applies exponential decay weighting across lookback window.
        """
        articles = self._fetch_news(ticker, company_name, start_date, end_date)
        if not articles:
            date_range = pd.date_range(start_date, end_date, freq="B")
            return pd.Series(0.0, index=date_range, name=f"{ticker}_sentiment")

        # Group articles by date and score
        by_date: Dict[str, List[str]] = {}
        for art in articles:
            date_str = art["date"][:10]
            text = f"{art['title']}. {art.get('description', '')}"
            by_date.setdefault(date_str, []).append(text)

        daily_scores = {}
        for date_str, texts in by_date.items():
            raw_scores = self.score_batch(texts)
            daily_scores[date_str] = float(np.mean(raw_scores))

        date_range = pd.date_range(start_date, end_date, freq="B")
        sentiment = pd.Series(0.0, index=date_range)
        for date_str, score in daily_scores.items():
            try:
                sentiment[pd.Timestamp(date_str)] = score
            except KeyError:
                pass

        # Apply exponential decay forward-fill: carry yesterday's score * decay
        sentiment = self._apply_decay(sentiment)
        sentiment.name = f"{ticker}_sentiment"
        return sentiment

    def build_sentiment_matrix(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        company_names: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Returns DataFrame of sentiment scores: rows=dates, cols=tickers."""
        self._ensure_initialized()
        frames = {}
        for ticker in tickers:
            name = (company_names or {}).get(ticker)
            try:
                frames[ticker] = self.fetch_and_score_news(ticker, start_date, end_date, name)
                time.sleep(0.2)  # rate limiting
            except Exception as e:
                logger.warning(f"Sentiment fetch failed for {ticker}: {e}")
                date_range = pd.date_range(start_date, end_date, freq="B")
                frames[ticker] = pd.Series(0.0, index=date_range, name=ticker)
        return pd.DataFrame(frames)

    # ------------------------------------------------------------------
    # Scoring backends
    # ------------------------------------------------------------------

    def _score_claude(self, text: str) -> float:
        prompt = (
            "You are a quantitative financial analyst. Analyze the following financial news or earnings text "
            "and return ONLY a single floating-point number between -1.0 (extremely bearish) and 1.0 (extremely bullish). "
            "Consider market impact, earnings quality, and macro implications. "
            "Return only the number, nothing else.\n\n"
            f"Text: {text[:2000]}"
        )
        try:
            response = self._client.messages.create(
                model=self.claude_model,
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
            return float(response.content[0].text.strip())
        except (ValueError, IndexError, Exception) as e:
            logger.debug(f"Claude scoring error: {e}")
            return self._score_keyword(text)

    def _score_claude_batch(self, texts: List[str]) -> List[float]:
        combined = "\n---\n".join(
            f"[{i+1}] {t[:500]}" for i, t in enumerate(texts)
        )
        prompt = (
            "You are a quantitative financial analyst. For each numbered text below, provide a sentiment score "
            "between -1.0 (extremely bearish) and 1.0 (extremely bullish). "
            f"Return ONLY {len(texts)} comma-separated numbers, e.g.: 0.7,-0.3,0.1\n\n"
            f"{combined}"
        )
        try:
            response = self._client.messages.create(
                model=self.claude_model,
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            scores = [float(x.strip()) for x in raw.split(",")]
            scores = [max(-1.0, min(1.0, s)) for s in scores]
            if len(scores) == len(texts):
                return scores
        except Exception as e:
            logger.debug(f"Claude batch scoring error: {e}")
        return [self._score_keyword(t) for t in texts]

    def _score_finbert(self, text: str) -> float:
        try:
            result = self._finbert(text[:512])
            label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
            # result is list of [{label, score}]
            best = max(result[0], key=lambda x: x["score"])
            return label_map.get(best["label"].lower(), 0.0) * best["score"]
        except Exception as e:
            logger.debug(f"FinBERT error: {e}")
            return self._score_keyword(text)

    def _score_finbert_batch(self, texts: List[str]) -> List[float]:
        truncated = [t[:512] for t in texts]
        try:
            results = self._finbert(truncated)
            label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
            scores = []
            for result in results:
                best = max(result, key=lambda x: x["score"])
                scores.append(label_map.get(best["label"].lower(), 0.0) * best["score"])
            return scores
        except Exception as e:
            logger.debug(f"FinBERT batch error: {e}")
            return [self._score_keyword(t) for t in texts]

    def _score_keyword(self, text: str) -> float:
        text_lower = text.lower()
        bull = sum(text_lower.count(kw) for kw in self.BULLISH_KEYWORDS)
        bear = sum(text_lower.count(kw) for kw in self.BEARISH_KEYWORDS)
        total = bull + bear
        if total == 0:
            return 0.0
        return (bull - bear) / total

    # ------------------------------------------------------------------
    # News fetching
    # ------------------------------------------------------------------

    def _fetch_news(
        self,
        ticker: str,
        company_name: Optional[str],
        start_date: str,
        end_date: str,
    ) -> List[Dict]:
        if not self.news_api_key:
            return []
        query = company_name or ticker
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": f"{query} stock earnings",
            "from": start_date,
            "to": end_date,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 100,
            "apiKey": self.news_api_key,
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", [])
            return [
                {"date": a["publishedAt"], "title": a["title"], "description": a.get("description", "")}
                for a in articles
                if a.get("title")
            ]
        except Exception as e:
            logger.warning(f"News API fetch failed for {ticker}: {e}")
            return []

    def _apply_decay(self, series: pd.Series) -> pd.Series:
        """Forward-fill with exponential decay: s[t] = s[t] + decay * s[t-1] if s[t] == 0."""
        out = series.copy()
        for i in range(1, len(out)):
            if out.iloc[i] == 0.0 and out.iloc[i - 1] != 0.0:
                out.iloc[i] = self.sentiment_decay * out.iloc[i - 1]
        return out
