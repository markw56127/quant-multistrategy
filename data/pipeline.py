from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from .ingestion import DataIngestion
from .technical_indicators import TechnicalIndicators
from .sentiment import SentimentEngine


class DataPipeline:
    """
    Orchestrates the full multimodal data pipeline:
      1. OHLCV ingestion
      2. Technical indicator computation
      3. LLM-based sentiment scoring
      4. Aligned feature matrix assembly
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.ingestion = DataIngestion(cache_dir=self.cfg["data"]["cache_dir"])
        self.indicators = TechnicalIndicators()
        self.sentiment = SentimentEngine(
            provider=self.cfg["sentiment"]["provider"],
            claude_model=self.cfg["sentiment"]["claude_model"],
            sentiment_decay=self.cfg["sentiment"]["sentiment_decay"],
            batch_size=self.cfg["sentiment"]["batch_size"],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        force_refresh: bool = False,
        include_sentiment: bool = True,
    ) -> "PipelineOutput":
        """
        Execute the full pipeline and return a PipelineOutput dataclass.
        """
        cfg_d = self.cfg["data"]
        tickers_by_sector = cfg_d["tickers"]
        start = cfg_d["start_date"]
        end = cfg_d["end_date"]
        interval = cfg_d["interval"]

        # 1. Fetch OHLCV
        logger.info("Step 1/4: Fetching OHLCV data")
        ohlcv = self.ingestion.fetch_universe(
            tickers_by_sector, start, end, interval, force_refresh=force_refresh
        )

        # Sector ETFs as macro anchors
        etf_ohlcv = self.ingestion.fetch_sector_etfs(cfg_d["sector_etfs"], start, end, interval)
        benchmark_ohlcv = self.ingestion.fetch_benchmark(cfg_d["benchmark"], start, end, interval)

        # 2. Build return matrices
        logger.info("Step 2/4: Computing returns and technical indicators")
        all_ohlcv = {**ohlcv, **etf_ohlcv}
        all_returns = self.ingestion.build_return_matrix(all_ohlcv)
        sector_returns = self.ingestion.build_sector_return_matrix(all_ohlcv, tickers_by_sector)
        benchmark_returns = benchmark_ohlcv["Close"].pct_change().dropna() if not benchmark_ohlcv.empty else pd.Series()

        # 3. Technical indicators
        ind_data = self.indicators.compute_universe(ohlcv)
        feature_matrix = self.indicators.build_feature_matrix(ind_data, all_returns)

        # 4. Earnings data
        all_tickers = [t for ts in tickers_by_sector.values() for t in ts]
        earnings = self.ingestion.fetch_earnings_calendar(all_tickers, start, end)

        # 5. Sentiment (optional – slow due to API calls)
        sentiment_matrix = pd.DataFrame()
        if include_sentiment:
            logger.info("Step 4/4: Computing LLM sentiment scores")
            sentiment_matrix = self.sentiment.build_sentiment_matrix(all_tickers, start, end)
        else:
            logger.info("Step 4/4: Skipping sentiment (include_sentiment=False)")

        return PipelineOutput(
            ohlcv=ohlcv,
            etf_ohlcv=etf_ohlcv,
            benchmark_returns=benchmark_returns,
            all_returns=all_returns,
            sector_returns=sector_returns,
            feature_matrix=feature_matrix,
            indicator_data=ind_data,
            sentiment_matrix=sentiment_matrix,
            earnings=earnings,
            tickers_by_sector=tickers_by_sector,
            config=self.cfg,
        )

    def get_aligned_input(
        self,
        output: "PipelineOutput",
        normalize: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns (X, y) where:
          X = (feature_matrix + sentiment) aligned on business days
          y = next-day log returns for all tickers
        """
        returns = output.all_returns
        features = output.feature_matrix

        # Align to same dates
        common_idx = features.index.intersection(returns.index)
        features = features.loc[common_idx]
        returns_aligned = returns.loc[common_idx]

        # Add sentiment
        if not output.sentiment_matrix.empty:
            sent = output.sentiment_matrix.reindex(common_idx).ffill().fillna(0.0)
            sent.columns = [f"sent_{c}" for c in sent.columns]
            features = pd.concat([features, sent], axis=1)

        # Next-day return as target
        y = returns_aligned.shift(-1).dropna()
        X = features.loc[y.index]

        if normalize:
            X = (X - X.mean()) / (X.std() + 1e-8)

        return X, y


class PipelineOutput:
    """Container for all outputs of the data pipeline."""

    def __init__(
        self,
        ohlcv: Dict[str, pd.DataFrame],
        etf_ohlcv: Dict[str, pd.DataFrame],
        benchmark_returns: pd.Series,
        all_returns: pd.DataFrame,
        sector_returns: Dict[str, pd.DataFrame],
        feature_matrix: pd.DataFrame,
        indicator_data: Dict[str, pd.DataFrame],
        sentiment_matrix: pd.DataFrame,
        earnings: pd.DataFrame,
        tickers_by_sector: Dict[str, List[str]],
        config: dict,
    ):
        self.ohlcv = ohlcv
        self.etf_ohlcv = etf_ohlcv
        self.benchmark_returns = benchmark_returns
        self.all_returns = all_returns
        self.sector_returns = sector_returns
        self.feature_matrix = feature_matrix
        self.indicator_data = indicator_data
        self.sentiment_matrix = sentiment_matrix
        self.earnings = earnings
        self.tickers_by_sector = tickers_by_sector
        self.config = config

    @property
    def all_tickers(self) -> List[str]:
        return [t for ts in self.tickers_by_sector.values() for t in ts]

    @property
    def n_tickers(self) -> int:
        return len(self.all_tickers)

    def __repr__(self) -> str:
        return (
            f"PipelineOutput("
            f"tickers={self.n_tickers}, "
            f"dates={len(self.all_returns)}, "
            f"features={self.feature_matrix.shape[1]})"
        )
