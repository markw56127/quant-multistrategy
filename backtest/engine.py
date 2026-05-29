"""
Parallel backtesting engine.

Runs multiple strategy configurations simultaneously using joblib multiprocessing,
then aggregates results and feeds them back into the RL reward function.
"""

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger

from risk.metrics import RiskMetrics


@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    max_position_size: float = 0.10
    rebalance_freq: int = 5          # days
    allow_short: bool = True
    name: str = "default"


@dataclass
class BacktestResult:
    name: str
    portfolio_values: np.ndarray
    returns: np.ndarray
    weights_history: pd.DataFrame
    turnover_history: np.ndarray
    metrics: Dict[str, float] = field(default_factory=dict)
    trade_log: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def total_return(self) -> float:
        return float((self.portfolio_values[-1] / self.portfolio_values[0]) - 1)

    @property
    def sharpe(self) -> float:
        return self.metrics.get("sharpe_ratio", 0.0)


class BacktestEngine:
    """
    Parallel backtesting engine that evaluates weight sequences against
    historical returns and computes comprehensive performance metrics.

    Backtesting results are structured to feed back into the RL reward:
    - Average Sharpe across runs → normalization baseline
    - Max drawdown → penalty term
    - Information ratio vs benchmark → comparison signal
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
        n_workers: int = 4,
    ):
        self.returns    = returns
        self.benchmark  = benchmark_returns
        self.n_workers  = n_workers
        self.risk       = RiskMetrics()

    # ------------------------------------------------------------------
    # Single strategy backtest
    # ------------------------------------------------------------------

    def run_single(
        self,
        weights: pd.DataFrame,
        config: BacktestConfig,
    ) -> BacktestResult:
        """
        Backtest a weight matrix against historical returns.

        Args:
            weights: (T, n_assets) DataFrame of target portfolio weights,
                     indexed by date. Can be sparse (only rebalance dates).
            config:  Backtest configuration
        """
        all_dates   = self.returns.index
        tickers     = list(self.returns.columns)
        n_assets    = len(tickers)
        cap         = config.initial_capital
        cur_weights = np.zeros(n_assets)

        portfolio_values = [cap]
        daily_returns    = []
        weights_list     = []
        turnover_list    = []
        trade_rows       = []

        # Reindex weights to all trading dates, forward-fill
        w_reindexed = weights.reindex(all_dates).ffill().fillna(0.0)
        w_reindexed = w_reindexed.reindex(columns=tickers, fill_value=0.0)

        for i, date in enumerate(all_dates):
            if i % config.rebalance_freq == 0:
                new_w = w_reindexed.loc[date].values.astype(np.float32)
                # Clip to position limits
                new_w = np.clip(new_w, -config.max_position_size, config.max_position_size)
                if not config.allow_short:
                    new_w = np.clip(new_w, 0, None)
                # Normalize
                abs_sum = np.abs(new_w).sum()
                if abs_sum > 1.0:
                    new_w /= abs_sum

                turnover = np.abs(new_w - cur_weights).sum()
                cost     = cap * turnover * (config.transaction_cost + config.slippage)
                cap     -= cost
                turnover_list.append(turnover)

                if turnover > 0.001:
                    trade_rows.append({"date": date, "turnover": turnover, "cost": cost})

                cur_weights = new_w

            # Daily return
            day_ret = float((cur_weights * self.returns.loc[date].values).sum())
            cap    *= (1 + day_ret)
            portfolio_values.append(cap)
            daily_returns.append(day_ret)
            weights_list.append(cur_weights.copy())

        pv = np.array(portfolio_values)
        r  = np.array(daily_returns)
        bm = self.benchmark.values[:len(r)] if self.benchmark is not None else None
        metrics = self.risk.full_report(r, bm)

        return BacktestResult(
            name=config.name,
            portfolio_values=pv,
            returns=r,
            weights_history=pd.DataFrame(weights_list, index=all_dates, columns=tickers),
            turnover_history=np.array(turnover_list + [0.0] * (len(all_dates) - len(turnover_list))),
            metrics=metrics,
            trade_log=pd.DataFrame(trade_rows),
        )

    # ------------------------------------------------------------------
    # Parallel multi-strategy backtest
    # ------------------------------------------------------------------

    def run_parallel(
        self,
        strategy_configs: List[Tuple[pd.DataFrame, BacktestConfig]],
    ) -> List[BacktestResult]:
        """
        Run multiple (weights, config) pairs in parallel.
        Returns list of BacktestResult sorted by Sharpe ratio.
        """
        logger.info(f"Running {len(strategy_configs)} backtests in parallel ({self.n_workers} workers)")

        results = Parallel(n_jobs=self.n_workers, prefer="threads")(
            delayed(self.run_single)(w, c)
            for w, c in strategy_configs
        )

        results = sorted(results, key=lambda r: r.sharpe, reverse=True)
        logger.info(f"Best Sharpe: {results[0].sharpe:.3f} ({results[0].name})")
        return results

    def cross_compare(
        self,
        results: List[BacktestResult],
        benchmark_name: str = "Benchmark",
    ) -> pd.DataFrame:
        """
        Build a comparison table of all backtest results.
        """
        rows = []
        for res in results:
            row = {"strategy": res.name, "total_return": res.total_return}
            row.update(res.metrics)
            rows.append(row)

        if self.benchmark is not None and len(self.benchmark) > 1:
            # benchmark is already daily returns (see data/pipeline.py)
            bm_r = self.benchmark.dropna().values
            bm_metrics = self.risk.full_report(bm_r)
            bm_row = {"strategy": benchmark_name}
            bm_row.update(bm_metrics)
            rows.append(bm_row)

        return pd.DataFrame(rows).set_index("strategy")

    # ------------------------------------------------------------------
    # RL reward integration
    # ------------------------------------------------------------------

    def compute_rl_reward_adjustment(
        self,
        results: List[BacktestResult],
        target_sharpe: float = 1.5,
    ) -> float:
        """
        Aggregates backtest results into a scalar reward adjustment for RL training.

        Logic:
          - If avg Sharpe > target: positive bonus → encourages this region of policy space
          - If max drawdown > 20%: penalty  → discourages high-risk policies
          - Feeds back into rl/environment.py reward normalization
        """
        sharpes  = [r.sharpe for r in results]
        avg_sharpe = np.mean(sharpes)
        best_dd    = min(r.metrics.get("max_drawdown", 0.0) for r in results)

        sharpe_bonus = np.clip(avg_sharpe - target_sharpe, -1.0, 1.0)
        dd_penalty   = np.clip(abs(best_dd) - 0.20, 0.0, 0.5) * 2.0  # scale to [0, 1]

        return float(sharpe_bonus - dd_penalty)

    def walk_forward_validation(
        self,
        weight_fn: Callable[[pd.DataFrame], pd.DataFrame],
        config: BacktestConfig,
        n_folds: int = 5,
        test_ratio: float = 0.2,
    ) -> List[BacktestResult]:
        """
        Walk-forward out-of-sample validation.
        Splits returns into n_folds train/test windows and evaluates weight_fn on each test fold.
        """
        T = len(self.returns)
        fold_size = T // n_folds
        results = []

        for fold in range(n_folds):
            test_start = fold * fold_size
            test_end   = min(test_start + fold_size, T)
            train_end  = test_start

            train_returns = self.returns.iloc[:train_end] if train_end > 0 else self.returns.iloc[:1]
            test_returns  = self.returns.iloc[test_start:test_end]

            try:
                weights = weight_fn(train_returns)
                cfg = copy.copy(config)
                cfg.name = f"fold_{fold+1}"
                engine = BacktestEngine(test_returns, self.benchmark)
                result = engine.run_single(weights, cfg)
                results.append(result)
                logger.info(f"Fold {fold+1}/{n_folds}: Sharpe={result.sharpe:.3f}, Return={result.total_return*100:.1f}%")
            except Exception as e:
                logger.warning(f"Walk-forward fold {fold+1} failed: {e}")

        return results
