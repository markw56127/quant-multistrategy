"""
Risk metrics: VaR, CVaR, Sharpe, Sortino, drawdown, confidence intervals.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


class RiskMetrics:
    """Computes a comprehensive suite of risk and performance metrics."""

    def __init__(self, confidence_levels: Tuple[float, ...] = (0.95, 0.99), annual_factor: float = 252.0):
        self.conf_levels   = confidence_levels
        self.annual_factor = annual_factor

    # ------------------------------------------------------------------
    # Core VaR / CVaR
    # ------------------------------------------------------------------

    def var_historical(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Historical simulation VaR."""
        return float(-np.percentile(returns, (1 - confidence) * 100))

    def var_parametric(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Parametric (normal) VaR."""
        mu  = returns.mean()
        sig = returns.std()
        return float(-(mu + stats.norm.ppf(1 - confidence) * sig))

    def var_cornish_fisher(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Cornish-Fisher VaR: adjusts for skewness and kurtosis."""
        mu  = returns.mean()
        sig = returns.std()
        s   = stats.skew(returns)
        k   = stats.kurtosis(returns)  # excess kurtosis
        z   = stats.norm.ppf(1 - confidence)
        z_cf = (z + (z**2 - 1) * s / 6
                  + (z**3 - 3*z) * k / 24
                  - (2*z**3 - 5*z) * s**2 / 36)
        return float(-(mu + z_cf * sig))

    def cvar(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Expected Shortfall (CVaR): mean return in the tail."""
        var = self.var_historical(returns, confidence)
        tail = returns[returns <= -var]
        return float(-tail.mean()) if len(tail) > 0 else var

    def var_monte_carlo(self, mc_paths: np.ndarray, confidence: float = 0.95) -> float:
        """VaR from Monte Carlo simulation paths."""
        total_rets = mc_paths.sum(axis=1)
        return float(-np.percentile(total_rets, (1 - confidence) * 100))

    # ------------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------------

    def sharpe_ratio(self, returns: np.ndarray, risk_free: float = 0.0) -> float:
        excess = returns - risk_free / self.annual_factor
        return float((excess.mean() / (excess.std() + 1e-8)) * np.sqrt(self.annual_factor))

    def sortino_ratio(self, returns: np.ndarray, risk_free: float = 0.0) -> float:
        excess = returns - risk_free / self.annual_factor
        downside = excess[excess < 0]
        downside_std = np.sqrt(np.mean(downside ** 2) + 1e-8)
        return float((excess.mean() / downside_std) * np.sqrt(self.annual_factor))

    def calmar_ratio(self, returns: np.ndarray) -> float:
        ann_ret = returns.mean() * self.annual_factor
        mdd = self.max_drawdown(returns)
        return float(ann_ret / (abs(mdd) + 1e-8))

    def max_drawdown(self, returns: np.ndarray) -> float:
        cum = (1 + returns).cumprod()
        peak = np.maximum.accumulate(cum)
        peak = np.where(peak < 1e-8, 1e-8, peak)
        dd = (cum - peak) / peak
        return float(dd.min())

    def drawdown_series(self, returns: np.ndarray) -> np.ndarray:
        cum  = (1 + returns).cumprod()
        peak = np.maximum.accumulate(cum)
        peak = np.where(peak < 1e-8, 1e-8, peak)
        return (cum - peak) / peak

    def omega_ratio(self, returns: np.ndarray, threshold: float = 0.0) -> float:
        gains  = returns[returns > threshold] - threshold
        losses = threshold - returns[returns < threshold]
        return float(gains.sum() / (losses.sum() + 1e-8))

    def information_ratio(self, returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        active = returns - benchmark_returns[:len(returns)]
        return float((active.mean() / (active.std() + 1e-8)) * np.sqrt(self.annual_factor))

    # ------------------------------------------------------------------
    # Confidence intervals
    # ------------------------------------------------------------------

    def bootstrap_ci(
        self,
        returns: np.ndarray,
        stat_fn: callable,
        n_boot: int = 10_000,
        confidence: float = 0.95,
    ) -> Tuple[float, float, float]:
        """
        Bootstrap confidence interval for any statistic.
        Returns (estimate, lower_ci, upper_ci).
        """
        rng    = np.random.default_rng(42)
        boot   = np.array([stat_fn(rng.choice(returns, size=len(returns), replace=True))
                           for _ in range(n_boot)])
        alpha  = (1 - confidence) / 2
        return float(stat_fn(returns)), float(np.percentile(boot, alpha * 100)), float(np.percentile(boot, (1 - alpha) * 100))

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def full_report(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
        mc_paths: Optional[np.ndarray] = None,
    ) -> Dict:
        r = portfolio_returns
        report = {
            "annualized_return": float(r.mean() * self.annual_factor),
            "annualized_vol":    float(r.std() * np.sqrt(self.annual_factor)),
            "sharpe_ratio":      self.sharpe_ratio(r),
            "sortino_ratio":     self.sortino_ratio(r),
            "calmar_ratio":      self.calmar_ratio(r),
            "omega_ratio":       self.omega_ratio(r),
            "max_drawdown":      self.max_drawdown(r),
            "skewness":          float(stats.skew(r)),
            "kurtosis":          float(stats.kurtosis(r)),
        }

        for conf in self.conf_levels:
            tag = f"{int(conf*100)}"
            report[f"var_{tag}_hist"]    = self.var_historical(r, conf)
            report[f"var_{tag}_param"]   = self.var_parametric(r, conf)
            report[f"var_{tag}_cf"]      = self.var_cornish_fisher(r, conf)
            report[f"cvar_{tag}"]        = self.cvar(r, conf)

        if benchmark_returns is not None:
            report["information_ratio"] = self.information_ratio(r, benchmark_returns)
            report["beta"] = float(np.cov(r, benchmark_returns[:len(r)])[0, 1] / (np.var(benchmark_returns[:len(r)]) + 1e-8))

        if mc_paths is not None:
            for conf in self.conf_levels:
                tag = f"{int(conf*100)}"
                report[f"var_{tag}_mc"] = self.var_monte_carlo(mc_paths, conf)

        return report

    def format_report(self, report: Dict) -> str:
        lines = ["=" * 55, "RISK METRICS REPORT", "=" * 55]
        for k, v in report.items():
            if isinstance(v, float):
                pct_keys = {"annualized_return", "annualized_vol", "max_drawdown"} | \
                           {k for k in report if "var" in k or "cvar" in k}
                if k in pct_keys:
                    lines.append(f"  {k:<35} {v*100:>8.3f}%")
                else:
                    lines.append(f"  {k:<35} {v:>8.4f}")
        lines.append("=" * 55)
        return "\n".join(lines)
