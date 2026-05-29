"""
Real-time Plotly/Dash dashboard for model interpretability and monitoring.

Panels:
  1. Portfolio performance vs benchmark
  2. Latent factor trajectories (PCA clusters)
  3. PINN density heatmap (2D projection of latent space)
  4. RL policy convergence (reward / Sharpe over training)
  5. Sector eigen-factor weights (explainability)
  6. Monte Carlo confidence bands
  7. VaR / risk metrics table
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class TradingDashboard:
    """Static figure factory + optional live Dash app."""

    def __init__(self, theme: str = "plotly_dark"):
        self.theme = theme

    # ------------------------------------------------------------------
    # 1. Portfolio performance
    # ------------------------------------------------------------------

    def portfolio_performance(
        self,
        portfolio_values: np.ndarray,
        benchmark_values: Optional[np.ndarray] = None,
        dates: Optional[pd.DatetimeIndex] = None,
        title: str = "Portfolio Performance",
    ) -> go.Figure:
        x = dates if dates is not None else np.arange(len(portfolio_values))
        fig = go.Figure()
        norm = portfolio_values / portfolio_values[0] * 100
        fig.add_trace(go.Scatter(x=x, y=norm, name="Strategy", line=dict(color="#00d4ff", width=2)))
        if benchmark_values is not None:
            bm_norm = benchmark_values[:len(norm)] / benchmark_values[0] * 100
            fig.add_trace(go.Scatter(x=x, y=bm_norm, name="Benchmark", line=dict(color="#ff6b6b", width=1, dash="dash")))
        fig.update_layout(
            title=title, xaxis_title="Date", yaxis_title="Indexed Value (100=start)",
            template=self.theme, height=400,
        )
        return fig

    # ------------------------------------------------------------------
    # 2. Drawdown plot
    # ------------------------------------------------------------------

    def drawdown_chart(
        self,
        returns: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> go.Figure:
        cum = (1 + returns).cumprod()
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak * 100
        x = dates if dates is not None else np.arange(len(dd))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=dd, fill="tozeroy", name="Drawdown",
                                  line=dict(color="#ff4444"), fillcolor="rgba(255,68,68,0.2)"))
        fig.update_layout(title="Drawdown (%)", xaxis_title="Date", yaxis_title="%",
                           template=self.theme, height=300)
        return fig

    # ------------------------------------------------------------------
    # 3. Latent factor scatter (PCA/t-SNE visualization)
    # ------------------------------------------------------------------

    def latent_factor_scatter(
        self,
        latent: np.ndarray,
        labels: Optional[np.ndarray] = None,
        title: str = "Latent Factor Space (PC1 vs PC2)",
    ) -> go.Figure:
        x, y = latent[:, 0], latent[:, 1]
        color = labels if labels is not None else np.arange(len(x))
        fig = go.Figure(go.Scatter(
            x=x, y=y,
            mode="markers",
            marker=dict(color=color, colorscale="Viridis", size=4, opacity=0.7,
                        colorbar=dict(title="Time" if labels is None else "Label")),
        ))
        fig.update_layout(title=title, xaxis_title="Factor 1", yaxis_title="Factor 2",
                           template=self.theme, height=400)
        return fig

    # ------------------------------------------------------------------
    # 4. PINN density heatmap (2D projection)
    # ------------------------------------------------------------------

    def pinn_density_heatmap(
        self,
        pinn_solver,
        t: float = 1.0,
        dim0: int = 0,
        dim1: int = 1,
        n_grid: int = 50,
        title: str = "PINN Probability Density",
    ) -> go.Figure:
        lo, hi = pinn_solver.cfg.lower_bound, pinn_solver.cfg.upper_bound
        x_vals = np.linspace(lo, hi, n_grid)
        y_vals = np.linspace(lo, hi, n_grid)
        xx, yy = np.meshgrid(x_vals, y_vals)

        n_latent = pinn_solver.cfg.n_latent
        z_grid = np.zeros((n_grid * n_grid, n_latent))
        z_grid[:, dim0] = xx.flatten()
        z_grid[:, dim1] = yy.flatten()

        density = pinn_solver.predict_density(z_grid, t).reshape(n_grid, n_grid)

        fig = go.Figure(go.Heatmap(
            x=x_vals, y=y_vals, z=density,
            colorscale="Plasma", showscale=True,
            colorbar=dict(title="p(z,t)"),
        ))
        fig.update_layout(
            title=f"{title} (t={t:.2f})",
            xaxis_title=f"Latent dim {dim0}", yaxis_title=f"Latent dim {dim1}",
            template=self.theme, height=400,
        )
        return fig

    # ------------------------------------------------------------------
    # 5. RL training convergence
    # ------------------------------------------------------------------

    def rl_convergence(
        self,
        reward_history: List[float],
        sharpe_history: Optional[List[float]] = None,
        title: str = "RL Training Convergence",
    ) -> go.Figure:
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Episode Reward", "Rolling Sharpe"])
        # Smooth reward with rolling mean
        r = np.array(reward_history)
        window = max(1, len(r) // 50)
        r_smooth = pd.Series(r).rolling(window).mean().values
        fig.add_trace(go.Scatter(y=r, name="Raw Reward", line=dict(color="#aaa", width=0.5)), row=1, col=1)
        fig.add_trace(go.Scatter(y=r_smooth, name="Smoothed", line=dict(color="#00d4ff", width=2)), row=1, col=1)
        if sharpe_history:
            s = np.array(sharpe_history)
            fig.add_trace(go.Scatter(y=s, name="Sharpe", line=dict(color="#ffd700", width=2)), row=1, col=2)
            fig.add_hline(y=1.5, line_dash="dash", line_color="green", row=1, col=2)
        fig.update_layout(title=title, template=self.theme, height=400)
        return fig

    # ------------------------------------------------------------------
    # 6. Sector eigen-factor loadings
    # ------------------------------------------------------------------

    def factor_loadings_bar(
        self,
        loadings: pd.Series,
        title: str = "Sector Explained Variance",
        top_n: int = 20,
    ) -> go.Figure:
        top = loadings.sort_values(ascending=False).head(top_n)
        colors = ["#00d4ff" if v > 0 else "#ff6b6b" for v in top.values]
        fig = go.Figure(go.Bar(
            x=top.index.tolist(), y=(top.values * 100).tolist(),
            marker_color=colors,
        ))
        fig.update_layout(
            title=title, xaxis_title="Factor", yaxis_title="Explained Var (%)",
            template=self.theme, height=350, xaxis_tickangle=-45,
        )
        return fig

    # ------------------------------------------------------------------
    # 7. Monte Carlo confidence bands
    # ------------------------------------------------------------------

    def mc_confidence_bands(
        self,
        bands: Dict[str, np.ndarray],
        title: str = "Monte Carlo Portfolio Projection",
    ) -> go.Figure:
        fig = go.Figure()
        x = list(range(len(list(bands.values())[0])))
        quantile_order = sorted(bands.keys())
        n = len(quantile_order)

        # Fill bands pairwise from outside in
        colors = ["rgba(0,212,255,0.08)", "rgba(0,212,255,0.12)", "rgba(0,212,255,0.18)"]
        for i, (lo_key, hi_key) in enumerate(zip(quantile_order, reversed(quantile_order))):
            if lo_key >= hi_key:
                break
            c = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=x + x[::-1], y=list(bands[hi_key]) + list(bands[lo_key])[::-1],
                fill="toself", fillcolor=c, line=dict(width=0),
                name=f"{lo_key}–{hi_key}",
            ))

        # Median
        med_key = quantile_order[len(quantile_order) // 2]
        fig.add_trace(go.Scatter(x=x, y=bands[med_key], name="Median",
                                  line=dict(color="#00d4ff", width=2)))

        fig.update_layout(
            title=title, xaxis_title="Days", yaxis_title="Cumulative Return",
            template=self.theme, height=400,
        )
        return fig

    # ------------------------------------------------------------------
    # 8. Full dashboard (combined subplots)
    # ------------------------------------------------------------------

    def full_dashboard(
        self,
        portfolio_values: np.ndarray,
        returns: np.ndarray,
        latent: np.ndarray,
        reward_history: List[float],
        mc_bands: Optional[Dict] = None,
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> go.Figure:
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "Portfolio Performance", "Drawdown",
                "Latent Factor Space",   "RL Reward Convergence",
                "Monte Carlo Bands",     "Factor Distribution",
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}],
            ],
            vertical_spacing=0.10,
            horizontal_spacing=0.08,
        )
        x = dates if dates is not None else np.arange(len(portfolio_values))

        # Row 1: performance + drawdown
        norm = portfolio_values / portfolio_values[0] * 100
        fig.add_trace(go.Scatter(x=list(range(len(norm))), y=norm, name="Portfolio",
                                  line=dict(color="#00d4ff")), row=1, col=1)
        cum = (1 + returns).cumprod()
        dd  = ((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)) * 100
        fig.add_trace(go.Scatter(y=dd, fill="tozeroy", name="Drawdown",
                                  line=dict(color="#ff4444")), row=1, col=2)

        # Row 2: latent scatter + RL reward
        if latent.shape[1] >= 2:
            fig.add_trace(go.Scatter(x=latent[:, 0], y=latent[:, 1], mode="markers",
                                      marker=dict(size=3, color=np.arange(len(latent)),
                                                  colorscale="Viridis"),
                                      name="Latent"), row=2, col=1)
        r_arr = np.array(reward_history)
        r_smooth = pd.Series(r_arr).rolling(max(1, len(r_arr)//50)).mean().values
        fig.add_trace(go.Scatter(y=r_smooth, name="Reward", line=dict(color="#ffd700")), row=2, col=2)

        # Row 3: MC bands + latent histogram
        if mc_bands:
            med_key = sorted(mc_bands.keys())[len(mc_bands) // 2]
            fig.add_trace(go.Scatter(y=mc_bands[med_key], name="MC Median",
                                      line=dict(color="#00ff88")), row=3, col=1)
        fig.add_trace(go.Histogram(x=returns, nbinsx=80, name="Daily Returns",
                                    marker_color="#9b59b6"), row=3, col=2)

        fig.update_layout(
            template=self.theme,
            title="Model Dashboard",
            height=1000,
            showlegend=False,
        )
        return fig

    # ------------------------------------------------------------------
    # Launch live Dash app
    # ------------------------------------------------------------------

    def launch_app(
        self,
        portfolio_values: np.ndarray,
        returns: np.ndarray,
        latent: np.ndarray,
        reward_history: List[float],
        mc_bands: Optional[Dict] = None,
        port: int = 8050,
    ):
        try:
            from dash import Dash, dcc, html
        except ImportError:
            raise ImportError("Install dash: pip install dash")

        app = Dash(__name__)
        fig = self.full_dashboard(portfolio_values, returns, latent, reward_history, mc_bands)
        app.layout = html.Div([
            html.H1("Trading Model Dashboard", style={"color": "white", "textAlign": "center"}),
            dcc.Graph(figure=fig, style={"height": "100vh"}),
        ], style={"backgroundColor": "#1e1e1e"})

        app.run(debug=False, port=port)
