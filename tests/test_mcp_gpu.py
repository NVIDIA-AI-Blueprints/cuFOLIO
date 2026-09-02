# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small end-to-end GPU tests for the optional MCP optimization service."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_optimization.mcp_server.schemas import (
    BacktestPortfolio,
    MeanCvarObjective,
    MeanVarianceObjective,
    PortfolioConstraints,
    PortfolioRunSource,
    ScenarioSettings,
)
from portfolio_optimization.mcp_server.service import PortfolioOptimizationService


def _prices() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    returns = rng.multivariate_normal(
        [0.0005, 0.0003, 0.0004],
        [
            [0.00012, 0.00003, 0.00002],
            [0.00003, 0.00010, 0.00001],
            [0.00002, 0.00001, 0.00011],
        ],
        size=100,
    )
    return pd.DataFrame(
        100.0 * np.exp(np.cumsum(returns, axis=0)),
        index=pd.bdate_range("2025-01-02", periods=100),
        columns=["AAA", "BBB", "CCC"],
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("objective", "scenarios"),
    [
        (
            MeanCvarObjective(
                type="mean_cvar",
                risk_aversion=1.0,
                scale_risk_aversion=False,
            ),
            ScenarioSettings(method="gaussian", count=200, seed=17),
        ),
        (
            MeanVarianceObjective(
                type="mean_variance",
                risk_aversion=1.0,
                scale_risk_aversion=False,
            ),
            None,
        ),
    ],
)
def test_mcp_service_runs_real_cuopt(objective, scenarios):
    pytest.importorskip("cuopt", reason="cuOpt GPU runtime required")
    service = PortfolioOptimizationService(_prices())
    result = service.optimize(
        objective=objective,
        scenarios=scenarios,
        constraints=PortfolioConstraints(
            weight_maximum=0.7,
            cash_minimum=0.0,
            cash_maximum=0.2,
            leverage_limit=1.0,
        ),
        time_limit_seconds=30,
    )

    assert result.solver == "cuOpt"
    assert result.total_allocation == pytest.approx(1.0, abs=1e-5)
    assert result.cash <= 0.2 + 1e-6
    assert all(weight <= 0.7 + 1e-6 for weight in result.weights.values())
    assert result.metrics.solve_time_seconds >= 0.0


@pytest.mark.gpu
def test_seeded_cvar_is_deterministic():
    pytest.importorskip("cuopt", reason="cuOpt GPU runtime required")
    service = PortfolioOptimizationService(_prices())
    kwargs = {
        "objective": MeanCvarObjective(
            type="mean_cvar",
            risk_aversion=1.0,
            scale_risk_aversion=False,
        ),
        "scenarios": ScenarioSettings(method="gaussian", count=200, seed=17),
        "constraints": PortfolioConstraints(
            weight_maximum=0.7,
            cash_maximum=0.2,
            leverage_limit=1.0,
        ),
    }

    first = service.optimize(**kwargs)
    second = service.optimize(**kwargs)

    assert first.weights == pytest.approx(second.weights, abs=1e-8)
    assert first.cash == pytest.approx(second.cash, abs=1e-8)

    run = PortfolioRunSource(
        source="run",
        run_id=first.run_id,
        equity_usd=100_000,
    )
    analysis = service.analyze(
        portfolio=run,
        scenarios=ScenarioSettings(method="gaussian", count=200, seed=17),
    )
    simulation = service.simulate(
        portfolio=run,
        horizon_trading_days=5,
        scenarios=ScenarioSettings(method="gaussian", count=200, seed=17),
        loss_thresholds_usd=[5_000],
    )
    backtest = service.backtest(
        portfolios=[BacktestPortfolio(name="optimized", portfolio=run)],
        method="historical",
    )

    assert analysis.scenario_count == 200
    assert simulation.loss_probabilities[0].threshold_usd == 5_000
    assert set(backtest.metrics) == {"optimized", "equal-weight"}
