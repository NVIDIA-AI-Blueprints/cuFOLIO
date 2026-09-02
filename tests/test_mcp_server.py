# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the optional local stdio MCP integration."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from mcp import Client
from pydantic import TypeAdapter, ValidationError

from portfolio_optimization.mcp_server._stdio import suppress_stdout
from portfolio_optimization.mcp_server.schemas import (
    BacktestPortfolio,
    ExistingPortfolio,
    MeanCvarObjective,
    MeanVarianceObjective,
    Objective,
    PortfolioConstraints,
    PortfolioRunSource,
    PortfolioWeightsSource,
    ScenarioSettings,
)
from portfolio_optimization.mcp_server.server import create_server
from portfolio_optimization.mcp_server.service import PortfolioOptimizationService
from portfolio_optimization.portfolio import Portfolio


@pytest.fixture()
def prices() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0004, 0.01, size=(80, 3))
    values = 100.0 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(
        values,
        index=pd.bdate_range("2025-01-02", periods=80),
        columns=["AAA", "BBB", "CCC"],
    )


def test_objective_schema_is_discriminated():
    adapter = TypeAdapter(Objective)
    assert isinstance(
        adapter.validate_python({"type": "mean_cvar"}),
        MeanCvarObjective,
    )
    assert isinstance(
        adapter.validate_python({"type": "mean_variance"}),
        MeanVarianceObjective,
    )
    with pytest.raises(ValidationError):
        adapter.validate_python({"type": "unknown"})


def test_startup_csv_and_dataset_info(tmp_path, prices):
    path = tmp_path / "prices.csv"
    prices.to_csv(path)
    service = PortfolioOptimizationService.from_path(path)

    info = service.dataset_info(query="B", limit=10)

    assert info.source_name == "prices.csv"
    assert info.rows == 80
    assert info.ticker_count == 3
    assert info.tickers == ["BBB"]
    assert info.truncated is False


@pytest.mark.parametrize(
    "bad_value, message",
    [
        (0.0, "finite, positive"),
        (np.inf, "finite, positive"),
    ],
)
def test_startup_dataset_rejects_invalid_prices(prices, bad_value, message):
    frame = prices.copy()
    frame.iloc[0, 0] = bad_value
    with pytest.raises(ValueError, match=message):
        PortfolioOptimizationService(frame)


def test_startup_dataset_excludes_incomplete_tickers(prices):
    frame = prices.copy()
    frame.iloc[0, 0] = np.nan

    service = PortfolioOptimizationService(frame)
    info = service.dataset_info()

    assert info.tickers == ["BBB", "CCC"]
    assert info.dropped_tickers == ["AAA"]
    assert "1 incomplete ticker" in info.summary


def test_cvar_request_maps_to_upstream_models(monkeypatch, prices):
    from portfolio_optimization.mcp_server import service as service_module

    captured = {}

    def fake_generate(returns, settings):
        captured["scenario_settings"] = settings
        return {
            **returns,
            "cvar_data": SimpleNamespace(p=np.full(settings.num_scen, 1.0)),
        }

    class FakeCVaR:
        def __init__(
            self,
            returns_dict,
            cvar_params,
            api_settings=None,
            existing_portfolio=None,
        ):
            captured["returns"] = returns_dict
            captured["params"] = cvar_params
            captured["api"] = api_settings
            captured["existing"] = existing_portfolio
            self.tickers = returns_dict["tickers"]

        def solve_optimization_problem(self, solver_settings, print_results=True):
            captured["solver_settings"] = solver_settings
            captured["print_results"] = print_results
            return (
                pd.Series(
                    {
                        "return": 0.001,
                        "CVaR": 0.02,
                        "obj": 0.019,
                        "solve time": 0.01,
                    }
                ),
                Portfolio(
                    name="fake",
                    tickers=self.tickers,
                    weights=np.array([0.5, 0.4, 0.0]),
                    cash=0.1,
                ),
            )

    monkeypatch.setattr(service_module, "generate_cvar_data", fake_generate)
    monkeypatch.setattr(service_module, "CVaR", FakeCVaR)
    service = PortfolioOptimizationService(prices)
    result = service.optimize(
        objective=MeanCvarObjective(
            type="mean_cvar",
            risk_aversion=2.0,
            confidence=0.975,
            scale_risk_aversion=False,
        ),
        constraints=PortfolioConstraints(
            weight_maximum={"AAA": 0.6, "others": 0.5},
            cash_maximum=0.2,
            leverage_limit=1.0,
            turnover_limit=0.3,
            groups=[
                {
                    "name": "pair",
                    "tickers": ["AAA", "BBB"],
                    "maximum": 0.9,
                }
            ],
        ),
        existing_portfolio=ExistingPortfolio(
            weights={"AAA": 0.4, "BBB": 0.4, "CCC": 0.1},
            cash=0.1,
        ),
        scenarios=ScenarioSettings(method="gaussian", count=200, seed=11),
        time_limit_seconds=12,
    )

    params = captured["params"]
    assert params.risk_aversion == 2.0
    assert params.confidence == 0.975
    assert params.T_tar == 0.3
    assert params.group_constraints[0]["group_name"] == "pair"
    assert captured["api"].api == "cuopt_python"
    assert captured["api"].scale_risk_aversion is False
    assert captured["existing"].cash == 0.1
    assert captured["scenario_settings"].seed == 11
    assert captured["solver_settings"] == {"time_limit": 12}
    assert captured["print_results"] is False
    assert result.scenario_count == 200
    assert result.total_allocation == pytest.approx(1.0)
    assert result.run_id.startswith("opt_")
    resolved = service._portfolio_values(
        PortfolioRunSource(source="run", run_id=result.run_id)
    )
    assert resolved[0] == pytest.approx(result.weights)
    assert resolved[1] == pytest.approx(result.cash)


def test_mean_variance_request_maps_without_scenarios(monkeypatch, prices):
    from portfolio_optimization.mcp_server import service as service_module

    captured = {}

    class FakeMeanVariance:
        def __init__(
            self,
            returns_dict,
            mean_variance_params,
            api_settings=None,
            existing_portfolio=None,
        ):
            captured["params"] = mean_variance_params
            captured["api"] = api_settings
            self.tickers = returns_dict["tickers"]

        def solve_optimization_problem(self, solver_settings, print_results=True):
            return (
                pd.Series(
                    {
                        "return": 0.0008,
                        "variance": 0.0001,
                        "obj": -0.0007,
                        "solve time": 0.02,
                    }
                ),
                Portfolio(
                    name="fake",
                    tickers=self.tickers,
                    weights=np.array([0.45, 0.45, 0.0]),
                    cash=0.1,
                ),
            )

    monkeypatch.setattr(service_module, "MeanVariance", FakeMeanVariance)
    service = PortfolioOptimizationService(prices)
    result = service.optimize(
        objective=MeanVarianceObjective(
            type="mean_variance",
            variance_limit=0.01,
            scale_risk_aversion=False,
        ),
        constraints=PortfolioConstraints(cash_maximum=0.2),
    )

    assert captured["params"].var_limit == 0.01
    assert captured["api"].api == "cuopt_python"
    assert result.metrics.risk_measure == "variance"
    assert result.scenario_count is None
    with pytest.raises(ValueError, match="scenarios apply only"):
        service.optimize(
            objective=MeanVarianceObjective(type="mean_variance"),
            scenarios=ScenarioSettings(method="gaussian"),
        )


def test_turnover_requires_existing_portfolio(prices):
    service = PortfolioOptimizationService(prices)
    with pytest.raises(ValueError, match="requires existing_portfolio"):
        service.optimize(
            objective=MeanCvarObjective(type="mean_cvar"),
            constraints=PortfolioConstraints(turnover_limit=0.2),
        )


def test_analysis_simulation_and_backtest_workflow(prices):
    service = PortfolioOptimizationService(prices)
    source = PortfolioWeightsSource(
        source="weights",
        weights={"AAA": 0.5, "BBB": 0.3, "CCC": 0.1},
        cash=0.1,
        equity_usd=100_000,
    )
    scenario_settings = ScenarioSettings(method="gaussian", count=200, seed=9)

    analysis = service.analyze(
        portfolio=source,
        scenarios=scenario_settings,
    )
    assert analysis.metrics.cvar_confidence == 0.95
    assert analysis.scenario_count == 200
    assert analysis.metrics.largest_position == "AAA"

    simulation = service.simulate(
        portfolio=source,
        horizon_trading_days=5,
        scenarios=scenario_settings,
        loss_thresholds_usd=[1_000, 5_000],
    )
    assert simulation.scenario_count == 200
    assert set(simulation.percentiles) == {
        "p1",
        "p5",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
    }
    assert len(simulation.loss_probabilities) == 2
    assert simulation.mean_usd is not None

    backtest = service.backtest(
        portfolios=[BacktestPortfolio(name="candidate", portfolio=source)],
        method="historical",
    )
    assert backtest.seed is None
    assert set(backtest.metrics) == {"candidate", "equal-weight"}
    assert backtest.metrics["candidate"].ending_value_of_10000 is not None


def test_run_cache_is_bounded_and_rejects_unknown_handles(prices):
    service = PortfolioOptimizationService(prices)
    run_ids = [
        service._store_run(
            {"AAA": 0.5 + index / 1_000},
            0.5 - index / 1_000,
            ["AAA", "BBB"],
        )
        for index in range(65)
    ]
    assert len(service._runs) == 64
    with pytest.raises(ValueError, match="unknown or was evicted"):
        service._portfolio_values(PortfolioRunSource(source="run", run_id=run_ids[0]))
    assert service._portfolio_values(
        PortfolioRunSource(source="run", run_id=run_ids[-1])
    )[0]["AAA"] == pytest.approx(0.564)


def test_scenario_workload_is_bounded(monkeypatch, prices):
    service = PortfolioOptimizationService(prices)
    monkeypatch.setattr(service, "_MAX_ASSET_SCENARIO_VALUES", 100)
    with pytest.raises(ValueError, match="assets × scenarios"):
        service.optimize(
            objective=MeanCvarObjective(type="mean_cvar"),
            scenarios=ScenarioSettings(method="gaussian", count=100),
        )
    with pytest.raises(ValidationError):
        ScenarioSettings(count=100_001)


def test_symbol_normalization_rejects_duplicates(prices):
    service = PortfolioOptimizationService(prices)
    with pytest.raises(ValueError, match="selected tickers contain duplicates"):
        service.optimize(
            objective=MeanCvarObjective(type="mean_cvar"),
            tickers=["aaa", "AAA"],
        )
    with pytest.raises(ValueError, match="weight bounds contain duplicate"):
        service.optimize(
            objective=MeanCvarObjective(type="mean_cvar"),
            constraints=PortfolioConstraints(
                weight_maximum={"aaa": 0.5, "AAA": 0.6, "others": 1.0}
            ),
        )


def test_native_stdout_guard_suppresses_python_and_fd_writes(capfd):
    with suppress_stdout():
        print("python solver noise")
        os.write(1, b"native solver noise\n")
    print("protocol output")
    captured = capfd.readouterr()
    assert "solver noise" not in captured.out
    assert "protocol output" in captured.out


def test_protocol_exposes_five_workflow_tools(prices):
    async def exercise():
        service = PortfolioOptimizationService(prices, source_name="test.csv")
        async with Client(create_server(service), raise_exceptions=True) as client:
            tools = await client.list_tools()
            assert {tool.name for tool in tools.tools} == {
                "portfolio_dataset_info",
                "portfolio_optimize",
                "portfolio_analyze",
                "portfolio_simulate",
                "portfolio_backtest",
            }
            optimize = next(
                tool for tool in tools.tools if tool.name == "portfolio_optimize"
            )
            assert (
                optimize.input_schema["properties"]["objective"]["discriminator"][
                    "propertyName"
                ]
                == "type"
            )
            result = await client.call_tool(
                "portfolio_dataset_info",
                {"query": "AA", "limit": 10},
            )
            assert not result.is_error
            assert result.structured_content["tickers"] == ["AAA"]
            failed = await client.call_tool(
                "portfolio_optimize",
                {
                    "objective": {"type": "mean_cvar"},
                    "constraints": {"turnover_limit": 0.2},
                },
            )
            assert failed.is_error
            error_text = " ".join(item.text for item in failed.content)
            assert "turnover_limit requires existing_portfolio" in error_text
            assert "Traceback" not in error_text

    asyncio.run(exercise())
