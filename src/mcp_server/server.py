# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local stdio MCP server for the portfolio-optimization package."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from typing import Annotated, Any, Callable, Literal

import anyio
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from portfolio_optimization import version

from . import _stdio as _stdio  # noqa: F401 -- sets MPLBACKEND before optimizers
from .schemas import (
    AnalyzeOutput,
    BacktestOutput,
    BacktestPortfolios,
    DatasetInfoOutput,
    ExistingPortfolio,
    LossThresholds,
    Objective,
    OptimizationOutput,
    PortfolioConstraints,
    PortfolioSource,
    Regime,
    ScenarioSettings,
    SimulationOutput,
)
from .service import PortfolioOptimizationService

_LOG = logging.getLogger("portfolio_optimization.mcp")


@dataclass(frozen=True)
class ApplicationContext:
    """Application state shared by both local MCP tools."""

    service: PortfolioOptimizationService


async def _invoke(operation: Callable[[], Any]):
    try:
        return await anyio.to_thread.run_sync(
            operation,
            abandon_on_cancel=True,
        )
    except ValueError as exc:
        raise ToolError(f"Invalid portfolio request: {exc}") from None
    except RuntimeError as exc:
        raise ToolError(f"Portfolio operation failed: {exc}") from None
    except Exception:
        _LOG.exception("unexpected portfolio MCP failure")
        raise RuntimeError(
            "Portfolio operation failed; inspect the local server logs."
        ) from None


def create_server(
    service: PortfolioOptimizationService,
) -> MCPServer[ApplicationContext]:
    """Create a five-tool MCP server around one startup dataset."""

    @asynccontextmanager
    async def lifespan(
        server: MCPServer[ApplicationContext],
    ) -> AsyncIterator[ApplicationContext]:
        del server
        yield ApplicationContext(service=service)

    mcp = MCPServer(
        "portfolio-optimization",
        version=version,
        lifespan=lifespan,
    )

    @mcp.tool(
        name="portfolio_dataset_info",
        title="Inspect portfolio dataset",
        description=(
            "Describe the trusted CSV or Parquet dataset loaded when this local "
            "server started. Use query, offset, and limit to inspect available "
            "ticker symbols before optimization. This tool does not load files "
            "and does not fetch market data."
        ),
        annotations=ToolAnnotations(
            read_only_hint=True,
            idempotent_hint=True,
            open_world_hint=False,
        ),
        structured_output=True,
    )
    async def portfolio_dataset_info(
        query: str | None = None,
        offset: Annotated[int, Field(ge=0)] = 0,
        limit: Annotated[int, Field(ge=1, le=500)] = 100,
    ) -> DatasetInfoOutput:
        return service.dataset_info(query=query, offset=offset, limit=limit)

    @mcp.tool(
        name="portfolio_optimize",
        title="Optimize a portfolio with cuOpt",
        description=(
            "Run GPU-only portfolio optimization against the price dataset "
            "loaded at server startup. objective.type='mean_cvar' generates "
            "KDE, Gaussian, or historical scenarios and solves Mean-CVaR; "
            "objective.type='mean_variance' solves Markowitz Mean-Variance. "
            "Weights and limits are decimals. The tool supports per-ticker "
            "weight bounds, cash bounds, leverage, named groups, optional "
            "turnover from an existing portfolio, CVaR/variance hard limits, "
            "and CVaR cardinality. It never falls back to a CPU solver."
            " Mean-CVaR is limited to 100,000 scenarios and 10,000,000 "
            "asset-scenario values per call."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
        structured_output=True,
    )
    async def portfolio_optimize(
        objective: Objective,
        tickers: list[str] | None = None,
        regime: Regime | None = None,
        constraints: PortfolioConstraints | None = None,
        existing_portfolio: ExistingPortfolio | None = None,
        scenarios: ScenarioSettings | None = None,
        time_limit_seconds: Annotated[float, Field(gt=0.0, le=3600.0)] = 60.0,
    ) -> OptimizationOutput:
        operation = partial(
            service.optimize,
            objective=objective,
            tickers=tickers,
            regime=regime,
            constraints=constraints,
            existing_portfolio=existing_portfolio,
            scenarios=scenarios,
            time_limit_seconds=time_limit_seconds,
        )
        return await _invoke(operation)

    @mcp.tool(
        name="portfolio_analyze",
        title="Analyze a portfolio",
        description=(
            "Evaluate an explicit portfolio or a prior portfolio_optimize run. "
            "Returns expected return, variance, volatility, signed CVaR loss, "
            "and concentration without changing the portfolio."
        ),
        annotations=ToolAnnotations(
            read_only_hint=True,
            idempotent_hint=True,
            open_world_hint=False,
        ),
        structured_output=True,
    )
    async def portfolio_analyze(
        portfolio: PortfolioSource,
        regime: Regime | None = None,
        confidence: Annotated[float, Field(ge=0.5, lt=1.0)] = 0.95,
        scenarios: ScenarioSettings | None = None,
    ) -> AnalyzeOutput:
        return await _invoke(
            partial(
                service.analyze,
                portfolio=portfolio,
                regime=regime,
                confidence=confidence,
                scenarios=scenarios,
            )
        )

    @mcp.tool(
        name="portfolio_simulate",
        title="Simulate portfolio outcomes",
        description=(
            "Generate a fixed-portfolio return distribution from an explicit "
            "allocation or prior optimization run. Returns percentiles, mean, "
            "spread, and optional dollar-loss probabilities. Multi-day "
            "horizons compound independently sampled daily log returns."
        ),
        annotations=ToolAnnotations(
            read_only_hint=True,
            idempotent_hint=True,
            open_world_hint=False,
        ),
        structured_output=True,
    )
    async def portfolio_simulate(
        portfolio: PortfolioSource,
        regime: Regime | None = None,
        horizon_trading_days: Annotated[int, Field(ge=1, le=252)] = 1,
        scenarios: ScenarioSettings | None = None,
        loss_thresholds_usd: LossThresholds | None = None,
    ) -> SimulationOutput:
        return await _invoke(
            partial(
                service.simulate,
                portfolio=portfolio,
                regime=regime,
                horizon_trading_days=horizon_trading_days,
                scenarios=scenarios,
                loss_thresholds_usd=loss_thresholds_usd,
            )
        )

    @mcp.tool(
        name="portfolio_backtest",
        title="Backtest portfolios",
        description=(
            "Backtest one to four explicit portfolios or prior optimization "
            "runs against an automatic equal-weight benchmark. Supports "
            "historical, KDE-simulated, and Gaussian-simulated returns."
        ),
        annotations=ToolAnnotations(
            read_only_hint=True,
            idempotent_hint=True,
            open_world_hint=False,
        ),
        structured_output=True,
    )
    async def portfolio_backtest(
        portfolios: BacktestPortfolios,
        tickers: list[str] | None = None,
        regime: Regime | None = None,
        method: Literal[
            "historical",
            "kde_simulation",
            "gaussian_simulation",
        ] = "historical",
        risk_free_rate_annualized: Annotated[float, Field(gt=-1.0)] = 0.0,
        seed: Annotated[int, Field(ge=0)] | None = 42,
    ) -> BacktestOutput:
        return await _invoke(
            partial(
                service.backtest,
                portfolios=portfolios,
                tickers=tickers,
                regime=regime,
                method=method,
                risk_free_rate_annualized=risk_free_rate_annualized,
                seed=seed,
            )
        )

    return mcp


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run portfolio-optimization as a local stdio MCP server against "
            "one trusted startup dataset."
        )
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Trusted CSV or Parquet price table; dates in the first/index column.",
    )
    return parser


def main() -> None:
    """Load the startup dataset and run the MCP stdio transport."""

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = _parser().parse_args()
    try:
        service = PortfolioOptimizationService.from_path(args.data)
    except (OSError, ValueError) as exc:
        _parser().error(str(exc))
    create_server(service).run(transport="stdio")


if __name__ == "__main__":
    main()
