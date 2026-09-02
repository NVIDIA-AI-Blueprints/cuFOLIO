# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict JSON contracts for the optional portfolio-optimization MCP server."""

from __future__ import annotations

import math
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    """Base model that rejects unknown MCP fields."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Regime(StrictModel):
    """Optional historical estimation window."""

    start: str = Field(description="ISO date for the first price observation.")
    end: str = Field(description="ISO date for the last price observation.")


class MeanCvarObjective(StrictModel):
    """Mean-CVaR objective and its risk-specific settings."""

    type: Literal["mean_cvar"]
    risk_aversion: float = Field(
        1.0,
        ge=0.0,
        description="Higher values put more weight on CVaR relative to return.",
    )
    confidence: float = Field(
        0.95,
        gt=0.0,
        lt=1.0,
        description="CVaR confidence as a decimal, for example 0.95.",
    )
    cvar_limit: float | None = Field(
        None,
        gt=0.0,
        description=(
            "Optional hard upper bound on portfolio CVaR. When set, the "
            "underlying optimizer maximizes return subject to this bound."
        ),
    )
    scale_risk_aversion: bool = Field(
        True,
        description="Use the package's upstream risk-aversion scaling heuristic.",
    )


class MeanVarianceObjective(StrictModel):
    """Mean-Variance objective and its risk-specific settings."""

    type: Literal["mean_variance"]
    risk_aversion: float = Field(
        1.0,
        ge=0.0,
        description="Higher values put more weight on variance relative to return.",
    )
    variance_limit: float | None = Field(
        None,
        gt=0.0,
        description="Optional hard upper bound on portfolio variance.",
    )
    scale_risk_aversion: bool = Field(
        True,
        description="Use the package's upstream risk-aversion scaling heuristic.",
    )


Objective = Annotated[
    Union[MeanCvarObjective, MeanVarianceObjective],
    Field(discriminator="type"),
]


class ScenarioSettings(StrictModel):
    """Scenario-generation settings used by Mean-CVaR."""

    method: Literal["kde", "gaussian", "no_fit"] = "kde"
    count: int = Field(10_000, gt=0, le=100_000)
    seed: int | None = Field(
        42,
        ge=0,
        description="Seed for reproducible scenarios; null draws fresh entropy.",
    )
    kde_bandwidth: float = Field(0.01, gt=0.0)
    kde_kernel: Literal[
        "gaussian",
        "tophat",
        "epanechnikov",
        "exponential",
        "linear",
        "cosine",
    ] = "gaussian"


WeightBound = float | dict[str, float]


def _validate_finite_bound(value: WeightBound) -> WeightBound:
    values = value.values() if isinstance(value, dict) else (value,)
    if not all(math.isfinite(float(item)) for item in values):
        raise ValueError("weight bounds must contain only finite numbers")
    return value


class GroupConstraint(StrictModel):
    """Combined weight bounds for a named group of tickers."""

    name: str = Field(min_length=1)
    tickers: list[str] = Field(min_length=1)
    minimum: float = 0.0
    maximum: float = 1.0

    @model_validator(mode="after")
    def valid_range(self) -> "GroupConstraint":
        if self.minimum > self.maximum:
            raise ValueError("group minimum must not exceed group maximum")
        return self


class PortfolioConstraints(StrictModel):
    """JSON-safe subset of the package's shared optimizer parameters."""

    weight_minimum: WeightBound = 0.0
    weight_maximum: WeightBound = 1.0
    cash_minimum: float = Field(0.0, ge=0.0, le=1.0)
    cash_maximum: float = Field(1.0, ge=0.0, le=1.0)
    leverage_limit: float = Field(1.6, gt=0.0)
    turnover_limit: float | None = Field(None, gt=0.0)
    cardinality: int | None = Field(None, gt=0)
    groups: list[GroupConstraint] = Field(default_factory=list)

    _finite_minimum = field_validator("weight_minimum")(_validate_finite_bound)
    _finite_maximum = field_validator("weight_maximum")(_validate_finite_bound)

    @model_validator(mode="after")
    def valid_cash_range(self) -> "PortfolioConstraints":
        if self.cash_minimum > self.cash_maximum:
            raise ValueError("cash_minimum must not exceed cash_maximum")
        return self


class ExistingPortfolio(StrictModel):
    """Current portfolio used only when turnover is constrained."""

    weights: dict[str, float] = Field(min_length=1)
    cash: float = Field(0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def self_financing(self) -> "ExistingPortfolio":
        values = [float(value) for value in self.weights.values()]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("existing portfolio weights must be finite")
        if abs(sum(values) + self.cash - 1.0) > 1e-3:
            raise ValueError("existing portfolio weights and cash must sum to 1")
        return self


class PortfolioWeightsSource(StrictModel):
    """Explicit portfolio allocation used by analysis tools."""

    source: Literal["weights"]
    weights: dict[str, float]
    cash: float = Field(0.0, ge=0.0, le=1.0)
    equity_usd: float | None = Field(None, gt=0.0)

    @model_validator(mode="after")
    def self_financing(self) -> "PortfolioWeightsSource":
        values = [float(value) for value in self.weights.values()]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("portfolio weights must be finite")
        if abs(sum(values) + self.cash - 1.0) > 1e-3:
            raise ValueError("portfolio weights and cash must sum to 1")
        return self


class PortfolioRunSource(StrictModel):
    """Reference to an optimization result from this local server process."""

    source: Literal["run"]
    run_id: str = Field(min_length=1)
    equity_usd: float | None = Field(
        None,
        gt=0.0,
        description="Optional dollar scale for simulation results.",
    )


PortfolioSource = Annotated[
    Union[PortfolioWeightsSource, PortfolioRunSource],
    Field(discriminator="source"),
]
PositiveLossThreshold = Annotated[float, Field(gt=0.0)]
LossThresholds = Annotated[
    list[PositiveLossThreshold],
    Field(max_length=8),
]


class BacktestPortfolio(StrictModel):
    """Named portfolio included in one comparison backtest."""

    name: str = Field(min_length=1)
    portfolio: PortfolioSource


BacktestPortfolios = Annotated[
    list[BacktestPortfolio],
    Field(min_length=1, max_length=4),
]


class DatasetInfoOutput(StrictModel):
    """Bounded description of the startup dataset."""

    summary: str
    source_name: str
    rows: int
    ticker_count: int
    start: str
    end: str
    tickers: list[str]
    dropped_tickers: list[str]
    offset: int
    limit: int
    truncated: bool


class OptimizationMetrics(StrictModel):
    """Stable scalar metrics returned by either optimizer."""

    expected_return_daily: float
    risk_measure: Literal["CVaR", "variance"]
    risk_value_daily: float
    objective_value: float
    solve_time_seconds: float


class OptimizationOutput(StrictModel):
    """JSON-safe optimization result returned over MCP."""

    summary: str
    run_id: str
    objective_type: Literal["mean_cvar", "mean_variance"]
    solver: Literal["cuOpt"]
    regime: Regime
    tickers: list[str]
    weights: dict[str, float]
    cash: float
    total_allocation: float
    metrics: OptimizationMetrics
    scenario_count: int | None


class AnalyzeMetrics(StrictModel):
    """Risk metrics for an existing fixed-weight portfolio."""

    expected_return_daily: float
    expected_return_annualized: float
    variance_daily: float
    volatility_annualized: float
    cvar_daily_signed: float
    cvar_loss_daily: float
    cvar_confidence: float
    hhi: float
    largest_position: str | None
    largest_weight: float


class AnalyzeOutput(StrictModel):
    """Existing-portfolio risk report."""

    summary: str
    weights: dict[str, float]
    cash: float
    equity_usd: float | None
    regime: Regime
    metrics: AnalyzeMetrics
    scenario_count: int


class OutcomePercentile(StrictModel):
    """One simulated return percentile."""

    return_decimal: float
    usd: float | None


class LossProbability(StrictModel):
    """Probability that a simulated loss exceeds a dollar threshold."""

    threshold_usd: float
    probability: float


class SimulationOutput(StrictModel):
    """Fixed-portfolio outcome distribution."""

    summary: str
    weights: dict[str, float]
    cash: float
    equity_usd: float | None
    regime: Regime
    horizon_trading_days: int
    method: str
    scenario_count: int
    percentiles: dict[str, OutcomePercentile]
    mean_return: float
    standard_deviation_return: float
    mean_usd: float | None
    standard_deviation_usd: float | None
    loss_probabilities: list[LossProbability]


class BacktestMetrics(StrictModel):
    """Historical or simulated metrics for one portfolio."""

    mean_return_daily: float | None
    return_annualized: float | None
    sharpe_annualized: float | None
    sortino_annualized: float | None
    max_drawdown: float | None
    ending_value_of_10000: float | None


class BacktestOutput(StrictModel):
    """Comparison backtest result."""

    summary: str
    method: Literal["historical", "kde_simulation", "gaussian_simulation"]
    regime: Regime
    seed: int | None
    metrics: dict[str, BacktestMetrics]
