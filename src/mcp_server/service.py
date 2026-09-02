# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin adapter from JSON-safe MCP contracts to the package's cuOpt APIs."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_optimization.backtest import portfolio_backtester
from portfolio_optimization.cvar_optimizer import CVaR
from portfolio_optimization.cvar_parameters import CvarParameters
from portfolio_optimization.cvar_utils import (
    evaluate_portfolio_performance,
    generate_cvar_data,
)
from portfolio_optimization.mean_variance_optimizer import MeanVariance
from portfolio_optimization.mean_variance_parameters import MeanVarianceParameters
from portfolio_optimization.portfolio import Portfolio
from portfolio_optimization.settings import (
    ApiSettings,
    KDESettings,
    ReturnsComputeSettings,
    ScenarioGenerationSettings,
)
from portfolio_optimization.utils import calculate_returns

from ._stdio import suppress_stdout
from .schemas import (
    AnalyzeMetrics,
    AnalyzeOutput,
    BacktestMetrics,
    BacktestOutput,
    BacktestPortfolio,
    DatasetInfoOutput,
    ExistingPortfolio,
    LossProbability,
    LossThresholds,
    MeanCvarObjective,
    Objective,
    OptimizationMetrics,
    OptimizationOutput,
    OutcomePercentile,
    PortfolioConstraints,
    PortfolioRunSource,
    PortfolioSource,
    PortfolioWeightsSource,
    Regime,
    ScenarioSettings,
    SimulationOutput,
    WeightBound,
)


@dataclass(frozen=True)
class _RunRecord:
    weights: dict[str, float]
    cash: float
    tickers: list[str]


class PortfolioOptimizationService:
    """Own one trusted startup dataset and execute GPU-only optimizations."""

    _MAX_ASSET_SCENARIO_VALUES = 10_000_000

    def __init__(self, prices: pd.DataFrame, *, source_name: str = "in-memory"):
        if not isinstance(prices, pd.DataFrame):
            raise ValueError("startup dataset must be a price table")
        incomplete = prices.isna().any(axis=0).to_numpy()
        self.dropped_tickers = [
            str(prices.columns[index]).strip().upper()
            for index, drop in enumerate(incomplete)
            if bool(drop)
        ]
        complete_prices = prices.loc[:, ~incomplete]
        self._prices = self._validate_prices(complete_prices)
        self.source_name = source_name
        self._runs: OrderedDict[str, _RunRecord] = OrderedDict()
        self._run_lock = threading.Lock()

    @classmethod
    def from_path(cls, path: str | Path) -> "PortfolioOptimizationService":
        """Load a trusted CSV or Parquet file supplied at server startup."""

        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"dataset does not exist: {source}")
        suffix = source.suffix.lower()
        if suffix == ".csv":
            frame = pd.read_csv(source, index_col=0)
        elif suffix == ".parquet":
            frame = pd.read_parquet(source)
        else:
            raise ValueError("startup dataset must be a CSV or Parquet file")
        return cls(frame, source_name=source.name)

    @staticmethod
    def _validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(prices, pd.DataFrame) or prices.empty:
            raise ValueError("startup dataset must be a non-empty price table")
        if len(prices.index) < 3:
            raise ValueError("startup dataset needs at least three price rows")
        try:
            parsed = pd.DatetimeIndex(pd.to_datetime(prices.index, errors="raise"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "startup dataset index must contain ISO-like dates"
            ) from exc
        if parsed.tz is not None:
            parsed = parsed.tz_convert(None)
        parsed = parsed.normalize()
        if parsed.has_duplicates:
            raise ValueError("startup dataset contains duplicate dates")

        tickers = [str(column).strip().upper() for column in prices.columns]
        if any(not ticker for ticker in tickers):
            raise ValueError("startup dataset contains a blank ticker")
        if len(tickers) != len(set(tickers)):
            raise ValueError("startup dataset contains duplicate tickers")

        try:
            normalized = prices.copy().astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("startup prices must be numeric") from exc
        normalized.index = parsed
        normalized.columns = tickers
        normalized = normalized.sort_index()
        values = normalized.to_numpy(dtype=float)
        if not np.isfinite(values).all() or bool((values <= 0.0).any()):
            raise ValueError("startup prices must be finite, positive, and non-missing")
        return normalized

    def dataset_info(
        self,
        *,
        query: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> DatasetInfoOutput:
        """Return bounded metadata for the startup dataset."""

        candidates = list(self._prices.columns)
        if query:
            needle = query.strip().upper()
            candidates = [ticker for ticker in candidates if needle in ticker]
        page = candidates[offset : offset + limit]
        dropped_note = (
            f" {len(self.dropped_tickers):,} incomplete ticker column(s) were "
            "excluded at startup."
            if self.dropped_tickers
            else ""
        )
        return DatasetInfoOutput(
            summary=(
                f"{self.source_name} contains {len(self._prices):,} price rows "
                f"for {len(self._prices.columns):,} tickers from "
                f"{self._prices.index[0].date().isoformat()} through "
                f"{self._prices.index[-1].date().isoformat()}."
                f"{dropped_note}"
            ),
            source_name=self.source_name,
            rows=len(self._prices),
            ticker_count=len(self._prices.columns),
            start=self._prices.index[0].date().isoformat(),
            end=self._prices.index[-1].date().isoformat(),
            tickers=page,
            dropped_tickers=self.dropped_tickers,
            offset=offset,
            limit=limit,
            truncated=offset + len(page) < len(candidates),
        )

    def _store_run(
        self,
        weights: dict[str, float],
        cash: float,
        tickers: list[str],
    ) -> str:
        encoded = json.dumps(
            {
                "weights": weights,
                "cash": cash,
                "tickers": tickers,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        run_id = "opt_" + hashlib.sha256(encoded).hexdigest()[:12]
        with self._run_lock:
            self._runs[run_id] = _RunRecord(
                weights=dict(weights),
                cash=float(cash),
                tickers=list(tickers),
            )
            self._runs.move_to_end(run_id)
            while len(self._runs) > 64:
                self._runs.popitem(last=False)
        return run_id

    def _portfolio_values(
        self,
        source: PortfolioSource,
    ) -> tuple[dict[str, float], float, float | None, list[str] | None]:
        if isinstance(source, PortfolioRunSource):
            with self._run_lock:
                record = self._runs.get(source.run_id)
                if record is not None:
                    self._runs.move_to_end(source.run_id)
            if record is None:
                raise ValueError(
                    "run_id is unknown or was evicted; rerun portfolio_optimize"
                )
            return (
                dict(record.weights),
                record.cash,
                source.equity_usd,
                list(record.tickers),
            )

        assert isinstance(source, PortfolioWeightsSource)
        normalized: dict[str, float] = {}
        for ticker, weight in source.weights.items():
            key = str(ticker).strip().upper()
            if not key:
                raise ValueError("portfolio contains a blank ticker")
            if key in normalized:
                raise ValueError(
                    "portfolio contains duplicate tickers after normalization"
                )
            normalized[key] = float(weight)
        return normalized, source.cash, source.equity_usd, None

    def _portfolio_context(
        self,
        source: PortfolioSource,
        regime: Regime | None,
        *,
        universe: list[str] | None = None,
    ) -> tuple[dict, list[str], dict[str, float], float, float | None]:
        weights, cash, equity, run_tickers = self._portfolio_values(source)
        selected = universe or run_tickers or list(weights)
        if not selected:
            selected = list(self._prices.columns[:2])
        frame, selected, regime_dict = self._selected_prices(selected, regime)
        unknown = sorted(set(weights) - set(selected))
        if unknown:
            raise ValueError(
                "portfolio contains tickers outside the selected universe: "
                + ", ".join(unknown)
            )
        returns = calculate_returns(
            frame,
            regime_dict=regime_dict,
            returns_compute_settings=ReturnsComputeSettings(
                return_type="LOG",
                freq=1,
                returns_compute_device="CPU",
                verbose=False,
            ),
        )
        return returns, selected, weights, cash, equity

    @staticmethod
    def _portfolio(
        name: str,
        tickers: list[str],
        weights: dict[str, float],
        cash: float,
    ) -> Portfolio:
        return Portfolio(
            name=name,
            tickers=tickers,
            weights=np.array([weights.get(ticker, 0.0) for ticker in tickers]),
            cash=cash,
        )

    def _scenario_data(
        self,
        returns: dict,
        settings: ScenarioSettings,
    ) -> dict:
        product = len(returns["tickers"]) * settings.count
        if product > self._MAX_ASSET_SCENARIO_VALUES:
            maximum = self._MAX_ASSET_SCENARIO_VALUES // len(returns["tickers"])
            raise ValueError(
                "assets × scenarios exceeds the local MCP safety limit; "
                f"for {len(returns['tickers'])} assets use at most "
                f"{maximum} scenarios"
            )
        generated = ScenarioGenerationSettings(
            num_scen=settings.count,
            fit_type=settings.method,
            kde_settings=(
                KDESettings(
                    bandwidth=settings.kde_bandwidth,
                    kernel=settings.kde_kernel,
                    device="GPU",
                )
                if settings.method == "kde"
                else None
            ),
            verbose=False,
            seed=settings.seed,
        )
        return generate_cvar_data(returns, generated)

    @staticmethod
    def _normalized_bound(bound: WeightBound) -> WeightBound:
        if not isinstance(bound, dict):
            return float(bound)
        normalized: dict[str, float] = {}
        for key, value in bound.items():
            normalized_key = (
                "others" if str(key).lower() == "others" else str(key).upper()
            )
            if normalized_key in normalized:
                raise ValueError(
                    "weight bounds contain duplicate tickers after normalization"
                )
            normalized[normalized_key] = float(value)
        return normalized

    @staticmethod
    def _bound_vector(bound: WeightBound, tickers: list[str]) -> np.ndarray:
        if not isinstance(bound, dict):
            return np.full(len(tickers), float(bound))
        unknown = sorted(set(bound) - set(tickers) - {"others"})
        if unknown:
            raise ValueError(
                "weight bounds contain tickers outside the selected universe: "
                + ", ".join(unknown)
            )
        if "others" not in bound:
            missing = [ticker for ticker in tickers if ticker not in bound]
            if missing:
                raise ValueError(
                    "weight bounds must specify every selected ticker or 'others'"
                )
        return np.array(
            [float(bound.get(ticker, bound.get("others"))) for ticker in tickers]
        )

    def _selected_prices(
        self,
        tickers: list[str] | None,
        regime: Regime | None,
    ) -> tuple[pd.DataFrame, list[str], dict]:
        if tickers is None:
            selected = list(self._prices.columns)
        else:
            selected = [str(ticker).strip().upper() for ticker in tickers]
            if len(selected) != len(set(selected)):
                raise ValueError("selected tickers contain duplicates")
        if len(selected) < 2:
            raise ValueError("select at least two distinct tickers")
        unknown = sorted(set(selected) - set(self._prices.columns))
        if unknown:
            raise ValueError(
                "selected tickers are not in the startup dataset: " + ", ".join(unknown)
            )
        frame = self._prices[selected]
        if regime is None:
            regime_dict = {
                "name": "startup_dataset",
                "range": (frame.index[0], frame.index[-1]),
            }
        else:
            start = pd.Timestamp(regime.start)
            end = pd.Timestamp(regime.end)
            if start > end:
                raise ValueError("regime.start must not be later than regime.end")
            frame = frame.loc[start:end]
            if len(frame) < 3:
                raise ValueError("selected regime needs at least three price rows")
            regime_dict = {
                "name": "requested_window",
                "range": (frame.index[0], frame.index[-1]),
            }
        return frame, selected, regime_dict

    @staticmethod
    def _existing_portfolio(
        value: ExistingPortfolio | None,
        tickers: list[str],
    ) -> Portfolio | None:
        if value is None:
            return None
        weights = {
            str(ticker).strip().upper(): float(weight)
            for ticker, weight in value.weights.items()
        }
        if len(weights) != len(value.weights):
            raise ValueError(
                "existing portfolio contains duplicate tickers after normalization"
            )
        unknown = sorted(set(weights) - set(tickers))
        if unknown:
            raise ValueError(
                "existing portfolio contains tickers outside the selected universe: "
                + ", ".join(unknown)
            )
        vector = np.array([weights.get(ticker, 0.0) for ticker in tickers])
        return Portfolio(
            name="existing",
            tickers=tickers,
            weights=vector,
            cash=value.cash,
        )

    @staticmethod
    def _groups(
        constraints: PortfolioConstraints,
        tickers: list[str],
    ) -> list[dict] | None:
        groups: list[dict] = []
        universe = set(tickers)
        for group in constraints.groups:
            members = [str(ticker).strip().upper() for ticker in group.tickers]
            if len(members) != len(set(members)):
                raise ValueError(f"group {group.name!r} contains duplicate tickers")
            unknown = sorted(set(members) - universe)
            if unknown:
                raise ValueError(
                    f"group {group.name!r} contains unknown tickers: "
                    + ", ".join(unknown)
                )
            groups.append(
                {
                    "group_name": group.name,
                    "tickers": members,
                    "weight_bounds": {
                        "w_min": group.minimum,
                        "w_max": group.maximum,
                    },
                }
            )
        return groups or None

    def optimize(
        self,
        *,
        objective: Objective,
        tickers: list[str] | None = None,
        regime: Regime | None = None,
        constraints: PortfolioConstraints | None = None,
        existing_portfolio: ExistingPortfolio | None = None,
        scenarios: ScenarioSettings | None = None,
        time_limit_seconds: float = 60.0,
    ) -> OptimizationOutput:
        """Map an MCP request directly to the package's cuOpt optimizers."""

        active_constraints = constraints or PortfolioConstraints()
        frame, selected, regime_dict = self._selected_prices(tickers, regime)
        w_min = self._normalized_bound(active_constraints.weight_minimum)
        w_max = self._normalized_bound(active_constraints.weight_maximum)
        if bool(
            (
                self._bound_vector(w_min, selected)
                > self._bound_vector(w_max, selected)
            ).any()
        ):
            raise ValueError("weight minimum exceeds maximum for a selected ticker")
        current = self._existing_portfolio(existing_portfolio, selected)
        if active_constraints.turnover_limit is not None and current is None:
            raise ValueError("turnover_limit requires existing_portfolio")
        if isinstance(objective, MeanCvarObjective):
            objective_type = "mean_cvar"
            risk_measure = "CVaR"
        else:
            objective_type = "mean_variance"
            risk_measure = "variance"
            if scenarios is not None:
                raise ValueError("scenarios apply only to mean_cvar")
            if active_constraints.cardinality is not None:
                raise ValueError(
                    "cardinality is not supported by the cuOpt Mean-Variance path"
                )

        returns = calculate_returns(
            frame,
            regime_dict=regime_dict,
            returns_compute_settings=ReturnsComputeSettings(
                return_type="LOG",
                freq=1,
                returns_compute_device="CPU",
                verbose=False,
            ),
        )
        group_constraints = self._groups(active_constraints, selected)
        api_settings = ApiSettings(
            api="cuopt_python",
            scale_risk_aversion=objective.scale_risk_aversion,
        )

        with suppress_stdout():
            if isinstance(objective, MeanCvarObjective):
                scenario_request = scenarios or ScenarioSettings()
                returns = self._scenario_data(returns, scenario_request)
                params = CvarParameters(
                    w_min=w_min,
                    w_max=w_max,
                    c_min=active_constraints.cash_minimum,
                    c_max=active_constraints.cash_maximum,
                    risk_aversion=objective.risk_aversion,
                    L_tar=active_constraints.leverage_limit,
                    T_tar=active_constraints.turnover_limit,
                    cardinality=active_constraints.cardinality,
                    group_constraints=group_constraints,
                    confidence=objective.confidence,
                    cvar_limit=objective.cvar_limit,
                )
                optimizer = CVaR(
                    returns,
                    params,
                    api_settings=api_settings,
                    existing_portfolio=current,
                )
                result, portfolio = optimizer.solve_optimization_problem(
                    {"time_limit": time_limit_seconds},
                    print_results=False,
                )
                scenario_count = len(returns["cvar_data"].p)
            else:
                params = MeanVarianceParameters(
                    w_min=w_min,
                    w_max=w_max,
                    c_min=active_constraints.cash_minimum,
                    c_max=active_constraints.cash_maximum,
                    risk_aversion=objective.risk_aversion,
                    L_tar=active_constraints.leverage_limit,
                    T_tar=active_constraints.turnover_limit,
                    group_constraints=group_constraints,
                    var_limit=objective.variance_limit,
                )
                optimizer = MeanVariance(
                    returns,
                    params,
                    api_settings=api_settings,
                    existing_portfolio=current,
                )
                result, portfolio = optimizer.solve_optimization_problem(
                    {"time_limit": time_limit_seconds},
                    print_results=False,
                )
                scenario_count = None

        risk_value = float(result[risk_measure])
        metrics = OptimizationMetrics(
            expected_return_daily=float(result["return"]),
            risk_measure=risk_measure,
            risk_value_daily=risk_value,
            objective_value=float(result["obj"]),
            solve_time_seconds=float(result["solve time"]),
        )
        weights = {
            ticker: float(portfolio.weights[index])
            for index, ticker in enumerate(selected)
        }
        cash = float(portfolio.cash)
        allocation = sum(weights.values()) + cash
        if not all(
            math.isfinite(value)
            for value in (
                *weights.values(),
                cash,
                metrics.expected_return_daily,
                metrics.risk_value_daily,
                metrics.objective_value,
                metrics.solve_time_seconds,
            )
        ):
            raise RuntimeError("optimizer returned non-finite output")
        if abs(allocation - 1.0) > 1e-4:
            raise RuntimeError(
                "optimizer returned weights and cash that do not sum to one"
            )
        run_id = self._store_run(weights, cash, selected)

        start, end = returns["regime"]["range"]
        output_regime = Regime(
            start=pd.Timestamp(start).date().isoformat(),
            end=pd.Timestamp(end).date().isoformat(),
        )
        return OptimizationOutput(
            summary=(
                f"cuOpt solved a {objective_type} portfolio over "
                f"{len(selected)} tickers with {cash:.2%} cash."
            ),
            run_id=run_id,
            objective_type=objective_type,
            solver="cuOpt",
            regime=output_regime,
            tickers=selected,
            weights=weights,
            cash=cash,
            total_allocation=allocation,
            metrics=metrics,
            scenario_count=scenario_count,
        )

    @staticmethod
    def _output_regime(returns: dict) -> Regime:
        start, end = returns["regime"]["range"]
        return Regime(
            start=pd.Timestamp(start).date().isoformat(),
            end=pd.Timestamp(end).date().isoformat(),
        )

    @staticmethod
    def _finite(value) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    def analyze(
        self,
        *,
        portfolio: PortfolioSource,
        regime: Regime | None = None,
        confidence: float = 0.95,
        scenarios: ScenarioSettings | None = None,
    ) -> AnalyzeOutput:
        """Evaluate return, variance, CVaR, and concentration for a portfolio."""

        returns, tickers, weights, cash, equity = self._portfolio_context(
            portfolio,
            regime,
        )
        scenario_request = scenarios or ScenarioSettings()
        with suppress_stdout():
            returns = self._scenario_data(returns, scenario_request)
            model = self._portfolio("analyzed", tickers, weights, cash)
            performance = evaluate_portfolio_performance(
                returns["cvar_data"],
                model,
                confidence,
                np.asarray(returns["covariance"], dtype=float),
            )

        vector = np.array([weights.get(ticker, 0.0) for ticker in tickers])
        portfolio_returns = returns["cvar_data"].R.T @ vector
        var_level = np.percentile(portfolio_returns, (1.0 - confidence) * 100.0)
        tail = portfolio_returns[portfolio_returns <= var_level]
        signed_cvar = float(tail.mean()) if tail.size else float(var_level)
        cvar_loss = max(0.0, -signed_cvar)
        variance = float(performance["variance"])
        expected_return = float(performance["return"])

        gross = sum(abs(weight) for weight in weights.values())
        shares = (
            {
                ticker: abs(weight) / gross
                for ticker, weight in weights.items()
                if abs(weight) > 0.0
            }
            if gross > 0.0
            else {}
        )
        largest = max(shares, key=shares.get) if shares else None
        hhi = sum(share**2 for share in shares.values())
        metrics = AnalyzeMetrics(
            expected_return_daily=expected_return,
            expected_return_annualized=expected_return * 252.0,
            variance_daily=variance,
            volatility_annualized=math.sqrt(max(variance, 0.0)) * math.sqrt(252.0),
            cvar_daily_signed=signed_cvar,
            cvar_loss_daily=cvar_loss,
            cvar_confidence=confidence,
            hhi=hhi,
            largest_position=largest,
            largest_weight=shares.get(largest, 0.0) if largest else 0.0,
        )
        return AnalyzeOutput(
            summary=(
                f"Analyzed {len(weights)} positions: annualized volatility is "
                f"{metrics.volatility_annualized:.2%} and the "
                f"{confidence:.0%} one-day CVaR loss is {cvar_loss:.2%}."
            ),
            weights=weights,
            cash=cash,
            equity_usd=equity,
            regime=self._output_regime(returns),
            metrics=metrics,
            scenario_count=len(returns["cvar_data"].p),
        )

    def simulate(
        self,
        *,
        portfolio: PortfolioSource,
        regime: Regime | None = None,
        horizon_trading_days: int = 1,
        scenarios: ScenarioSettings | None = None,
        loss_thresholds_usd: LossThresholds | None = None,
    ) -> SimulationOutput:
        """Generate a fixed-portfolio return distribution."""

        returns, tickers, weights, cash, equity = self._portfolio_context(
            portfolio,
            regime,
        )
        if loss_thresholds_usd and equity is None:
            raise ValueError("loss_thresholds_usd requires portfolio equity_usd")
        scenario_request = scenarios or ScenarioSettings()
        with suppress_stdout():
            returns = self._scenario_data(returns, scenario_request)

        vector = np.array([weights.get(ticker, 0.0) for ticker in tickers])
        one_day = np.asarray(returns["cvar_data"].R.T @ vector, dtype=float)
        if horizon_trading_days == 1:
            horizon_log_returns = one_day
        else:
            rng = np.random.default_rng(scenario_request.seed)
            sampled = rng.integers(
                0,
                len(one_day),
                size=(scenario_request.count, horizon_trading_days),
            )
            horizon_log_returns = one_day[sampled].sum(axis=1)
        outcome_returns = np.exp(horizon_log_returns) - 1.0
        outcome_usd = outcome_returns * equity if equity is not None else None

        percentiles: dict[str, OutcomePercentile] = {}
        for percentile in (1, 5, 10, 25, 50, 75, 90, 95, 99):
            value = float(np.percentile(outcome_returns, percentile))
            percentiles[f"p{percentile}"] = OutcomePercentile(
                return_decimal=value,
                usd=value * equity if equity is not None else None,
            )
        losses = [
            LossProbability(
                threshold_usd=float(threshold),
                probability=float(np.mean(outcome_usd < -float(threshold))),
            )
            for threshold in (loss_thresholds_usd or [])
        ]
        count = len(outcome_returns)
        return SimulationOutput(
            summary=(
                f"Generated {count:,} fixed-portfolio outcomes over "
                f"{horizon_trading_days} trading day(s); median return is "
                f"{percentiles['p50'].return_decimal:.2%}."
            ),
            weights=weights,
            cash=cash,
            equity_usd=equity,
            regime=self._output_regime(returns),
            horizon_trading_days=horizon_trading_days,
            method=(
                scenario_request.method
                if horizon_trading_days == 1
                else f"{scenario_request.method}_iid_compounding"
            ),
            scenario_count=count,
            percentiles=percentiles,
            mean_return=float(outcome_returns.mean()),
            standard_deviation_return=float(outcome_returns.std()),
            mean_usd=(float(outcome_usd.mean()) if outcome_usd is not None else None),
            standard_deviation_usd=(
                float(outcome_usd.std()) if outcome_usd is not None else None
            ),
            loss_probabilities=losses,
        )

    def backtest(
        self,
        *,
        portfolios: list[BacktestPortfolio],
        tickers: list[str] | None = None,
        regime: Regime | None = None,
        method: str = "historical",
        risk_free_rate_annualized: float = 0.0,
        seed: int | None = 42,
    ) -> BacktestOutput:
        """Compare portfolios with the package's backtesting implementation."""

        if not portfolios:
            raise ValueError("provide at least one portfolio")
        names = [item.name for item in portfolios]
        if len(names) != len(set(names)):
            raise ValueError("backtest portfolio names must be unique")
        if "equal-weight" in {name.lower() for name in names}:
            raise ValueError("'equal-weight' is reserved for the automatic benchmark")
        resolved = [self._portfolio_values(item.portfolio) for item in portfolios]
        if tickers is None:
            universe: list[str] = []
            for weights, _, _, run_tickers in resolved:
                for ticker in run_tickers or list(weights):
                    if ticker not in universe:
                        universe.append(ticker)
            if not universe:
                universe = list(self._prices.columns[:2])
        else:
            universe = list(
                dict.fromkeys(str(ticker).strip().upper() for ticker in tickers)
            )

        frame, selected, regime_dict = self._selected_prices(universe, regime)
        returns = calculate_returns(
            frame,
            regime_dict=regime_dict,
            returns_compute_settings=ReturnsComputeSettings(
                return_type="LOG",
                freq=1,
                returns_compute_device="CPU",
                verbose=False,
            ),
        )
        models: list[Portfolio] = []
        for item, (weights, cash, _, _) in zip(portfolios, resolved, strict=True):
            unknown = sorted(set(weights) - set(selected))
            if unknown:
                raise ValueError(
                    f"portfolio {item.name!r} contains tickers outside the "
                    "backtest universe: " + ", ".join(unknown)
                )
            models.append(self._portfolio(item.name, selected, weights, cash))

        test = models[0]
        equal_weight = Portfolio(
            name="equal-weight",
            tickers=selected,
            weights=np.full(len(selected), (1.0 - test.cash) / len(selected)),
            cash=test.cash,
        )
        benchmarks = models[1:] + [equal_weight]
        daily_risk_free = (1.0 + risk_free_rate_annualized) ** (1.0 / 252.0) - 1.0
        with suppress_stdout():
            runner = portfolio_backtester(
                test_portfolio=test,
                returns_dict=returns,
                risk_free_rate=daily_risk_free,
                test_method=method,
                benchmark_portfolios=benchmarks,
                seed=(None if method == "historical" else seed),
            )
            result_frame, _ = runner.backtest_against_benchmarks(plot_returns=False)

        output_metrics: dict[str, BacktestMetrics] = {}
        for name, row in result_frame.iterrows():
            period_returns = np.asarray(row["returns"], dtype=float)
            mean_daily = self._finite(period_returns.mean())
            cumulative = np.asarray(row["cumulative returns"], dtype=float)
            output_metrics[str(name)] = BacktestMetrics(
                mean_return_daily=mean_daily,
                return_annualized=(
                    mean_daily * 252.0 if mean_daily is not None else None
                ),
                sharpe_annualized=self._finite(row["sharpe"]),
                sortino_annualized=self._finite(row["sortino"]),
                max_drawdown=self._finite(row["max drawdown"]),
                ending_value_of_10000=(
                    self._finite(cumulative[-1] * 10_000.0) if cumulative.size else None
                ),
            )
        return BacktestOutput(
            summary=(
                f"Backtested {len(models)} supplied portfolio(s) plus an "
                f"equal-weight benchmark using {method}."
            ),
            method=method,
            regime=self._output_regime(returns),
            seed=None if method == "historical" else seed,
            metrics=output_metrics,
        )
