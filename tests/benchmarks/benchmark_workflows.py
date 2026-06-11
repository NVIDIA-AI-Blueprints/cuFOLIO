# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Layer 3 skill-performance benchmarks for the cufolio skill.

Each ``run_*`` function executes one documented SKILL.md workflow end-to-end on a
GPU (cuOpt), exactly as the skill instructs an agent to write it — so this module
doubles as a check that the SKILL.md recipes still run. Each ``check_*`` function
grades a workflow's metrics against ``thresholds.toml`` (the explicit "standards").

Run as a script for a report::

    python tests/benchmarks/benchmark_workflows.py            # fast, small universe
    python tests/benchmarks/benchmark_workflows.py --full     # full S&P 500
    python tests/benchmarks/benchmark_workflows.py --check    # also print PASS/FAIL

or as a pytest gate (auto-skips off-GPU): ``uv run pytest -m gpu``.

Requires the ``cufolio`` package with NVIDIA cuOpt + cuML installed (e.g. the Brev
launchable or ``uv sync --extra cuda12``), and network access on first run to
download price data. The CLI exits cleanly with a SKIP message when the GPU runtime is absent.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import tempfile
import time

import cvxpy as cp
import numpy as np
import pandas as pd

from cufolio import backtest, cvar_optimizer, cvar_utils, rebalance, utils
from cufolio.cvar_parameters import CvarParameters
from cufolio.portfolio import Portfolio
from cufolio.settings import (
    KDESettings,
    ReturnsComputeSettings,
    ScenarioGenerationSettings,
)

# Canonical solver settings from SKILL.md — verbatim.
SOLVER_SETTINGS = {"solver": cp.CUOPT, "verbose": False, "solver_method": "PDLP"}

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
THRESHOLDS_PATH = os.path.join(_HERE, "thresholds.toml")

# Fast-mode defaults keep the suite inside a CI / launchable time budget.
SMALL_UNIVERSE = 15
FAST_NUM_SCEN = 2000
FULL_NUM_SCEN = 10000
DEFAULT_START = "2022-01-01"
DEFAULT_END = "2024-01-01"


class DataUnavailable(RuntimeError):
    """Raised when price data cannot be located or downloaded (test -> skip)."""


def gpu_runtime_available() -> tuple[bool, str]:
    """Return whether the cuOpt/cuML GPU benchmark runtime is importable."""
    missing = []
    for module_name in ("cuopt", "cuml"):
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            missing.append(module_name)
    if not hasattr(cp, "CUOPT"):
        missing.append("cvxpy.CUOPT")
    if missing:
        return False, "missing " + ", ".join(missing)
    return True, ""


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def resolve_csv_path(csv_path: str | None = None, allow_download: bool = True) -> str:
    """Locate a price CSV, downloading the default S&P 500 dataset if needed."""
    candidates = [
        csv_path,
        os.path.join("data", "stock_data", "sp500.csv"),
        os.path.join(_REPO_ROOT, "data", "stock_data", "sp500.csv"),
    ]
    for cand in candidates:
        if cand and os.path.exists(cand):
            return cand
    if allow_download:
        target = os.path.join(_REPO_ROOT, "data", "stock_data")
        try:
            utils.download_data(target, datasets=["sp500"])
        except Exception as exc:  # network / yfinance failure
            raise DataUnavailable(f"could not download price data: {exc}") from exc
        sp500 = os.path.join(target, "sp500.csv")
        if os.path.exists(sp500):
            return sp500
    raise DataUnavailable(
        "no price CSV found; pass --csv or run "
        "cufolio.utils.download_data('data/stock_data', datasets=['sp500']) first"
    )


def load_prices(
    csv_path: str | None = None,
    n_tickers: int | None = SMALL_UNIVERSE,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    allow_download: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Load a (optionally small) price panel with complete history in [start, end]."""
    path = resolve_csv_path(csv_path, allow_download)
    prices = utils.get_input_data(path)
    prices = prices.loc[start:end].dropna(axis=1)
    if prices.shape[1] == 0 or len(prices) < 60:
        raise DataUnavailable(
            f"price data at {path} has too few complete columns/rows in "
            f"[{start}, {end}] (got {prices.shape})"
        )
    if n_tickers and prices.shape[1] > n_tickers:
        prices = prices.iloc[:, :n_tickers]
    return prices, path


def prepare(prices: pd.DataFrame, num_scen: int = FAST_NUM_SCEN) -> dict:
    """returns -> KDE/GPU scenarios, the shared front half of every workflow."""
    returns_dict = utils.calculate_returns(
        prices,
        regime_dict=None,
        returns_compute_settings=ReturnsComputeSettings(return_type="LOG"),
    )
    scenario_settings = ScenarioGenerationSettings(
        num_scen=num_scen,
        fit_type="kde",
        kde_settings=KDESettings(bandwidth=0.01, kernel="gaussian", device="GPU"),
    )
    return cvar_utils.generate_cvar_data(returns_dict, scenario_settings)


def _full_invested_params() -> CvarParameters:
    """Long-only, fully invested (Trap 1 + Trap 2 fixes baked in)."""
    return CvarParameters(
        w_min=0.0,
        w_max=1.0,
        c_min=0.0,
        c_max=0.0,
        risk_aversion=1.0,
        confidence=0.95,
    )


def _close(fig) -> None:
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Workflows (mirror the SKILL.md recipes)
# --------------------------------------------------------------------------- #
def run_build_optimal(returns_dict: dict) -> tuple[dict, Portfolio]:
    """SKILL.md: build the optimal Mean-CVaR portfolio (Traps 1 & 2 applied)."""
    params = _full_invested_params()
    optimizer = cvar_optimizer.CVaR(returns_dict, params)
    start = time.time()
    result, portfolio = optimizer.solve_optimization_problem(
        solver_settings=SOLVER_SETTINGS, print_results=False
    )
    elapsed = time.time() - start
    weights = np.asarray(portfolio.weights, dtype=float).flatten()
    cash = float(np.asarray(portfolio.cash).squeeze())
    metrics = {
        "expected_return": float(result["return"]),
        "cvar": float(result["CVaR"]),
        "cash_weight": cash,
        "weight_sum": float(weights.sum() + cash),
        "max_weight": float(np.max(np.abs(weights))) if weights.size else 0.0,
        "n_positions": int(np.sum(np.abs(weights) > 1e-4)),
        "solver": str(result["solver"]),
        "solve_seconds": float(result.get("solve time", elapsed)),
    }
    return metrics, portfolio


def run_efficient_frontier(returns_dict: dict, ra_num: int = 25) -> dict:
    """SKILL.md: efficient frontier (results_df carries metrics + per-asset weights)."""
    results_df, fig, _ = cvar_utils.create_efficient_frontier(
        returns_dict,
        _full_invested_params(),
        SOLVER_SETTINGS,
        ra_num=ra_num,
        min_risk_aversion=-3,
        max_risk_aversion=1,
        show_plot=False,
        show_discretized_portfolios=False,  # skip the discretized overlay (extra compute)
        benchmark_portfolios=False,
        print_portfolio_results=False,
    )
    _close(fig)
    ordered = results_df.sort_values("CVaR")["return"].to_numpy()
    return {
        "num_points": int(len(results_df)),
        "return_at_min_cvar": float(ordered[0]),
        "return_at_max_cvar": float(ordered[-1]),
    }


def run_weights_table(returns_dict: dict, n_steps: int = 12) -> dict:
    """SKILL.md: per-asset weights by risk aversion from the frontier's weights column."""
    results_df, fig, _ = cvar_utils.create_efficient_frontier(
        returns_dict,
        _full_invested_params(),
        SOLVER_SETTINGS,
        ra_num=n_steps,
        min_risk_aversion=-3,
        max_risk_aversion=1,
        show_plot=False,
        show_discretized_portfolios=False,
        benchmark_portfolios=False,
        print_portfolio_results=False,
    )
    _close(fig)
    weights_table = pd.DataFrame(results_df["weights"].tolist(), index=results_df.index)
    return {
        "rows": int(len(weights_table)),
        "weight_columns": int(weights_table.shape[1]),
    }


def run_backtest(returns_dict: dict, portfolio: Portfolio) -> dict:
    """SKILL.md: backtest the optimal portfolio vs an equal-weight benchmark.

    In-sample by construction (same regime used to optimize), which is enough to
    assert the optimizer beats a naive allocation on a risk-adjusted basis.
    """
    tickers = returns_dict["tickers"]
    optimal = Portfolio(
        name="cuOpt Optimal",
        tickers=tickers,
        weights=np.asarray(portfolio.weights, dtype=float).flatten(),
        cash=float(np.asarray(portfolio.cash).squeeze()),
    )
    n = len(tickers)
    equal_weight = Portfolio(
        name="Equal Weight", tickers=tickers, weights=np.ones(n) / n, cash=0.0
    )
    tester = backtest.portfolio_backtester(
        test_portfolio=optimal,
        returns_dict=returns_dict,
        risk_free_rate=0.0,
        test_method="historical",
        benchmark_portfolios=[equal_weight],
    )
    results, _ = tester.backtest_against_benchmarks(plot_returns=False)
    return {
        "optimized_sharpe": float(results.loc["cuOpt Optimal", "sharpe"]),
        "naive_sharpe": float(results.loc["Equal Weight", "sharpe"]),
        "optimized_sortino": float(results.loc["cuOpt Optimal", "sortino"]),
        "optimized_max_drawdown": float(results.loc["cuOpt Optimal", "max drawdown"]),
    }


def run_rebalance(prices: pd.DataFrame) -> dict:
    """SKILL.md: monthly rebalancing via drift_from_optimal with threshold=0."""
    look_back, look_forward = 126, 21
    index = prices.index
    start_pos = look_back + 1
    end_pos = min(len(index) - 1, start_pos + 84)
    if start_pos >= end_pos:
        raise DataUnavailable("not enough history for a rebalancing window")

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "prices.csv")
        prices.to_csv(csv_path)  # date index -> column 0 (read_csv index_col=0)
        rebalancer = rebalance.rebalance_portfolio(
            dataset_directory=csv_path,
            returns_compute_settings=ReturnsComputeSettings(return_type="LOG"),
            scenario_generation_settings=ScenarioGenerationSettings(
                num_scen=FAST_NUM_SCEN,
                fit_type="kde",
                kde_settings=KDESettings(device="GPU"),
            ),
            trading_start=str(pd.Timestamp(index[start_pos]).date()),
            trading_end=str(pd.Timestamp(index[end_pos]).date()),
            look_forward_window=look_forward,
            look_back_window=look_back,
            cvar_params=_full_invested_params(),
            solver_settings=SOLVER_SETTINGS,
            re_optimize_criteria={
                "type": "drift_from_optimal",
                "threshold": 0,
                "norm": 1,
            },
        )
        results_df, rebalance_dates, cumulative_value = rebalancer.re_optimize(
            transaction_cost_factor=0.001,
            plot_results=False,
            plot_title="Monthly Rebalancing",
        )
    return {
        "n_rebalance_dates": int(len(rebalance_dates)),
        "cumulative_value_len": int(len(cumulative_value)),
        "result_rows": int(len(results_df)),
    }


# --------------------------------------------------------------------------- #
# Standards checks
# --------------------------------------------------------------------------- #
def load_thresholds(path: str = THRESHOLDS_PATH) -> dict:
    import tomllib  # stdlib (py>=3.11); lazy import keeps it out of the hot path

    with open(path, "rb") as handle:
        return tomllib.load(handle)


def check_build_optimal(m: dict, th: dict) -> list[str]:
    fails = []
    if abs(m["weight_sum"] - 1.0) > th["weight_sum_tol"]:
        fails.append(f"weights+cash sum to {m['weight_sum']:.6f}, not ~1")
    if m["cash_weight"] >= th["cash_weight_max"]:
        fails.append(f"degenerate all-cash optimum (cash={m['cash_weight']:.3f})")
    if m["max_weight"] >= th["max_single_weight"]:
        fails.append(f"single asset holds {m['max_weight']:.3f} of the portfolio")
    if m["n_positions"] < th["min_positions"]:
        fails.append(f"only {m['n_positions']} position(s); expected diversification")
    if not (
        th["expected_return_daily_min"]
        <= m["expected_return"]
        <= th["expected_return_daily_max"]
    ):
        fails.append(f"expected daily return {m['expected_return']:.5f} out of range")
    if not (th["cvar_min"] <= m["cvar"] <= th["cvar_max"]):
        fails.append(f"CVaR {m['cvar']:.5f} out of range")
    if th["solver_contains"] not in m["solver"].lower():
        fails.append(f"solver was {m['solver']!r}, not cuOpt")
    if m["solve_seconds"] > th["solve_seconds_max"]:
        fails.append(f"solve took {m['solve_seconds']:.2f}s")
    return fails


def check_efficient_frontier(m: dict, th: dict) -> list[str]:
    fails = []
    if m["num_points"] != th["num_points"]:
        fails.append(f"{m['num_points']} frontier points, expected {th['num_points']}")
    if m["return_at_max_cvar"] < m["return_at_min_cvar"] - th["monotonic_return_tol"]:
        fails.append("frontier not monotonic: higher CVaR did not yield higher return")
    return fails


def check_weights_table(m: dict, th: dict) -> list[str]:
    fails = []
    if m["rows"] < th["min_rows"]:
        fails.append(f"{m['rows']} rows, expected >= {th['min_rows']}")
    if m["weight_columns"] < th["min_weight_columns"]:
        fails.append(f"{m['weight_columns']} per-asset weight columns")
    return fails


def check_backtest(m: dict, th: dict) -> list[str]:
    fails = []
    if th.get("require_finite_metrics") and not all(
        np.isfinite(v)
        for v in (
            m["optimized_sharpe"],
            m["optimized_sortino"],
            m["optimized_max_drawdown"],
        )
    ):
        fails.append("non-finite backtest metric")
    if th.get("optimized_beats_naive_sharpe") and not (
        m["optimized_sharpe"] > m["naive_sharpe"]
    ):
        fails.append(
            f"optimized Sharpe {m['optimized_sharpe']:.3f} did not beat naive "
            f"{m['naive_sharpe']:.3f}"
        )
    return fails


def check_rebalance(m: dict, th: dict) -> list[str]:
    fails = []
    if m["n_rebalance_dates"] < th["min_rebalance_dates"]:
        fails.append("no rebalancing dates produced")
    if th.get("require_cumulative_value") and m["cumulative_value_len"] <= 0:
        fails.append("empty cumulative value series")
    return fails


# --------------------------------------------------------------------------- #
# Orchestration / CLI
# --------------------------------------------------------------------------- #
def run_all(full: bool = False, csv_path: str | None = None) -> dict:
    """Run every workflow once and return {workflow: metrics}."""
    prices, _ = load_prices(
        csv_path=csv_path, n_tickers=None if full else SMALL_UNIVERSE
    )
    returns_dict = prepare(prices, FULL_NUM_SCEN if full else FAST_NUM_SCEN)
    build_metrics, portfolio = run_build_optimal(returns_dict)
    return {
        "build_optimal": build_metrics,
        "efficient_frontier": run_efficient_frontier(returns_dict),
        "weights_table": run_weights_table(returns_dict, n_steps=25 if full else 12),
        "backtest": run_backtest(returns_dict, portfolio),
        "rebalance": run_rebalance(prices),
    }


_CHECKERS = {
    "build_optimal": check_build_optimal,
    "efficient_frontier": check_efficient_frontier,
    "weights_table": check_weights_table,
    "backtest": check_backtest,
    "rebalance": check_rebalance,
}


def check_all(results: dict, thresholds: dict) -> dict[str, list[str]]:
    return {
        name: _CHECKERS[name](metrics, thresholds[name])
        for name, metrics in results.items()
        if name in _CHECKERS
    }


def _format_report(results: dict) -> str:
    lines = ["| workflow | key metrics |", "|---|---|"]
    for name, metrics in results.items():
        summary = ", ".join(f"{k}={v}" for k, v in metrics.items())
        lines.append(f"| {name} | {summary} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="full S&P 500 universe")
    parser.add_argument("--csv", default=None, help="path to a price CSV")
    parser.add_argument("--check", action="store_true", help="grade vs thresholds.toml")
    args = parser.parse_args()

    gpu_ok, reason = gpu_runtime_available()
    if not gpu_ok:
        print(f"SKIP cuFOLIO GPU benchmark workflows: {reason}")
        return 0

    results = run_all(full=args.full, csv_path=args.csv)
    print(_format_report(results))

    if args.check:
        failures = check_all(results, load_thresholds())
        any_failed = False
        print("\nStandards:")
        for name, fails in failures.items():
            if fails:
                any_failed = True
                print(f"  FAIL {name}: " + "; ".join(fails))
            else:
                print(f"  PASS {name}")
        return 1 if any_failed else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
