---
name: cufolio
version: "25.10.00"
description: Build GPU-accelerated Mean-CVaR portfolios with NVIDIA cuOpt — CVaR optimization, efficient frontier, scenario generation, backtesting, and rebalancing.
license: Apache-2.0
metadata:
  author: Jake Goldberg <jgoldberg@nvidia.com>
  tags:
    - portfolio-optimization
    - cvar
    - cuopt
    - quantitative-finance
    - gpu
---

# cuFOLIO Skill

<!--
SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Purpose

Build and analyze quantitative portfolios with NVIDIA-accelerated Mean-CVaR
optimization: compute returns, generate scenarios, solve CVaR-optimal allocations on
the cuOpt GPU solver, trace the efficient frontier, backtest, and rebalance. Applies
whenever a task is about constructing or evaluating a portfolio of assets from price data.

## Setup

This skill drives the installed `cufolio` Python package (GPU-accelerated; requires
NVIDIA cuOpt + cuML). It assumes that environment is already available — e.g. the
[Brev launchable](https://brev.nvidia.com/launchable/deploy?launchableID=env-360InRZzyHqDnJYQKIxaSggF8xI),
or a clone of `NVIDIA-AI-Blueprints/quantitative-portfolio-optimization` set up with
`uv sync --extra cuda12` (or `--extra cuda13`). Import everything from the `cufolio`
package (see **API** and **Guidelines**).

The default price dataset is **not** shipped in the repo (it is gitignored). If
`data/stock_data/sp500.csv` is missing, download it first:

```python
from cufolio.utils import download_data
download_data("data/stock_data", datasets=["sp500"])   # also available: "sp100", "dow30"
```

## Instructions

Canonical workflow — apply the **Defaults** below and the **Traps** without prompting:

1. **Load price data** (`data/stock_data/sp500.csv`; download first if missing, see Setup) and filter to the requested tickers on the DataFrame.
2. **Compute returns** with `utils.calculate_returns(...)` (`return_type="LOG"`).
3. **Generate scenarios** with `cvar_utils.generate_cvar_data(...)` (KDE, `device="GPU"`).
4. **Define `CvarParameters`** explicitly: `w_min=0.0, w_max=1.0` (Trap 1) and, for a "build the optimal portfolio" request, `c_max=0.0` to avoid the all-cash optimum (Trap 2).
5. **Solve on GPU** via `cvar_optimizer.CVaR(...).solve_optimization_problem(SOLVER_SETTINGS)` — always cuOpt, never a CPU solver (see Solver).
6. **Deliver** the allocation + expected return + CVaR, plus any requested efficient frontier, backtest, or rebalancing output (see API).

See **Traps** for the non-obvious fixes, **Solver** for the canonical settings, and **API** for entry points and source modules.

## Data

Default dataset: `data/stock_data/sp500.csv` — daily prices for S&P 500 constituents
(historical snapshot, ~397 tickers). Use this unless the user supplies a different file
or tickers. If the file is absent, fetch it via `download_data` (see **Setup**).

Coverage notes:
- Constituents reflect a historical snapshot and may not include every current S&P 500 name. If a requested ticker is not a column in the CSV, drop it and proceed with the rest (log the omission); do not fetch it from an external source unless the user explicitly asks.
- Ticker filtering is the caller's responsibility — filter the loaded DataFrame to the desired columns before passing it to `utils.calculate_returns`. `regime_dict` does NOT take a tickers field.

## Defaults (use without asking)

| Parameter | Default |
|---|---|
| Dataset | `data/stock_data/sp500.csv` |
| Date range | Full available range in dataset |
| Portfolio type | Long-only |
| Max weight | None (unconstrained) |
| Risk aversion | 1.0 |
| CVaR confidence | 0.95 |
| Scenario method | KDE |
| KDE device | `GPU` |
| Solver | cuOpt GPU (see Solver section below) |
| Output | Numerical results + plots (allocation, backtest) |
| Rebalancing | None unless requested |

Override only when the user explicitly specifies a different value. Do not prompt for confirmation on covered defaults.

## Traps (read before writing code)

These are non-obvious behaviors that have caused wrong or degenerate results on past runs. Apply the fix without prompting the user.

### Trap 1 — `CvarParameters` has inverted weight-bound defaults

`CvarParameters()` with no weight args defaults to `w_min=1.0`, `w_max=0.0` (infeasible). For long-only optimization you MUST set `w_min=0.0`, `w_max=1.0` explicitly.

### Trap 2 — Degenerate all-cash optimum on small universes

Observed on small universes: the default `c_max=1.0` (cash is a feasible asset) combined with the optimizer's internal `scale_risk_aversion=True` heuristic — which rescales any user-provided `risk_aversion` down to ≈ `max_i(μ_i / CVaR_i)` — can make cash tie the best single risky asset, so cuOpt returns a degenerate 100% cash allocation.

**Symptom:** allocation prints as `Cash 100.00%` with `Expected Return: 0.0000%`, even with `risk_aversion=1.0` and a healthy stock universe.

**Fix for "build the optimal portfolio" queries:** pin `c_max=0.0` to force full investment via `sum(w)=1`. This matches user intent (a portfolio *of* the named stocks, not "stocks or cash, whichever wins"):

```python
cvar_params = CvarParameters(
    w_min=0.0, w_max=1.0,
    c_min=0.0, c_max=0.0,        # force sum(w)=1, no cash
    risk_aversion=1.0, confidence=0.95,
)
```

Keep `c_max=1.0` only when the user explicitly wants cash as a feasible asset (e.g., an efficient-frontier sweep where cash naturally appears at the min-risk corner).

### Trap 3 — `create_efficient_frontier` does not return per-portfolio weights

The function returns `(results_df, fig, ax)`. `results_df` has only metrics columns (return, CVaR, variance, sharpe) — **no per-asset weights**. The internal `portfolios` list is built and discarded before return.

If the user wants a weights-by-risk-aversion table, use the manual loop recipe in the Guidelines section. **Do not** call `create_efficient_frontier` AND a recovery loop — that doubles the solve count for no benefit.

### Trap 4 — `create_efficient_frontier`'s discretized-overlay code path crashes

The default `show_discretized_portfolios=True` triggers an internal call to `evaluate_all_linear_combinations(...)` with a `sum_to_one` kwarg the helper doesn't accept, raising:

```
evaluate_all_linear_combinations() got an unexpected keyword argument 'sum_to_one'
```

This is a kwarg-drift bug in the library, independent of the optimization. The 25 cuOpt solves themselves complete fine; only the post-solve overlay fails. **Always pass `show_discretized_portfolios=False`** when calling `create_efficient_frontier` until the underlying signature is fixed:

```python
results_df, fig, ax = cvar_utils.create_efficient_frontier(
    returns_dict, cvar_params, solver_settings=SOLVER_SETTINGS,
    ra_num=25,
    show_discretized_portfolios=False,   # workaround for kwarg-drift bug
)
```

## Solver — always use cuOpt (GPU)

Use NVIDIA cuOpt for all optimization. **Never use CPU solvers** (e.g. CLARABEL, SCS, ECOS).

**Canonical solver settings (use these verbatim):**
```python
import cvxpy as cp
SOLVER_SETTINGS = {"solver": cp.CUOPT, "verbose": False, "solver_method": "PDLP"}
```

Pass `SOLVER_SETTINGS` to every solve call — single-shot or in a loop. The `solver_method="PDLP"` entry selects the first-order PDLP method used throughout the notebooks; keep it to avoid solver instability on repeated solves.

**Option 1 — CVXPY with cuOpt solver (preferred):**
```python
result, portfolio = cvar_problem.solve_optimization_problem(
    solver_settings=SOLVER_SETTINGS
)
```

**Option 2 — cuOpt Python API directly:**
```python
from cufolio.settings import ApiSettings
api_settings = ApiSettings(api="cuopt_python")
optimizer = CVaR(returns_dict, cvar_params, api_settings=api_settings)
result, portfolio = optimizer.solve_optimization_problem(
    solver_settings={"time_limit": 60}
)
```

For KDE scenario generation, always set `device='GPU'`:
```python
kde_settings = KDESettings(bandwidth=0.01, kernel='gaussian', device='GPU')
```

## API (always use these; never reimplement)

Each bullet lists the canonical entry point and the source module to consult for full signatures, kwargs, and return shapes. Read the referenced module when you need a detail the bullet does not cover. (In the product repo these modules live under `src/`; they import as `cufolio.*`.)

- **Returns** — `utils.calculate_returns(input_dataset, regime_dict, returns_compute_settings)` (`cufolio/utils.py`). `input_dataset` is a CSV path or a pre-filtered DataFrame; filter tickers on the DataFrame beforehand. `regime_dict` shape is `{"name": str, "range": (start, end)}` — no `tickers` key, no nested wrapper.
- **Scenarios** — `cvar_utils.generate_cvar_data(returns_dict, scenario_generation_settings)` (`cufolio/cvar_utils.py`).
- **CVaR problem** — `cvar_optimizer.CVaR(returns_dict, cvar_params)` (`cufolio/cvar_optimizer.py`).
- **Solve** — `cvar_problem.solve_optimization_problem(solver_settings=SOLVER_SETTINGS)` (`cufolio/cvar_optimizer.py`). Pass the canonical dict from the Solver section above; the same call works for a single solve or inside a loop.
- **Backtest** — `backtest.portfolio_backtester(...)` / `backtester.backtest_against_benchmarks(...)` (`cufolio/backtest.py`). `test_method` is one of `"historical"`, `"kde_simulation"`, `"gaussian_simulation"`; returns cumulative returns, Sharpe, Sortino, max drawdown.
- **Efficient frontier** — `cvar_utils.create_efficient_frontier(returns_dict, cvar_params, solver_settings=SOLVER_SETTINGS, ra_num=...)` (`cufolio/cvar_utils.py`). Returns `(results_df, fig, ax)` where `results_df` is **metrics-only** (no per-asset weights — see Trap 3). Use this when the deliverable is the plot + metrics. If the user needs a weights table, use the loop recipe in Guidelines instead.
- **Rebalancing** — `rebalance.rebalance_portfolio(...)` / `rebal_obj.re_optimize(...)` (`cufolio/rebalance.py`). The re-optimization trigger is a dict: `re_optimize_criteria={"type": ..., "threshold": ..., "norm": ...}`, where `type` is one of `"pct_change"`, `"drift_from_optimal"` (also needs `"norm"`: `1` or `2`), or `"max_drawdown"`. For a fixed monthly schedule, use `"drift_from_optimal"` with `threshold=0`.
- **Plots** — `portfolio.plot_portfolio(...)` (`cufolio/portfolio.py`), `backtester.backtest_against_benchmarks(plot_returns=True)`, `utils.portfolio_plot_with_backtest(...)`, `rebal_obj.plot_weights_vs_prices(...)`.
- **Settings models** — `ReturnsComputeSettings`, `ScenarioGenerationSettings`, `KDESettings`, `ApiSettings` in `cufolio/settings.py`; `CvarParameters` in `cufolio/cvar_parameters.py`.

## Guidelines

- **SKILL.md is the primary reference.** It covers the typical workflow; consult the source module listed on each API bullet for anything it does not spell out (full signatures, optional kwargs, return shapes). Use this path before reimplementing behaviour.
- **Always use cuOpt GPU solver** — never fall back to CPU solvers (CLARABEL, SCS, ECOS). Use `cp.CUOPT` or `api="cuopt_python"`.
- **Always pass the canonical `SOLVER_SETTINGS`** (`{"solver": cp.CUOPT, "verbose": False, "solver_method": "PDLP"}`) to every solve call. Keep the `solver_method="PDLP"` entry.
- **Efficient frontier — pick ONE recipe, never both:**

  **(a) Plot + metrics only** (no per-portfolio weights needed): call `cvar_utils.create_efficient_frontier(...)` and use its `results_df` + figure.

  **(b) Need per-portfolio weights** (CSV table, allocation drilldown): write the loop directly, since `create_efficient_frontier` does not expose weights:

  ```python
  from cufolio import cvar_optimizer
  problem = cvar_optimizer.CVaR(returns_dict, cvar_params)
  risk_aversions = np.logspace(-3, 1, 25)[::-1]
  rows = []
  for ra in risk_aversions:
      problem.params.update_risk_aversion(ra)
      problem.risk_aversion_param.value = ra
      result, portfolio = problem.solve_optimization_problem(SOLVER_SETTINGS)
      row = dict(result); row["risk_aversion"] = float(ra)
      w = np.asarray(portfolio.weights).flatten()
      for t, wv in zip(returns_dict["tickers"], w):
          row[f"w_{t}"] = float(wv)
      row["cash"] = float(np.asarray(portfolio.cash).squeeze())
      rows.append(row)
  results_df = pd.DataFrame(rows)
  ```

  Never call `create_efficient_frontier` *and* run this loop — that doubles the solve count.

  **Do not substitute `evaluate_all_linear_combinations`** — it is a weight-grid sweep without the optimizer, not a frontier replacement.
- All settings must be Pydantic objects (`ReturnsComputeSettings`, `ScenarioGenerationSettings`, `KDESettings`, `CvarParameters`). Do not pass plain dicts.
- Import cuFOLIO modules from the installed `cufolio` package: e.g. `from cufolio import cvar_optimizer, cvar_utils, backtest, utils, rebalance, portfolio`, `from cufolio.settings import ReturnsComputeSettings, ScenarioGenerationSettings, KDESettings, ApiSettings`, `from cufolio.cvar_parameters import CvarParameters`.
- For fixed-schedule rebalancing via `drift_from_optimal` with `threshold=0`, set `plot_title` to reflect the strategy (e.g. "Monthly Rebalancing") instead of the default.

## Examples

- *"Build the optimal portfolio from the S&P 500."* → load data, compute LOG returns, generate KDE scenarios on GPU, `CvarParameters(w_min=0.0, w_max=1.0, c_max=0.0, confidence=0.95)`, solve with the cuOpt `SOLVER_SETTINGS`; report the diversified allocation, expected return, and CVaR.
- *"Plot the efficient frontier."* → `cvar_utils.create_efficient_frontier(..., ra_num=25, show_discretized_portfolios=False)` (Trap 4); present `(results_df, fig)`.
- *"Give me a weights-by-risk-aversion table."* → use the manual solve loop (Trap 3), not `create_efficient_frontier`.
- *"Backtest a monthly rebalancing strategy."* → `rebalance.rebalance_portfolio(..., re_optimize_criteria={"type": "drift_from_optimal", "threshold": 0, "norm": 1})` then `re_optimize(transaction_cost_factor=...)`.

## Limitations

- Requires an NVIDIA GPU with cuOpt + cuML; there is no CPU fallback (CPU solvers are intentionally disallowed — see Solver).
- The default S&P 500 dataset is a historical snapshot and may omit current constituents; unavailable tickers are dropped (see Data).
- Known upstream library quirks (inverted `CvarParameters` weight-bound defaults; `create_efficient_frontier` returning no weights and crashing on the discretized overlay) are worked around via the **Traps** above, not yet fixed upstream.
