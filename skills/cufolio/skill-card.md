# Skill Card — cufolio

<!--
SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Trust-manifest STUB. NVCARPS auto-generates the authoritative SKILLCARD.yaml during
signing (identity, provenance, scan results, evaluation metrics, signature). This file
documents the team-owned fields (behavioral boundaries, runtime, data handling) up front.
-->

## Identity
- **Name:** cufolio
- **Version:** tracks the `cufolio` package (`pyproject.toml` → currently 25.10)
- **License:** Apache-2.0
- **Type:** instruction-only "use the product" skill (ships no executable scripts of its own)
- **Source repo:** NVIDIA-AI-Blueprints/quantitative-portfolio-optimization
- **Skill path:** `skills/cufolio/`
- **nSpect ID:** _pending_ (auto-allocated on registration; not yet available)
- **Signature:** _applied by NVCARPS CI_ (`skill.oms.sig`)

## What it does
GPU-accelerated Mean-CVaR portfolio optimization with NVIDIA cuOpt: returns
computation, KDE/Gaussian scenario generation, CVaR optimization, efficient frontier,
backtesting (Sharpe/Sortino/max drawdown), and dynamic rebalancing — by driving the
installed `cufolio` Python package.

## Behavioral boundaries
- **In scope:** building/optimizing portfolios from price data, CVaR/mean-CVaR
  problems, efficient frontiers, backtests, rebalancing strategies.
- **Out of scope (must not trigger):** generic financial Q&A, price forecasting / ML,
  and non-portfolio optimization (e.g. vehicle routing). See the negative cases in
  [`evals/evals.json`](evals/evals.json).
- **Always** solves on GPU with cuOpt (`{"solver": cp.CUOPT, "solver_method": "PDLP"}`);
  never falls back to CPU solvers.
- Encodes four documented "Traps" so the agent avoids known degenerate/wrong results.

## Runtime requirements
- NVIDIA GPU with cuOpt + cuML; the `cufolio` package installed (Brev launchable or
  `uv sync --extra cuda12|13`). The skill is not runnable without this environment.

## Data handling
- Uses local S&P 500 price CSVs (`data/stock_data/`), downloaded on demand via
  `cufolio.utils.download_data` (yfinance). No secrets or PII; no user data is
  transmitted by the skill itself.

## Evaluation
- See [`BENCHMARK.md`](BENCHMARK.md). Dataset: [`evals/evals.json`](evals/evals.json).
  Standing performance standards: [`evals/thresholds.toml`](evals/thresholds.toml).
