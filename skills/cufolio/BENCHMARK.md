# cufolio — Skill Evaluation Benchmark

<!--
SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

How the `cufolio` skill was evaluated, and the measured uplift it provides over an
agent reasoning from scratch. Required for catalog publication.

> Status: methodology is final; result cells marked _TBD_ are filled from a GPU run
> (see "Reproducing" below). The numbers must be regenerated whenever SKILL.md or the
> `cufolio` product changes materially.

## Setup

| | |
|---|---|
| Skill | `cufolio` (instruction-only; drives the installed `cufolio` package) |
| Agents | Claude Code **and** Codex (evaluate both per the publishing guide) |
| Model(s) | _TBD_ (record exact model + version) |
| Harness | NV-BASE (NV-ACES / Harbor) |
| Dataset | [`evals/evals.json`](evals/evals.json) — 5 positive + 4 negative cases |
| Hardware | NVIDIA GPU (cuOpt + cuML); record GPU model |
| Data | S&P 500 daily prices via `cufolio.utils.download_data` |

## Metrics

NV-BASE emits five evaluators that roll up into the five NVIDIA dimensions:

| Evaluator | Kind | Dimension |
|---|---|---|
| `skill_execution` | deterministic | Correctness |
| `skill_efficiency` | deterministic | Efficiency |
| `accuracy` | LLM judge (5-criterion) | Correctness |
| `goal_accuracy` | full-conversation judge | Effectiveness |
| `behavior_check` | per-step YES/NO | Effectiveness |
| (scan: prompt-injection/secrets/PII) | NV-CARPS | Security |
| (trigger on positive / silence on negative) | discoverability | Discoverability |

## Track A — Agent uplift (with vs. without the skill)

Each task run with the skill installed and again with it removed (baseline).

| Metric | Without skill | With skill |
|---|---|---|
| Positive tasks completed (goal_accuracy) | _TBD_ | _TBD_ |
| Behavior steps passed (behavior_check) | _TBD_ | _TBD_ |
| Trigger accuracy — fires on the 5 positives | _TBD_ | _TBD_ |
| Trigger accuracy — silent on the 4 negatives | _TBD_ | _TBD_ |
| Avg tokens / task | _TBD_ | _TBD_ |
| Avg wall-clock / task | _TBD_ | _TBD_ |

Expected qualitative uplift (what the skill encodes that a baseline agent misses):
forcing `c_max=0.0` to avoid the all-cash optimum (Trap 2), passing
`show_discretized_portfolios=False` (Trap 4), using the manual loop only when weights
are needed (Trap 3), and always solving on GPU with cuOpt (`SOLVER_SETTINGS`).

## Track B — Skill performance standards (Layer 3)

Deterministic end-to-end runs of the documented workflows, graded against
[`evals/thresholds.toml`](evals/thresholds.toml). Source: `tests/test_skill_benchmarks.py`.

| Workflow | Standard | Result |
|---|---|---|
| build-optimal | non-degenerate (not all-cash), sum(w)≈1, cuOpt, < 60s | _TBD_ |
| efficient-frontier | 25 points, return monotonic in CVaR, no `sum_to_one` crash | _TBD_ |
| weights-table | per-asset weight columns present | _TBD_ |
| backtest | optimized Sharpe > equal-weight Sharpe | _TBD_ |
| rebalance | ≥1 rebalance date, cumulative value series produced | _TBD_ |

## Reproducing

```bash
# Track B (no API key; needs GPU). Prints a metrics table + PASS/FAIL:
uv run pytest -m gpu tests/test_skill_benchmarks.py -v
uv run python skills/cufolio/evals/benchmark_workflows.py --check

# Track A (needs NVIDIA_INFERENCE_KEY + GPU), per evals/EVAL.md:
nv-base validate --external skills/cufolio
```
