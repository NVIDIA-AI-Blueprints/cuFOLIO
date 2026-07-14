# Changelog

All notable changes to this project are documented here, one PR per entry,
newest first, dated by merge to `main`. Backfilled from git history.

## 2026-07-03 — PR #54

Clarify daily return units and add a GPU-vs-CPU composition comparison in
`notebooks/mean_variance_basic.ipynb`.

- Added a units note and per-statistic labels: all figures derive from 1-day
  LOG returns and are daily, not annualized; added an annualized block
  (×252 log return, exponential conversion to simple return, √252 scaling for
  volatility/Sharpe).
- Added a portfolio-composition comparison cell (direct cuOpt vs CVXPY-cuOpt
  vs CLARABEL weights side by side with per-weight differences), complementing
  the metrics-only `compare_results`.
- Renamed the CVXPY-API solve results to `cvxpy_gpu_*` so the direct cuOpt
  solve no longer shadows them.
- Notebook re-executed end-to-end on GPU (cuOpt 26.06, fresh S&P 500 snapshot).

## 2026-06-26 — PR #52

Reconcile the project and package version strings.

- Set `pyproject` `[project].version` and `src/__init__.py` `version`
  (exported as `cufolio.version`) both to `26.4`, replacing the disagreeing
  `25.10` / `1.0.0` and aligning with the cuOpt 26.4 release line and the
  `cuopt-cu13==26.4.*` dependency-pin style.
- Regenerated `uv.lock`.

## 2026-06-26 — PR #50

Add cuOpt SOCP variance-cap support to the Mean-Variance optimizer.

- Added a direct cuOpt SOCP path for Mean-Variance variance-cap constraints:
  the variance cap is modeled as a convex quadratic (QCQP) row via
  `QuadraticExpression` that cuOpt converts to second-order-cone form and solves
  with the barrier method (`src/mean_variance_optimizer.py`,
  `src/mean_variance_parameters.py`).
- Added `_covariance_for_quadratic_terms` to sanitize the covariance for
  QP/SOCP terms (shape/finiteness checks, symmetrization, and PSD enforcement
  via a minimum-eigenvalue diagonal floor).
- Bumped cuOpt/cuML to 26.06 for the `cuda12` extra and added a cuOpt-only
  `cuda13-socp` extra (`cuopt-cu13==26.6.*`) for CUDA 13 SOCP solves, since
  `cuml-cu13` 26.06 is not yet published; `cuda13` still tracks 26.04.
- Switched cvxpy from the pinned git-master build to the released
  `cvxpy>=1.9.2` and dropped the `[tool.uv.sources]` git pin.
- Added SOCP unit + GPU tests (`tests/test_core.py`), a SOCP benchmark workflow
  (`tests/benchmarks/benchmark_workflows.py`, `tests/benchmarks/thresholds.toml`,
  `tests/test_skill_benchmarks.py`), and the `mean-variance-socp-var-limit`
  eval case (`skills/cufolio/evals/evals-full.json`, now 10 cases).
- Updated docs: `skills/cufolio/SKILL.md`,
  `skills/cufolio/references/workflows/agent_recipes.md`,
  `skills/cufolio/skill-card.md`, `skills/cufolio/evals/EVAL.md`, and the root
  `README.md` install instructions for the new extra.

## 2026-06-05 — PR #47

Add the public Streamlit rebalancing demo.

- Added `demo/rebalancing_streamlit_app.py`, shared demo parameters, demo
  requirements, Streamlit theme config, and supporting architecture / benchmark
  / animation media.
- Updated the root README and demo README with Docker port publishing,
  dataset-download, and remote Streamlit launch instructions.

## 2026-06-02 — PR #46

Add the cuFOLIO agent skill for the NVIDIA skills catalog.

- Added agent/plugin manifests, `skills/cufolio/SKILL.md`, skill card,
  benchmark docs, eval definitions, and workflow recipes.
- Added skill validation workflow/utilities plus benchmark and static skill tests.
- Hardened cuFOLIO APIs and docs around scenario inputs, portfolio behavior,
  and benchmark workflows.

## 2026-05-28 — PR #45

Fix efficient-frontier benchmark behavior for zero-risk portfolios.

- Treat zero-volatility efficient-frontier points as undefined Sharpe ratios
  instead of allowing `0/0` to participate in max-Sharpe selection.
- Added regression coverage in `tests/test_core.py`.

## 2026-05-27 — PR #41

Refresh source workflows for backtesting and rebalancing.

- Updated `src/backtest.py`, `src/rebalance.py`, and related CVaR utility logic.
- Added tests covering the refreshed workflow behavior.

## 2026-05-26 — PR #42

Enhance notebook setup and reduce CI/logging noise.

- Updated the main CI workflow and `notebooks/launchable.ipynb` setup flow.
- Reduced noisy data-download logging in `src/utils.py`.

## 2026-05-04 — PR #37

Add `CHANGELOG.md`, backfilled from `git log --merges --first-parent origin/main`.
One entry per merged PR, newest first, `## YYYY-MM-DD — PR #N` headers.
No source/config changes.

## 2026-04-22 — PR #40

Fix invalid notebook JSON to satisfy the `nbformat` schema.

- Regenerated / repaired notebook JSON for launchable, efficient-frontier,
  and mean-variance notebooks.

## 2026-04-22 — PR #39

Sync launchable notebooks with source updates and add Mean-Variance coverage.

- Updated Brev and launchable notebook content/setup.
- Added related dependency, license, utility, and test updates.

## 2026-04-22 — PR #38

Update launchable notebook API usage for cuOpt 26.4 and the Pydantic settings
migration.

## 2026-04-22 — PR #36

Fix notebook cuOpt presolve settings for cuOpt 26.4 compatibility.

- Changed notebook presolve values from `False` to `0` where required.

## 2026-04-21 — PR #34

**Breaking — minimum Python bumped from `>=3.10` to `>=3.11`.** Required by
`cuml-cu13==26.4`. Also updates `target-version` in the black config and
`CONTRIBUTING.md`.

## 2026-04-20 — PR #33

Switch `cvar_basic.ipynb` to render the architecture diagram via PNG to
avoid SVG rendering issues in `nbconvert` output.

## 2026-04-20 — PR #32

Large bundled change:

- **Added** — Mean-Variance (Markowitz) optimizer with CVXPY and cuOpt
  backends; `notebooks/mean_variance_basic.ipynb`.
- **Added** — `BaseOptimizer` / `BaseParameters` classes and Pydantic
  settings models (`ReturnsComputeSettings`, `ScenarioGenerationSettings`,
  `KDESettings`, `ApiSettings`) under `cufolio.settings`.
- **Added** — SP100, DOW30, and Global Titans dataset download support in
  `utils.download_data`.
- **Added** — `compute_absolute_returns` (simple-diff semantics) as a
  distinct function.
- **Breaking — API rename** — `compute_abs_returns` → `compute_linear_returns`
  (pct-change). Previous `diff` semantics now live in `compute_absolute_returns`.
- **Breaking — settings API** — `calculate_returns`, `generate_cvar_data`,
  and optimizer constructors now take Pydantic settings models instead of
  plain dicts.
- **Breaking — dependency pins** — `cuml-cu{12,13}` and `cuopt-cu{12,13}`
  bumped from `25.12.*` to `26.4.*` / `26.04.*`; `pydantic>=2.12.5` added.
- **Changed** — `optimize_market_regimes()` unified to support both CVaR
  and mean-variance.
- **Fixed** — cuML KDE sampler returns numpy arrays via `using_output_type`.
- **Fixed** — `quad_form` numerical issues resolved via `psd_wrap` on the
  covariance matrix.
- Notebooks rewritten against the new API; outputs regenerated.

## 2025-12-17 — PR from fork (phuo-nv)

- **Changed** — cuOpt model construction switched to `LinearExpression` to
  avoid recursion depth errors on large problems.
- **Changed** — refactored returns-calculation methods; made exception
  handling explicit.
- **Added** — KDE timing instrumentation in the CVaR pipeline.
- **Fixed** — lazy imports in `cvar_optimizer` to avoid eager GPU-module
  load at import time.

## 2025-12-02 — PR #28

Small touch-up of `notebooks/launchable.ipynb` (1 file, +161/-30).

## 2025-12-02 — PR #27

Minor launchable-notebook fix (4 files, +33/-10).

## 2025-12-02 — PR #26

Major launchable-notebook iteration: GPU checks, setup steps, regenerated
outputs (7 files, +3095/-173).

## 2025-12-01 — PR #25

Rename / replace the architecture diagram asset (`docs/arch_diagram.png`).

## 2025-12-01 — PR #24

Refresh notebook outputs to match latest src behavior (4 files, +213/-171).

## 2025-11-26 — PR #23

Final pass on the stock-data directory setup (4 files, +1596/-1813) —
regenerated dataset artifacts and README polish.

## 2025-11-25 — PR #22

Add a bundled dataset file (1 file, +432/-12).

## 2025-11-25 — PR #21

Polish on the stock-data directory layout (3 files, +15/-2).

## 2025-11-25 — PR #20

Initial `data/stock_data/` scaffolding (5 files, +51/-68).

## 2025-11-25 — PR #19

Small source touch-up (1 file, +7/-6).

## 2025-11-25 — PR #18

Launchable-notebook refinements (7 files, +1187/-1326).

## 2025-11-24 — PR #17

Add Brev-deployable `notebooks/launchable.ipynb` (9 files, +4855/-747).

## 2025-11-24 — PR #16

Finalize `uv.lock` migration and sequence diagram (5 files, +1377/-1091).

## 2025-11-23 — PR #15

Refine `uv.lock` sequence diagram docs (6 files, +67/-49).

## 2025-11-21 — PR #14

Commit `uv.lock` and migrate project to `uv` as the dependency manager;
add a sequence diagram documenting the dependency-management flow
(5 files, +2928/-6).

## 2025-11-21 — PR #13

Add `release-2512.yml` workflow, `docs/arch_diagram.png`, large notebook
regeneration, and `src/` cleanup (11 files, +1505/-2082).

## 2025-11-20 — PR #12

Small notebook fixes in `cvar_basic` and `rebalancing_strategies`
(2 files, +4/-4).

## 2025-11-19 — PR #11

Notebook regeneration and src cleanup in backtest / rebalance / utils
(5 files, +703/-912).

## 2025-11-19 — PR #10

Notebook and src refinements across cvar_basic, backtest, cvar_utils,
portfolio, rebalance, utils (7 files, +304/-141).

## 2025-11-17 — PR #9

README additions (+36 lines).

## 2025-11-13 — PR #8

`src/readme.md` fix (1 file, +2/-2).

## 2025-11-12 — PR #7

Remove `deploy/1_Deploy_Template.ipynb`; major `cvar_basic.ipynb` update;
add `notebooks/readme.md`; minor `src/cvar_utils.py` and `src/utils.py`
tweaks (7 files, +1358/-1235).

## 2025-11-12 — PR #6

README polish (1 file, +18/-12).

## 2025-11-12 — PR #5

Remove legacy scaffolding: `Maintainers/`, old `CHANGELOG.md`,
`README_ARCHIVED.md`, empty `customize/` and `evaluate/` readmes; README
polish (10 files, +74/-409).

## 2025-11-07 — PR #4

`LICENSE-3rd-party.txt` edits, remove a `pyproject.toml` entry
(2 files, +5/-7).

## 2025-11-06 — PR #3

Add `LICENSE-3rd-party.txt`; README touch-up (2 files, +119/-2).

## 2025-11-06 — PR #2

README polish (1 file, +2).

## 2025-11-06 — PR #1

**Initial code drop.** Core `cufolio` package: CVaR optimizer
(`cufolio.cvar_optimizer.CVaR`), portfolio backtester
(`cufolio.backtest.portfolio_backtester`), utils, scenario generation
(KDE / Gaussian / historical). Example notebooks: `cvar_basic.ipynb`,
`efficient_frontier.ipynb`, `rebalancing_strategies.ipynb`. `pyproject.toml`
with `cuda12` / `cuda13` extras pinning `cuml-cu{12,13}` and
`cuopt-cu{12,13}`. `README.md`, `CONTRIBUTING.md`, `CODEOWNERS`,
NVIDIA-AI-Blueprints branding and issue templates (22 files, +10885/-25).
