# AGENTS.md — cuFOLIO AI Agent Entry Point

AI agent skills for NVIDIA-accelerated quantitative portfolio optimization. Skills live in **`skills/`** at the repo root and drive the installed `cufolio` package (NVIDIA cuOpt + cuML).

## Skills

- `skills/cufolio/` — GPU-accelerated **Mean-CVaR portfolio optimization** with NVIDIA cuOpt: optimal portfolio construction, CVaR/mean-CVaR optimization, efficient frontier, KDE/Gaussian scenario generation, backtesting, and dynamic rebalancing. Read `skills/cufolio/SKILL.md` for the workflow, defaults, and the non-obvious "Traps". Not for generic finance Q&A, price forecasting/ML, or non-portfolio (e.g. routing) optimization.

## Notes

- Always solve on the cuOpt GPU solver (`cp.CUOPT`, `solver_method="PDLP"`); never fall back to a CPU solver.
- The skill assumes the `cufolio` package is installed (Brev launchable or `uv sync --extra cuda12|13`).
