# Local MCP integration

The optional MCP integration runs this package as a local stdio server. It
exposes the existing NVIDIA cuOpt Mean-CVaR and Mean-Variance workflows to MCP
clients without adding a hosted API, authentication layer, or CPU solver
fallback.

For an executable walkthrough that reuses downloaded S&P 500 prices, run the
MCP section in [`notebooks/cvar_basic.ipynb`](../notebooks/cvar_basic.ipynb).

## Install

Install the MCP extra together with exactly one CUDA extra:

```bash
uv sync --extra cuda12 --extra mcp
```

For a CUDA 13 environment, replace `cuda12` with `cuda13`.

## Prepare the startup dataset

The server loads one trusted CSV or Parquet file at startup. The first/index
column must contain dates; all remaining columns must be ticker symbols and all
usable values must be finite, positive prices. Ticker columns containing missing
observations are excluded at startup and reported by `portfolio_dataset_info`,
matching the package's existing complete-column data workflow.

```text
date,AAA,BBB,CCC
2025-01-02,100.0,80.0,120.0
2025-01-03,100.8,79.6,121.1
2025-01-06,101.2,80.4,120.7
```

The model never supplies a filesystem path and the server does not fetch market
data.

Mean-CVaR requests are capped at 100,000 scenarios and 10,000,000
asset-scenario values to reject accidental memory-exhausting tool calls before
allocation.

## Run

Download the repository's example S&P 500 dataset, or substitute your own
trusted price file:

```bash
uv run python -c \
  "from portfolio_optimization.utils import download_data; download_data('data/stock_data', datasets=['sp500'])"
```

```bash
uv run portfolio-optimization-mcp \
  --data data/stock_data/sp500.csv
```

The process uses stdout only for MCP frames. Package and solver output is
suppressed so native cuOpt messages cannot corrupt the stdio transport; server
logs go to stderr.

## Configure an MCP client

Use an absolute repository and dataset path when a desktop client launches the
server:

```json
{
  "mcpServers": {
    "portfolio-optimization": {
      "command": "/home/user/.local/bin/uv",
      "args": [
        "--directory",
        "/path/to/portfolio-optimization",
        "run",
        "portfolio-optimization-mcp",
        "--data",
        "/path/to/portfolio-optimization/data/stock_data/sp500.csv"
      ]
    }
  }
}
```

## Tools

### `portfolio_dataset_info`

Returns the startup dataset's date range, row count, ticker count, and a bounded
page of matching symbols. Use `query`, `offset`, and `limit` to inspect a large
universe without returning every ticker at once.

```json
{"query": "NV", "offset": 0, "limit": 100}
```

### `portfolio_optimize`

Runs one GPU-only optimization. The objective is either `mean_cvar` or
`mean_variance`. The request can select tickers and a historical regime and can
set position, cash, leverage, group, turnover, cardinality, and hard-risk
parameters supported by the underlying package.

Mean-CVaR example:

```json
{
  "objective": {
    "type": "mean_cvar",
    "risk_aversion": 1.0,
    "confidence": 0.95,
    "scale_risk_aversion": false
  },
  "tickers": ["AAPL", "MSFT", "NVDA"],
  "constraints": {
    "weight_minimum": 0.0,
    "weight_maximum": 0.6,
    "cash_minimum": 0.0,
    "cash_maximum": 0.2,
    "leverage_limit": 1.0
  },
  "scenarios": {
    "method": "kde",
    "count": 10000,
    "seed": 42,
    "kde_bandwidth": 0.01
  }
}
```

Mean-Variance example:

```json
{
  "objective": {
    "type": "mean_variance",
    "risk_aversion": 1.0,
    "scale_risk_aversion": false
  },
  "constraints": {
    "weight_maximum": {"AAPL": 0.25, "others": 0.5},
    "cash_maximum": 0.15
  },
  "time_limit_seconds": 60
}
```

Responses contain JSON-safe ticker weights, cash, objective type, regime,
expected daily return, CVaR or variance, objective value, solve time, and
scenario count where applicable. They also include a process-local `run_id`
that can be passed to the remaining workflow tools without resending weights.

`mean_variance.variance_limit` requires cuOpt 26.06 or newer. The standard
`cuda13` extra currently tracks cuOpt 26.04, so use `cuda12` or
`cuda13-socp` when exercising that hard-limit path. `cuda13-socp` does not
include cuML and therefore is not the full GPU KDE environment.

### `portfolio_analyze`

Reports expected return, variance, volatility, signed CVaR loss, and
concentration for explicit weights or a prior optimization run:

```json
{
  "portfolio": {"source": "run", "run_id": "opt_..."},
  "confidence": 0.95,
  "scenarios": {"method": "gaussian", "count": 1000, "seed": 42}
}
```

### `portfolio_simulate`

Returns fixed-portfolio percentiles, mean, spread, and optional dollar-loss
probabilities. Dollar fields require `equity_usd` on the portfolio source.
Multi-day horizons use independently sampled daily log returns and are an
approximation.

```json
{
  "portfolio": {
    "source": "run",
    "run_id": "opt_...",
    "equity_usd": 100000
  },
  "horizon_trading_days": 21,
  "loss_thresholds_usd": [5000, 10000],
  "scenarios": {"method": "gaussian", "count": 1000, "seed": 42}
}
```

### `portfolio_backtest`

Runs historical, KDE-simulated, or Gaussian-simulated comparisons. The first
portfolio is the test portfolio; later entries and an automatic equal-weight
portfolio are benchmarks.

```json
{
  "portfolios": [
    {
      "name": "optimized",
      "portfolio": {"source": "run", "run_id": "opt_..."}
    }
  ],
  "method": "historical"
}
```

Run handles live only in the current server process. Restarting the server or
evicting an old result requires rerunning `portfolio_optimize`.

## Scope

This is a local developer integration:

- stdio only;
- one trusted startup dataset;
- one process and no durable state;
- no HTTP endpoint, authentication, multi-user tenancy, market-data provider,
  order execution, or trade recommendation;
- NVIDIA cuOpt only, with no CPU solver fallback.
