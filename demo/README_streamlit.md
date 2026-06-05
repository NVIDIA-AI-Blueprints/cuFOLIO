# cuFOLIO Rebalancing Demo — Streamlit App

Interactive web application for GPU-accelerated dynamic portfolio rebalancing using Mean-CVaR optimization.

## Application

### Rebalancing Strategies (`rebalancing_streamlit_app.py`)

Dynamic portfolio rebalancing simulation with multiple trigger strategies.

- Rebalancing triggers: loss threshold, drift from target, peak-to-trough decline, buy & hold
- Progressive backtesting with real-time cumulative return plots
- GPU vs CPU solver comparison with KDE timing breakdown
- Advanced mode for technical parameters: windows, transaction costs, turnover, and CVaR limits
- Masked dataset and solver names for conference presentations

## Quick Start

Use `uv` for the current cuFOLIO dependency set:

```bash
# From the repository root
uv sync --extra cuda13        # or: uv sync --extra cuda12
uv pip install -r demo/requirements.txt
uv run python -c 'from cufolio.utils import download_data; download_data("data/stock_data", datasets=["sp500"])'
uv run streamlit run demo/rebalancing_streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

For a standard virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[cuda13]"       # or .[cuda12], or plain -e . for CPU-only setup
pip install -r demo/requirements.txt
python -c 'from cufolio.utils import download_data; download_data("data/stock_data", datasets=["sp500"])'
streamlit run demo/rebalancing_streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Open `http://localhost:8501` for local runs. On a remote GPU instance, forward or expose port `8501` according to your environment.

## Requirements

- Python 3.11+
- Streamlit, Plotly, and Squarify from `demo/requirements.txt`
- CVXPY and cuFOLIO core dependencies from `pyproject.toml`
- Optional: NVIDIA GPU + CUDA with the matching `cuda12` or `cuda13` extra for cuOpt acceleration

## Troubleshooting

**Import Errors**: Run `pip install -e .` or `uv sync` from the repository root so the `cufolio` package is registered.

**Dataset Not Found**: Download a dataset before launching the app:

```bash
uv run python -c 'from cufolio.utils import download_data; download_data("data/stock_data", datasets=["sp500"])'
```

**GPU Solver Unavailable**: The UI can still boot without a GPU, but the full GTC comparison needs cuOpt. Install the matching CUDA extra and run on an NVIDIA GPU instance.
