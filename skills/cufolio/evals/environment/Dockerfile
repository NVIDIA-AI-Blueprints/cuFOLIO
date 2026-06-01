FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/cufolio-eval

RUN python -m pip install --no-cache-dir \
    ragas~=0.4.0 \
    langchain-nvidia-ai-endpoints>=1.0 \
    numpy>=2.0.0 \
    pandas>=2.0 \
    cvxpy>=1.7.0 \
    scikit-learn>=1.5 \
    seaborn>=0.13 \
    yfinance>=0.2.0 \
    pydantic>=2.12.5 \
    matplotlib

COPY pkg/ /opt/cufolio-eval/
COPY input/ /workspace/

RUN mkdir -p /workspace/skills /workspace/input /workspace/output \
    /logs/verifier /logs/agent

WORKDIR /workspace
