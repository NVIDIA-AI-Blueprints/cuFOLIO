# Publishing cufolio to the build.nvidia.com skills catalog

<!--
SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

This skill is authored **here** (`skills/cufolio/`) and mirrored to the separate public
catalog at **github.com/NVIDIA/skills** (which feeds build.nvidia.com). Registration is a
one-time PR to that repo; afterwards a daily sync pulls `skills/` from this repo, runs the
NV-CARPS pre-catalog gate (scan + validate + sign + skill-card), and mirrors the signed
skill into the catalog.

## 0. Which path applies (read first)

NVCARPS has two distribution tracks — don't mix their steps:

- **External public catalog (our primary path).** For GitHub-canonical repos under an
  NVIDIA org (we are `NVIDIA-AI-Blueprints/*`). Register via the `components.d` PR below and
  trigger NV-CARPS with `/nvskills-ci`; the daily sync publishes to **github.com/NVIDIA/skills**
  (build.nvidia.com). Confluence runbook: *GitHub-First — Outbound Repos Onboarding*
  (`nvidia.atlassian.net/wiki/spaces/GAIT/pages/3483240468`).
- **Internal NVCARPS ingestion (GitLab).** For repos canonical on gitlab-master.nvidia.com:
  grant `NVCARPS_CI` to the repo at **aitoolsonboarding.nvidia.com** (adds the
  `nvcarps-webhook-bot` + webhook), enable the *Pipelines must succeed* merge check, then open
  an MR — ingests to **registry.nvidia.com / NICC / nvcarps-mcp** (~15 min). We do **not** do
  this directly; internal NICC visibility comes as a side effect of the external path. Only
  relevant if you take the interim *GitHub→GitLab bridge* (the "One-time migration setup for
  GitHub-first repo" section of the GitLab onboarding page).

Either way the skill lives at `skills/cufolio/` at the repo root, and ingestion attaches a
per-skill signature only when the pipeline passes.

## 1. Registration entry

Create `components.d/cufolio.yml` on a branch of **your fork of NVIDIA/skills**, then open
a PR back to NVIDIA/skills (each component is its own file — no shared list to edit):

```yaml
name: cuFOLIO
repo: NVIDIA-AI-Blueprints/quantitative-portfolio-optimization
description: >-
  GPU-accelerated Mean-CVaR portfolio optimization skills built on NVIDIA cuOpt:
  optimal portfolio construction, efficient frontier, backtesting, and rebalancing.
skills:
  - path: /skills/
    catalog_dir: cufolio
```

> `path: /skills/` is the canonical repo-root location the sync pipeline reads — treat it
> as public API and do not move/rename it without updating this entry.

## 2. Pre-submission checklist (team-owned)

- [ ] **IP Review** complete for the skill commits — all six questions affirmative
      (https://nvidia.atlassian.net/wiki/spaces/OSS/pages/2529034695).
- [ ] **IP-review self-attestation NVBug** filed (ACK of process; license + attribution
      validated; 3rd-party OSS / codec rights; VP approval).
- [ ] **License** is Apache-2.0 (✓ repo `LICENSE`) — or CC-BY-4.0 / dual for docs-only.
      No new license or 3rd-party component introduced (else return to OSRB first).
- [ ] Repo is public under an NVIDIA org (✓ `NVIDIA-AI-Blueprints/*` — inherits OSRB).
- [ ] `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`, `README.md` present (✓).
- [ ] Skill assets present: `SKILL.md`, `evals/evals.json`, `BENCHMARK.md`, `skill-card.md` (✓).
- [ ] Ran locally before submitting:
      - `uv run pytest tests/test_skill.py` (Layer 1 compliance) — green
      - `uv run pytest -m gpu tests/test_skill_benchmarks.py` (Layer 3, on a GPU box) — green
      - `nv-base validate --external skills/cufolio` (the catalog gate) — green
- [ ] Commits are **DCO signed** (`git commit -s`); the NVIDIA/skills PR enforces it.

## 3. Trigger signing + verify

- On the registration PR, a repo maintainer/admin comments **`/nvskills-ci`** to run the
  NVSkills CI pipeline (scan → validate → sign → skill card → BENCHMARK).
- Before merge: confirm the **"Attach NVSkills validation signatures"** commit is present,
  that no unsigned content landed after it, and that the `.sig` matches the changed content.
- After merge: the daily sync mirrors the skill to `nvidia/skills/cufolio/`. Verify it
  appears there and that the Available Skills table shows the upstream SHA.

## Contacts
NVIDIA/skills onboarding: Sayali Kandarkar (skandarkar@nvidia.com) · OSS External Skills PIC:
Moshe Abramovitch (moshea@nvidia.com) · or `#nv-skills-onboarding`.
