## Description: <br>
Build GPU-accelerated Mean-CVaR portfolios with NVIDIA cuOpt — CVaR optimization, efficient frontier, scenario generation, backtesting, and rebalancing. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Quantitative researchers and engineers use this skill to construct and analyze Mean-CVaR portfolios with NVIDIA's GPU-accelerated cuOpt solver: optimal allocation, efficient frontier generation, strategy backtesting, and dynamic rebalancing. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Quantitative Portfolio Optimization README](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization) <br>
- [Brev Launchable](https://brev.nvidia.com/launchable/deploy?launchableID=env-360InRZzyHqDnJYQKIxaSggF8xI) <br>
- [Eval dataset](evals/evals.json) <br>

## Skill Output: <br>
**Output Type(s):** [Code, API Calls, Analysis] <br>
**Output Format:** [Python code with inline solver output and plots] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- claude-code <br>
- codex <br>

## Evaluation Tasks: <br>
Evaluated against 9 cases (5 positive + 4 negative) with 2 attempts per agent; pass threshold 60%. NVSkills-Eval profile: external. Results pending the GPU agent-eval run (see `evals/EVAL.md` and `BENCHMARK.md`). <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>

Underlying evaluation signals: <br>
- `security`, `skill_execution`, `skill_efficiency`, `accuracy`, `goal_accuracy`, `behavior_check`, `token_efficiency`. <br>

## Evaluation Results: <br>
_Pending the GPU agent-eval run; see `BENCHMARK.md` for the with-skill vs without-skill tables._ <br>

## Skill Version(s): <br>
25.10.00 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
