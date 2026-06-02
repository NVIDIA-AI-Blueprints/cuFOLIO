# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layer 1: static compliance gate for the cufolio agent skill.

Checks that SKILL.md is spec-compliant and that evals.json is well-formed. Pure
stdlib + pytest (no GPU, no cufolio import, no network), so it runs in the normal
CI lane and is the fast "unit test for the skill" guard.
"""

import json
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SKILL_DIR = REPO_ROOT / "skills" / "cufolio"
SKILL_MD = SKILL_DIR / "SKILL.md"
EVALS_JSON = SKILL_DIR / "evals" / "evals.json"

VALID_EXPECTED_SKILLS = {"cufolio", None}
# 'alwaysApply' / 'globs' are rules-only fields; 'usage' is non-standard and is
# NOT read for skill triggering (only 'description' is) — keep keywords there.
FORBIDDEN_FRONTMATTER_KEYS = {"usage", "globs", "alwaysApply"}
MAX_SKILL_LINES = 500


def _frontmatter_lines(text):
    lines = text.splitlines()
    assert lines and lines[0].strip() == "---", (
        "SKILL.md must open with '---' frontmatter"
    )
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return lines[1:i]
    raise AssertionError("SKILL.md frontmatter is not closed with '---'")


@pytest.fixture(scope="module")
def skill_text():
    assert SKILL_MD.exists(), f"missing {SKILL_MD}"
    return SKILL_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(skill_text):
    return _frontmatter_lines(skill_text)


@pytest.fixture(scope="module")
def evals():
    assert EVALS_JSON.exists(), f"missing {EVALS_JSON}"
    return json.loads(EVALS_JSON.read_text(encoding="utf-8"))


def test_name_matches_directory(frontmatter):
    names = [
        ln.split(":", 1)[1].strip() for ln in frontmatter if ln.startswith("name:")
    ]
    assert names, "frontmatter missing 'name:'"
    assert names[0] == SKILL_DIR.name == "cufolio", (
        "frontmatter 'name' must match the skill directory name"
    )


def test_has_description(frontmatter):
    assert any(ln.startswith("description:") for ln in frontmatter), (
        "frontmatter missing 'description:' (the only field used for triggering)"
    )


def test_has_license(frontmatter):
    # NV-BASE Tier 1 requires a non-empty 'license' field in the frontmatter.
    licenses = [
        ln.split(":", 1)[1].strip() for ln in frontmatter if ln.startswith("license:")
    ]
    assert licenses and licenses[0], (
        "frontmatter must include a non-empty 'license' (NV-BASE Tier 1 requirement)"
    )


def test_no_forbidden_frontmatter_keys(frontmatter):
    for ln in frontmatter:
        key = ln.split(":", 1)[0].strip()
        assert key not in FORBIDDEN_FRONTMATTER_KEYS, (
            f"frontmatter must not include '{key}'"
        )


def test_under_line_budget(skill_text):
    n_lines = len(skill_text.splitlines())
    assert n_lines < MAX_SKILL_LINES, (
        f"SKILL.md is {n_lines} lines (limit {MAX_SKILL_LINES})"
    )


# NV-BASE/NV-ACES contract: ground_truth describes the useful outcome, not
# activation metadata or a scoring rubric. These substrings are disallowed.
DISALLOWED_GROUND_TRUTH = (
    "should trigger",
    "should not trigger",
    "must not activate",
    "must not trigger",
    "did not use",
    "test case",
    "handled by",
)


def test_evals_is_nonempty_array(evals):
    # NV-BASE/NV-ACES contract: evals.json is a top-level JSON array of cases.
    assert isinstance(evals, list) and evals, (
        "evals.json must be a non-empty JSON array"
    )


def test_eval_case_schema(evals):
    required = ("id", "question", "expected_skill", "ground_truth", "expected_behavior")
    seen = set()
    for case in evals:
        for field in required:
            assert field in case, f"case {case.get('id')!r} missing field '{field}'"
        case_id = case["id"]
        assert case_id and case_id not in seen, f"duplicate or empty id: {case_id!r}"
        seen.add(case_id)
        assert isinstance(case["question"], str) and case["question"].strip()
        assert case["expected_skill"] in VALID_EXPECTED_SKILLS, (
            f"{case_id}: invalid expected_skill {case['expected_skill']!r}"
        )
        assert isinstance(case["ground_truth"], str) and case["ground_truth"].strip()
        assert isinstance(case["expected_behavior"], list) and case["expected_behavior"]
        if "expected_script" in case:
            assert case["expected_script"] is None or isinstance(
                case["expected_script"], str
            )
        if "should_trigger" in case:
            assert isinstance(case["should_trigger"], bool)


def test_negative_cases_are_well_formed(evals):
    # Negatives must be null-skill, should_trigger=false (contract requirement).
    for case in evals:
        if case["expected_skill"] is None:
            assert case.get("should_trigger") is False, (
                f"{case['id']}: negative case must set should_trigger=false"
            )


def test_ground_truth_describes_outcome(evals):
    # Reject activation-metadata / rubric phrasing in ground_truth (contract).
    for case in evals:
        gt = case["ground_truth"].lower()
        for phrase in DISALLOWED_GROUND_TRUTH:
            assert phrase not in gt, (
                f"{case['id']}: ground_truth must describe the outcome, not "
                f"contain {phrase!r}"
            )


def test_has_positive_and_negative_cases(evals):
    expected_skills = [c["expected_skill"] for c in evals]
    assert any(s == "cufolio" for s in expected_skills), (
        "need >= 1 positive case (expected_skill == 'cufolio')"
    )
    assert any(s is None for s in expected_skills), (
        "need >= 1 negative case (expected_skill == null)"
    )
