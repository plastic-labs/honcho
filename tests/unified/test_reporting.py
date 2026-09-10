"""Tests for how a unified run is reported to Discord and the job summary.

Discord rejects an over-long payload with a 400, which loses the whole
notification, so the size behavior here is worth pinning down.
"""

from __future__ import annotations

import pytest

from tests.unified.runner import (
    DISCORD_MAX_CONTENT,
    RunArtifact,
    RunArtifacts,
    StepFailure,
    TestOutcome,
    artifact_line,
    artifact_lines,
    clamp_lines,
    failure_lines,
    gha_run_lines,
)

_PREFIX = "unified-test-results/2026-09-03/1123-merge-abc1234-33779689337-1"


def _presigned(name: str, token_len: int) -> RunArtifact:
    """A presigned URL of realistic shape; OIDC session tokens dominate its length."""
    url = (
        f"https://honcho-unified-tests.s3.amazonaws.com/{_PREFIX}/{name}"
        "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Expires=259200"
        f"&X-Amz-Security-Token={'t' * token_len}&X-Amz-Signature={'0' * 64}"
    )
    return RunArtifact(key=f"{_PREFIX}/{name}", url=url)


def _discord_lines(artifacts: RunArtifacts) -> list[str]:
    """Mirror of the Discord report the runner assembles."""
    return [
        "⚠️ **Unified Test Results**",
        "Results: 35/41 passed, 6/41 failed",
        "Execution time: 1015.42s",
        *artifact_line("View Complete Results", artifacts.results),
        *gha_run_lines(),
        *([f"Reasoning traces: `{artifacts.traces.key}`"] if artifacts.traces else []),
    ]


@pytest.fixture
def in_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_RUN_ID", "33779689337")
    monkeypatch.setenv("GITHUB_REPOSITORY", "plastic-labs/honcho")


def test_clamp_lines_leaves_a_short_report_alone() -> None:
    lines = ["one", "two", "three"]
    assert clamp_lines(lines, DISCORD_MAX_CONTENT) == "one\ntwo\nthree"


def test_clamp_lines_drops_the_longest_line_not_the_last() -> None:
    """The Actions link is short and leads everywhere; a presigned URL is neither."""
    lines = ["head", "x" * 100, "[View GHA](url)"]
    assert clamp_lines(lines, 40) == "head\n[View GHA](url)"


def test_clamp_lines_preserves_display_order() -> None:
    lines = ["a", "y" * 50, "b", "c"]
    assert clamp_lines(lines, 10) == "a\nb\nc"


@pytest.mark.usefixtures("in_actions")
@pytest.mark.parametrize("token_len", [0, 400, 900, 1400, 1800])
def test_discord_report_never_exceeds_the_webhook_limit(token_len: int) -> None:
    """Regression: six judge verdicts plus two presigned URLs returned a 400."""
    artifacts = RunArtifacts(
        results=_presigned("results.json", token_len),
        traces=_presigned("unified-reasoning-traces.jsonl", token_len),
    )
    sent = clamp_lines(_discord_lines(artifacts), DISCORD_MAX_CONTENT)
    assert len(sent) <= DISCORD_MAX_CONTENT


@pytest.mark.usefixtures("in_actions")
@pytest.mark.parametrize("token_len", [0, 400, 900, 1400, 1800])
def test_actions_link_always_survives_clamping(token_len: int) -> None:
    """However long the presigned URLs get, the run stays reachable."""
    artifacts = RunArtifacts(
        results=_presigned("results.json", token_len),
        traces=_presigned("unified-reasoning-traces.jsonl", token_len),
    )
    sent = clamp_lines(_discord_lines(artifacts), DISCORD_MAX_CONTENT)
    assert (
        "[View GHA](https://github.com/plastic-labs/honcho/actions/runs/33779689337)"
        in sent
    )


@pytest.mark.usefixtures("in_actions")
def test_discord_report_omits_per_test_failures() -> None:
    """Failure detail belongs in the job summary the Actions link points at."""
    artifacts = RunArtifacts(results=_presigned("results.json", 400))
    assert not any("**Failures**" in line for line in _discord_lines(artifacts))


def test_gha_lines_are_empty_outside_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    assert gha_run_lines() == []


def test_artifact_line_falls_back_to_the_key_when_presigning_failed() -> None:
    assert artifact_line("Traces", RunArtifact(key="k/x.jsonl")) == [
        "Traces: `k/x.jsonl`"
    ]
    assert artifact_line("Traces", None) == []


def test_job_summary_keeps_both_signed_links() -> None:
    artifacts = RunArtifacts(
        results=_presigned("results.json", 900),
        traces=_presigned("unified-reasoning-traces.jsonl", 900),
    )
    lines = artifact_lines(artifacts)
    assert len(lines) == 2
    assert all("https://" in line for line in lines)


def test_failure_lines_reports_every_failure_in_full() -> None:
    reason = "LLM Judge failed: " + "the model did not recall the fact. " * 20
    results = {
        "a.json": TestOutcome("FAIL", 1.0, StepFailure(4, "query", reason)),
        "b.json": TestOutcome("PASS", 1.0),
        "c.json": TestOutcome("INVALID SCHEMA", 0.1),
    }
    lines = failure_lines(results)
    assert lines[:2] == ["", "**Failures**"]
    assert len(lines) == 4  # blank, header, and one bullet per non-PASS
    assert reason in lines[2]  # untruncated
    assert "INVALID SCHEMA" in lines[3]  # falls back to status when no StepFailure


def test_failure_lines_empty_when_everything_passed() -> None:
    assert failure_lines({"a.json": TestOutcome("PASS", 1.0)}) == []
