---
name: pre-pr
description: Prepare a Honcho change for a pull request to plastic-labs/honcho. Invoke before opening a PR, when drafting a PR body, when asked if a branch is PR-ready, or when filling the pull request template. Checks the linked issue, required tests and docs, then writes Description / Proofs / Fixes.
---

# Pre-PR checklist

Do this after the change works, before anyone opens the GitHub PR. Output is a filled template body — do not create the PR.

The template lives at `.github/pull_request_template.md`. Do not add extra sections.

## 1. Issue gate (hard stop)

A PR without a maintainer-approved issue will be closed.

```bash
gh issue view <N> --repo plastic-labs/honcho --json number,title,labels,state
```

Stop if any of these fail:

- no issue number, or the issue is not in `plastic-labs/honcho`
- issue is closed (unless this PR is explicitly reopening it)
- labels do not include `maintainer-approved`

Say which check failed. Do not draft a PR body around it.

## 2. Classify the diff

```bash
git diff main...HEAD --stat
```

Pick one primary kind: bug, feature, docs. Then decide layers:

| Surface touched | Required |
| --- | --- |
| `src/` (non-prompt) | unit tests under the matching `tests/` tree |
| deriver / dialectic / dreamer / LLM path | unit + consider live-llm (`tests/live_llm`) |
| queue, config hierarchy, multi-turn, SDK contract | unified (`uv run python -m tests.unified.run`) |
| `/v3` HTTP or deriver queue behavior | `/verify` skill (runtime, not just pytest) |
| public API, SDK exports, `config.toml` / settings, mintlify `docs/` | documentation in the matching file |

Skip a layer only with a one-line reason (e.g. "docs-only", "comment-only"). "When appropriate" is not a skip.

Invoke `/verify` when the runtime surface moved. Do not restate that skill here.

Lint/type before claiming tests are green: `uv run ruff check src/` → `uv run basedpyright` → the pytest command for the layer.

## 3. Proofs

Collect evidence that belongs in the PR, not in the commit:

- command + pass/fail for what you ran
- a log snippet, screenshot, or file path that shows the new behavior
- for bugs: the failing case before vs after, if you have it

If `/verify` ran, the proofs *are* that session's output. Do not invent green runs.

## 4. Write the body

Fill the description, proofs, checklist portion of the pull request description template.
Make sure to link the related github issue, otherwise the PR will be auto-closed.
