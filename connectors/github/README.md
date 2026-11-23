# GitHub connector

Reads GitHub repositories' history as Honcho ingest records, and writes them to
JSONL files or sends them to the ingest API through the Python SDK. This is the
backfill half of a GitHub connector: it reads history on demand. It has no
webhooks or stored read position yet; running it again is safe (see
[Running again](#running-again)).

> **Experimental.** Ingest into auto workspaces is experimental, and so is
> this connector: its records, its options and the ingest API it sends to may
> change or stop working without notice. The CLI commands that import and review
> its output run only with `HONCHO_EXPERIMENTAL=1` set.

## What it needs

A token, in a file, that can read the repositories. A fine-grained token with
read-only **Metadata**, **Contents** (for commit history), **Issues** and
**Pull requests** access is enough. It
only uses GitHub's GraphQL API, and waits out rate limits.

## Use

```bash
# The repositories an export would read
honcho-github repos --token-file ~/.github.token --org plastic-labs

# Export to files (refuses a directory inside a git checkout)
honcho-github export --token-file ~/.github.token \
    --repo plastic-labs/honcho [--repo OWNER/NAME ...] [--org LOGIN ...] \
    [--since 2026-01-01] --out ~/github-export

# Or send to Honcho's ingest API, deriving later
HONCHO_API_KEY=... honcho-github export --token-file ~/.github.token \
    --repo plastic-labs/honcho --ingest-url http://127.0.0.1:8000 \
    --workspace acme --api-key-env HONCHO_API_KEY
```

`--org` reads an organization's or account's **public** repositories, skipping
archived ones and forks. A private repository is read only when it's named with
`--repo`: who can see a repository isn't who can query the workspace.

`--bots` picks the bot records sent with `--ingest-url`: `interacted` (the
default) sends a bot's record only when a person replied to it or @-mentioned
the bot about it in the same issue or pull request, `all` sends every one,
`none` none. On our repositories that holds back about 90% of bot records and
half of all text. `--out` always writes every record; `honcho ingest import
--bots` makes the same choice when the file is imported.

`--since` keeps the issues and pull requests updated since that date, each one
whole (all its comments and reviews), and the commits made since it.

## What a record holds

A repository is one container, so one session. Each record opens with a line
naming what it belongs to by number only (`Comment on PR #12:`): a title can
change, and a record's text must not change unless its author edits it.

| Ingest field | From GitHub |
|---|---|
| `source`, `source_instance` | `github`, the host with dots as dashes (`github-com`). The host, not the organization: an account is one person in every organization. |
| `source_id` | the item's GraphQL node ID |
| `created_at`, `revised_at` | when it was posted (a review: submitted; a commit: committed), and when its text was last edited |
| `container` | the repository; `parent_id` is its owner |
| `author` | the account's numeric ID and login, and `bot: true` for GitHub Apps and `[bot]` accounts. A deleted account (ghost) has no author. |
| `participants` | the accounts it @-mentions (outside code), and for a review reply the author of the comment it answers |
| `content` | the header line, then the text (see below) |
| `metadata` | `repo`, `kind`, `url`, `thread` (`pr:12`, `issue:3`: the conversation inside the repository, which scopes mentions), the issue or PR number and its state at export time, a review's verdict, a review comment's `path` and `line`, a commit's `sha`, and `ai_assisted` when a coding assistant is credited |

Records, per repository:

- each **issue** and **pull request**: its title and body;
- each **comment** on them;
- each submitted **review**: its verdict and text. An approval or change request
  with no text is kept with reasoning off: context about who reviewed, nothing
  to derive. A comment-only review with no text is skipped (its inline comments
  are records);
- each **inline review comment**, with its file and line;
- each **default-branch commit**: what shipped, with the top directories its
  pull request touched. Branch-sync merges are skipped, and so are commits whose
  author has no GitHub account.

Text is cleaned first: CodeRabbit's generated summary (which it writes into the
human author's PR body), HTML comments, `Co-Authored-By` and `Signed-off-by`
trailers and the Claude Code footer are removed, so a bot's words never become
a person's. A bot's own records also drop their collapsed `<details>` blocks
(run configuration, prompts for AI agents, walkthroughs): on our repositories
that's about 70% of bot text, and none of it is conversation.

Not exported: timeline events (labels, assignments, closing), Discussions,
reactions and diffs.

Files written by `--out`: `records.jsonl` (ingest records, in send order),
`accounts.jsonl` (each account's ID, login, profile name, type and bot flag),
`containers.jsonl`, and `export.json` (counts, what was skipped, and the
GraphQL rate-limit cost).

## Running again

Ingest deduplicates on `source_id`, so an overlapping or repeated export adds
only what's new. What changed at GitHub since the first export doesn't replace
what's stored yet: ingest refuses an edited comment (`edit_not_supported`), and
a record whose text changed without an edit, such as a review later dismissed
or an issue renamed (its opening post carries the title), comes back as
`content_conflict`. Both are reported as rejections and the export exits
non-zero; the stored version stays. A repository that can't be read completely
sends nothing, and the export exits non-zero.

Pending reviews and review comments, which only their author can see, are never
exported, even with that author's token.

## Limits

- GitHub's API returns each item's current text, and never a deleted one, so an
  export has no edit history or deletions.
- A review's verdict and a pull request's state are as of the export.
- Logins are as of the export; the numeric ID is what ties an account together.
