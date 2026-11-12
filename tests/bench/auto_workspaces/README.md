# Auto workspaces benchmark

The data and scoring used to judge the auto-workspaces organizer: does it link
one person's accounts across sources without merging two people, group records
into the right scopes, and spot bots, at a cost and review load people accept?
No organizer change ships without beating the rules-only baseline here.

```bash
python -m tests.bench.auto_workspaces.run                  # synthetic, dev split
python -m tests.bench.auto_workspaces.run --split heldout  # judge on held-out
python -m tests.bench.auto_workspaces.run --json           # machine-readable
python -m tests.bench.auto_workspaces.run --write DIR      # save the fixtures
python -m tests.bench.auto_workspaces.run --data DIR       # score a saved dataset
```

## Data

A dataset is what a connector would send, what it could know, and the answers:

- `records.jsonl`: ingest API records, exactly as `POST /v3/workspaces/{id}/ingest`
  takes them, each with its case, its unit and its operation (`create`, `edit`,
  `delete`).
- `profiles.jsonl`: per account, what a connector could know besides the
  records: display name, username, email, the source's bot flag.
- `truth.json`: which person each account belongs to, which groups each record
  belongs in, which accounts are bots. An organizer never reads it.

Accounts are keyed the way ingest binds people (`model.account_key`): by
`(source, source_instance, external id)`, or for a person with no id by the
container and the casefolded name, so the same name in two containers is two
accounts until someone links them.

**Synthetic fixtures** (`fixtures.py`) cover the hard cases: same-name people,
different people who share a username across platforms, one person across
Discord, Slack and GitHub (with and without a shared handle),
outside users in a shared channel, account changes, edits and deletes, explicit
and inferred peers mixed, records with no container, a channel that discusses
several customers, coding-agent turns with injected system reminders, and
flagged and unflagged bots. Every id is opaque and independent across sources,
and an organizer gets `Dataset.blind()`: no case or unit labels, no truth.

**Replayed data** (the internal Discord and activity workspaces) is exported to
the same format outside the repository, never committed: it holds internal
conversations and the team's names.

## Replay through ingest

`replay.py` sends a dataset through the real ingest API, oldest first in
batches of 100, with `derive: "later"`, then checks what ingest built: every
account of a stored record (authors and participants) has exactly one peer, an
explicit `peer_id` is that peer, no two accounts share one, and edits are
rejected `edit_not_supported` until edits ship. Anything else fails the run.
The report gives the token count of what was deferred, to approve before
deriving.

```bash
python -m tests.bench.auto_workspaces.replay --base-url http://127.0.0.1:8000 \
    --workspace bench-1 [--data DIR] --report report.json
```

Replay creates `--workspace` as an auto workspace, so name a new workspace (or
one already auto): ingest refuses any other.

Deferring skips derivation, not embedding: with `EMBED_MESSAGES` on, ingest
embeds new messages right away. For a replay that costs nothing, run the
server with `EMBED_MESSAGES=false` or against the mock provider. Only a local
server is accepted without `--allow-remote`.

## Split

Whole units are held out, never single records: a fixture instance, or for
replayed data a person and the channels they post in. A stable hash assigns
30% of units to `heldout` (`split.py`). Build and tune on `dev`; report
`heldout`.

## Metrics

| Metric | How | Why |
| -- | -- | -- |
| Identity | Pairwise precision and recall over accounts; headline F0.5 | A false merge (two people as one) costs more than a false split |
| Grouping | Best-match F1 between true and predicted groups, both ways, size-weighted | Records can belong to several groups |
| Bots | Precision and recall | Bots shouldn't get memory or merge with people |
| Cost | Organizer tokens per 1M ingested tokens | Organizer calls aren't billed to customers |
| Review load | Suggestions per 1,000 records | A person reviews only the calls that matter |

Answer quality (does the dialectic answer better on the organized workspace) is
scored separately with the team side-by-side comparison harness; it needs real
model calls, so it runs only on request.

## Rules-only baseline

`baseline.py`: link accounts that share an email, a username across sources,
or an explicit `peer_id` equal to a username (case kept: peer names are
case-sensitive); trust the source's bot flag and treat the source peer of
author-less records as a bot; one group per container. It never matches
display names or reads content. A username is unique only within its platform,
so it merges two people who happen to share one (`shared_handle`): exact rules
aren't safe rules. On the synthetic data (seed 7, 30 instances per case):

| Split | Identity F0.5 | Precision | Recall | Grouping F1 | Bot recall |
| -- | -- | -- | -- | -- | -- |
| dev | 0.825 | 0.877 | 0.668 | 0.728 | 0.808 |
| held-out | 0.760 | 0.833 | 0.561 | 0.728 | 0.784 |

## Thresholds

Agreed 2026-11-12, before any organizer work. On held-out data, an organizer
must reach all of these and beat the baseline:

- Identity precision at least 0.99, and no false merge in `same_name` or `outside_users`.
- Identity recall at least 0.85.
- Grouping F1 at least 0.90.
- Bots: precision at least 0.95, recall at least 0.90.
- Cost at most 100,000 organizer tokens per 1M ingested tokens.
- Review load at most 20 suggestions per 1,000 records.
