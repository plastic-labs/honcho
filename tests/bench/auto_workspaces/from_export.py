"""Turn a Discord connector export into a benchmark dataset.

    python -m tests.bench.auto_workspaces.from_export \\
        --export DIR --people FILE --out DIR

`--export` is what `honcho-discord export --out` wrote (`records.jsonl`,
`accounts.jsonl`, `export.json`). `--people` names who each account is:

    {"people": {"<person>": {"discord": ["<username>", ...]}, ...},
     "bots": {"discord": ["<username>", ...]}}

An account the people file doesn't list is its own person. A bot is an account
the source flags, or one the people file lists under `bots` (integrations that
post as ordinary accounts). Each person, or each bot, is one unit of the
held-out split, with every record they wrote. In a chat, people name each
other, so a record on one side can name a person held out on the other; that
person is scored only on their own side, and a false merge between people on
opposite sides counts in neither. (Units that never cross would be whole
connected groups of people, and a team's chat is one group.) Score identity
across sides with `--split all`.

`--only` keeps part of an export: a category name or a channel ID, each with
its threads. Repeatable; without it, the whole export.

No grouping is labeled: `truth.groups_of` stays empty until someone labels
which records belong together, so grouping scores on this data mean nothing.

Exports and the datasets made from them hold internal conversations and
people's names: keep them out of the repository.
"""

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tests.bench.auto_workspaces.model import (
    AccountKey,
    BenchRecord,
    Dataset,
    Profile,
    Truth,
    account_key,
    record_accounts,
)

CASE = "discord"
THREAD_TYPES = frozenset({10, 11, 12})


def _lines(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as lines:
        for line in lines:
            if line.strip():
                yield json.loads(line)


def _containers(export: Path, meta: dict[str, Any], only: set[str]) -> set[str]:
    """The channels and threads `only` names, by category name or channel ID."""
    containers = {c["id"]: c for c in _lines(export / "containers.jsonl")}
    categories: dict[str, str] = meta.get("categories", {})
    kept: set[str] = set()
    for container in containers.values():
        # A thread's channel is its parent, listed or not.
        thread = container["type"] in THREAD_TYPES
        channel_id: str = container["parent_id"] if thread else container["id"]
        channel = containers.get(channel_id) or {}
        category = categories.get(channel.get("parent_id") or "")
        if channel_id in only or category in only:
            kept.add(container["id"])
    if not kept:
        raise ValueError(f"no channel or category in the export matches {sorted(only)}")
    return kept


def convert(
    export: Path, people: dict[str, Any], *, only: set[str] | None = None
) -> Dataset:
    """The dataset for one export, with identity and bot truth from `people`.

    With `only`, just those categories or channels (and their threads), and
    the accounts their records name.
    """
    meta: dict[str, Any] = json.loads((export / "export.json").read_text())
    source: str = meta["source"]
    instance: str = meta["source_instance"]
    kept = _containers(export, meta, only) if only else None

    def key(external_id: str) -> AccountKey:
        return account_key(source, instance, external_id=external_id)

    person_of_username = {
        username: person
        for person, ids in people.get("people", {}).items()
        for username in ids.get(source, [])
    }
    bot_usernames = set(people.get("bots", {}).get(source, []))

    profiles: dict[AccountKey, Profile] = {}
    truth = Truth()
    for account in _lines(export / "accounts.jsonl"):
        k = key(account["id"])
        username: str | None = account.get("username")
        profiles[k] = Profile(
            account=k,
            display_name=account.get("display_name"),
            username=username,
            is_bot=bool(account.get("bot")),
        )
        if account.get("bot") or username in bot_usernames:
            truth.bots.add(k)
            truth.person_of[k] = f"bot:{k}"
        else:
            truth.person_of[k] = person_of_username.get(username or "", f"self:{k}")
        truth.unit_of_account[k] = f"{CASE}:{truth.person_of[k]}"

    records: list[BenchRecord] = []
    named: set[AccountKey] = set()
    for record in _lines(export / "records.jsonl"):
        container: dict[str, Any] = record.get("container") or {}
        if kept is not None and container.get("id") not in kept:
            continue
        named.update(record_accounts(record))
        for k in record_accounts(record):
            if k not in truth.person_of:
                raise ValueError(f"{k} is in a record but not in accounts.jsonl")
        author = record_accounts(record)[0]
        records.append(
            BenchRecord(
                key=f"{source}/{instance}/{record['source_id']}",
                case=CASE,
                unit=truth.unit_of_account[author],
                record=record,
            )
        )
    if kept is not None:
        profiles = {k: p for k, p in profiles.items() if k in named}
        truth = Truth(
            person_of={k: v for k, v in truth.person_of.items() if k in named},
            unit_of_account={
                k: v for k, v in truth.unit_of_account.items() if k in named
            },
            bots=truth.bots & named,
        )
    return Dataset(records=records, profiles=profiles, truth=truth)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Turn a Discord connector export into a benchmark dataset."
    )
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--people", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--only", action="append", help="a category name or channel ID; repeatable"
    )
    args = parser.parse_args(argv)
    dataset = convert(
        args.export,
        json.loads(args.people.read_text()),
        only=set(args.only) if args.only else None,
    )
    dataset.write(args.out)
    print(
        f"wrote {len(dataset.records)} records, {len(dataset.profiles)} accounts"
        + f" ({len(dataset.truth.bots)} bots) to {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
