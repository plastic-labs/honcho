"""The identity rules (`src.organizer.identity`) as a benchmark organizer.

It reads what an organizer may read: records and account profiles, never the
truth. People are linked as if a person accepted every `high` suggestion and
nothing else, which is what "accept all high" does in one step. The review
load counts the `low` suggestions, the ones a person looks at one by one;
`high_suggestions` reports the others. Bots and groups are the baseline's:
these rules only suggest who is who.
"""

import datetime
import hashlib
from typing import Any

from src.organizer.identity import Account, AccountKind, Post, suggest
from tests.bench.auto_workspaces import baseline
from tests.bench.auto_workspaces.model import (
    AccountKey,
    Dataset,
    Prediction,
    record_accounts,
)


def _kind(key: AccountKey) -> AccountKind:
    if key.startswith("explicit:"):
        return "explicit"
    if key.endswith("/source"):
        return "source"
    if "/name:" in key:
        return "provisional"
    return "external"


def _source(key: AccountKey) -> tuple[str, str]:
    if key.startswith("explicit:"):
        return ("explicit", "")
    source, instance, _ = key.split("/", 2)
    return source, instance


def accounts_of(dataset: Dataset) -> list[Account]:
    """Every account the records name, with what its profile says."""
    accounts: list[Account] = []
    for key in dataset.accounts():
        profile = dataset.profiles.get(key)
        kind = _kind(key)
        source, instance = _source(key)
        names = (profile.display_name,) if profile and profile.display_name else ()
        handle = key.removeprefix("explicit:") if kind == "explicit" else None
        if profile and profile.username:
            handle = profile.username
        email = profile.email if profile else None
        accounts.append(
            Account(
                key=key,
                source=source,
                instance=instance,
                kind=kind,
                names=names,
                handle=handle,
                email_digest=hashlib.sha256(
                    email.strip().casefold().encode()
                ).hexdigest()
                if email
                else None,
                bot=bool(profile and profile.is_bot),
                # Before any link, each account is its own person.
                peer=key,
            )
        )
    return accounts


def posts_of(dataset: Dataset) -> list[Post]:
    """One post per created record: an edit or a delete is not new writing."""
    posts: list[Post] = []
    for r in dataset.records:
        if r.op != "create":
            continue
        keys = record_accounts(r.record)
        metadata: dict[str, Any] = r.record.get("metadata") or {}
        url = metadata.get("url")
        posts.append(
            Post(
                author=keys[0],
                at=datetime.datetime.fromisoformat(
                    r.record["created_at"].replace("Z", "+00:00")
                ),
                text=r.record.get("content") or "",
                url=url if isinstance(url, str) else None,
                participants=tuple(keys[1:]),
            )
        )
    return posts


def organize(dataset: Dataset) -> Prediction:
    """Accept every `high` suggestion; keep the baseline's bots and groups."""
    accounts = accounts_of(dataset)
    found = suggest(accounts, posts_of(dataset))
    groups = baseline._UnionFind()  # pyright: ignore[reportPrivateUsage]
    for account in accounts:
        groups.find(account.key)
    for s in found:
        if s.confidence == "high":
            groups.union(*s.accounts)
    rest = baseline.organize(dataset)
    return Prediction(
        person_of={a.key: groups.find(a.key) for a in accounts},
        groups_of=rest.groups_of,
        bots=rest.bots,
        organizer_tokens=0,
        suggestions=sum(1 for s in found if s.confidence == "low"),
    )


def high_suggestions(dataset: Dataset) -> int:
    """How many `high` suggestions the rules make: accepted together, not reviewed."""
    return sum(
        1
        for s in suggest(accounts_of(dataset), posts_of(dataset))
        if s.confidence == "high"
    )
