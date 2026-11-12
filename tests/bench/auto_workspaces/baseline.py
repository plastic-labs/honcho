"""The rules-only baseline every organizer must beat.

It reads what an organizer may read (records and account profiles, never the
truth) and decides with exact rules only:

- **People:** accounts are linked when they share a handle: the same email,
  the same username across sources, or an explicit `peer_id` equal to an
  account's username. Display names are never matched and content is never
  read. A username is unique only within its platform, so two people who
  happen to share one across platforms are merged: the `shared_handle` case.
- **Bots:** accounts the source flags as bots, and the source peer that authors
  records with no author.
- **Groups:** one group per container (a channel, a thread's channel, a
  repository), and each record without a container on its own.
"""

from tests.bench.auto_workspaces.model import (
    AccountKey,
    Dataset,
    Prediction,
    record_accounts,
)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[max(rx, ry)] = min(rx, ry)


def _handles(dataset: Dataset, account: AccountKey) -> list[str]:
    """The exact handles an account can be linked on."""
    profile = dataset.profiles.get(account)
    handles: list[str] = []
    if account.startswith("explicit:"):
        # Peer names are case-sensitive: "Alice" and "alice" are two peers, and
        # an explicit peer matches a username exactly.
        handles.append("peer:" + account.removeprefix("explicit:"))
    if profile is None:
        return handles
    if profile.email:
        handles.append("email:" + profile.email.casefold())
    if profile.username:
        # Usernames match each other across sources whatever their case, and
        # match an explicit peer only exactly.
        handles.append("user:" + profile.username.casefold())
        handles.append("peer:" + profile.username)
    return handles


def organize(dataset: Dataset) -> Prediction:
    """The baseline's prediction for a dataset."""
    accounts = dataset.accounts()
    links = _UnionFind()
    by_handle: dict[str, AccountKey] = {}
    for account in accounts:
        links.find(account)
        for handle in _handles(dataset, account):
            if handle in by_handle:
                links.union(account, by_handle[handle])
            else:
                by_handle[handle] = account

    groups: dict[str, frozenset[str]] = {}
    for r in dataset.records:
        record = r.record
        container = record.get("container")
        if container:
            # A thread or a pull request groups with its channel or repository.
            anchor = container.get("parent_id") or container["id"]
            group = f"{record['source']}/{record['source_instance']}/{anchor}"
        else:
            group = f"standalone/{r.key}"
        groups[r.key] = frozenset({group})

    # The source peer: the author ingest assigns to records with no author.
    source_peers = {
        record_accounts(r.record)[0]
        for r in dataset.records
        if not r.record.get("author") and not r.record.get("peer_id")
    }
    bots = source_peers | {
        account
        for account in accounts
        if (profile := dataset.profiles.get(account)) is not None and profile.is_bot
    }
    # Exact rules need no review, so the baseline suggests nothing, and it
    # calls no model.
    return Prediction(
        person_of={account: links.find(account) for account in accounts},
        groups_of=groups,
        bots=bots,
        organizer_tokens=0,
        suggestions=0,
    )
