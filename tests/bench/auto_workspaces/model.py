"""The auto-workspaces benchmark's data: records to ingest, what is known about
their accounts, and the ground truth an organizer is scored against.

A dataset is what a connector would send (`BenchRecord.record`, an ingest API
record exactly as `POST /ingest` takes it), plus what it could know about each
account (`Profile`), plus the answers (`Truth`), which no organizer may read.
"""

import datetime
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from src.utils.sanitization import strip_nul

# An account's stable key, mirroring how ingest binds people (`peer_key_for` and
# `author_key_for` in src/crud/ingest.py), with the same unambiguous encoding:
#   "{source}/{source_instance}/id:{external id}"
#       an author or participant with an id
#   "{source}/{source_instance}/name:container:{len}:{container id}:{casefolded name}"
#   "{source}/{source_instance}/name:standalone:{casefolded name}"
#       no id: a provisional person, never merged across containers
#   "{source}/{source_instance}/source"
#       the author of author-less records
#   "explicit:{peer_id}"
#       an explicit `peer_id`, case and all
AccountKey = str
# A record's stable key inside a dataset.
RecordKey = str

Op = Literal["create", "edit", "delete"]


def account_key(
    source: str,
    source_instance: str,
    *,
    external_id: str | None = None,
    name: str | None = None,
    container: str | None = None,
) -> AccountKey:
    """The key ingest would bind this account under."""
    if external_id:
        return f"{source}/{source_instance}/id:{external_id}"
    if name:
        # ingest's provisional_name key value: the container id is
        # length-prefixed, so no (container, name) pair spells another's key.
        # Display names are NUL-stripped before ingest keys them.
        scope = f"container:{len(container)}:{container}" if container else "standalone"
        return f"{source}/{source_instance}/name:{scope}:{strip_nul(name).casefold()}"
    return f"{source}/{source_instance}/source"


def explicit_key(peer_id: str) -> AccountKey:
    """The key of an explicit `peer_id`."""
    return f"explicit:{peer_id}"


def record_accounts(record: dict[str, Any]) -> list[AccountKey]:
    """Every account an ingest record names: its author first, then participants."""
    source = record["source"]
    instance = record["source_instance"]
    container_fields: dict[str, Any] = record.get("container") or {}
    container: str | None = container_fields.get("id")
    keys: list[AccountKey] = []
    if record.get("peer_id"):
        keys.append(explicit_key(record["peer_id"]))
    elif record.get("author"):
        author = record["author"]
        keys.append(
            account_key(
                source,
                instance,
                external_id=author.get("id"),
                name=author["name"],
                container=container,
            )
        )
    else:
        keys.append(account_key(source, instance))
    participants: list[dict[str, Any]] = record.get("participants") or []
    for participant in participants:
        keys.append(
            account_key(
                source,
                instance,
                external_id=participant.get("id"),
                name=participant["name"],
                container=container,
            )
        )
    return list(dict.fromkeys(keys))


@dataclass(frozen=True)
class Profile:
    """What a connector could know about an account, besides its records.

    Every field is optional: sources differ in what they expose.
    """

    account: AccountKey
    display_name: str | None = None
    username: str | None = None
    email: str | None = None
    # The source says the account is a bot (Discord's bot flag, a GitHub App).
    # Integrations that post through webhooks often aren't flagged.
    is_bot: bool = False


@dataclass(frozen=True)
class BenchRecord:
    """One record to send, the case it belongs to, and its operation.

    `unit` is what the held-out split assigns: every record and account of one
    unit lands on the same side, so an organizer tuned on one side never sees
    the people or channels of the other.
    """

    key: RecordKey
    case: str
    unit: str
    record: dict[str, Any]
    op: Op = "create"


@dataclass
class Truth:
    """The answers. An organizer never reads these."""

    # Account -> the person it belongs to. Every account in the dataset is here.
    person_of: dict[AccountKey, str] = field(default_factory=dict)
    # Account -> the unit its person belongs to, for the split.
    unit_of_account: dict[AccountKey, str] = field(default_factory=dict)
    # Record -> the groupings (scopes) it belongs in. Most records have one.
    groups_of: dict[RecordKey, frozenset[str]] = field(default_factory=dict)
    # Accounts that are bots or integrations, not people.
    bots: set[AccountKey] = field(default_factory=set)


_OP_RANK: dict[str, int] = {"create": 0, "edit": 1, "delete": 2}


def event_time(r: BenchRecord) -> datetime.datetime:
    """When the event happened, in UTC: an edit at its revision time."""
    when: str = (r.record.get("revised_at") if r.op == "edit" else None) or r.record[
        "created_at"
    ]
    moment = datetime.datetime.fromisoformat(when.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=datetime.UTC)
    return moment.astimezone(datetime.UTC)


def _feed_order(r: BenchRecord) -> tuple[datetime.datetime, int, str]:
    """When a connector would send this event; at equal times, the original first."""
    return (event_time(r), _OP_RANK[r.op], r.key)


@dataclass
class Dataset:
    """Records, account profiles and ground truth."""

    records: list[BenchRecord] = field(default_factory=list)
    profiles: dict[AccountKey, Profile] = field(default_factory=dict)
    truth: Truth = field(default_factory=Truth)

    def accounts(self) -> list[AccountKey]:
        """Every account the records name, in first-seen order."""
        seen: dict[AccountKey, None] = {}
        for r in self.records:
            for key in record_accounts(r.record):
                seen.setdefault(key, None)
        return list(seen)

    def blind(self) -> "Dataset":
        """What an organizer may see: records and profiles, no labels or truth.

        Case and unit labels are evaluation bookkeeping; an organizer that read
        them would be told which accounts belong together. Records come in the
        order a connector sends them (by when each event happened: an edit at
        its revision time, after its original; a delete after its original),
        and profiles by key, never in the order they were generated.
        """
        ordered = sorted(self.records, key=_feed_order)
        return Dataset(
            records=[replace(r, case="", unit="") for r in ordered],
            # Sorted by key, not kept in the order the accounts were made.
            profiles={k: self.profiles[k] for k in sorted(self.profiles)},
        )

    def subset(self, units: Iterable[str]) -> "Dataset":
        """The records, profiles and truth of the given units only."""
        keep = set(units)
        records = [r for r in self.records if r.unit in keep]
        accounts = {
            key for key, unit in self.truth.unit_of_account.items() if unit in keep
        }
        record_keys = {r.key for r in records}
        return Dataset(
            records=records,
            profiles={k: p for k, p in self.profiles.items() if k in accounts},
            truth=Truth(
                person_of={
                    k: v for k, v in self.truth.person_of.items() if k in accounts
                },
                unit_of_account={
                    k: v for k, v in self.truth.unit_of_account.items() if k in accounts
                },
                groups_of={
                    k: v for k, v in self.truth.groups_of.items() if k in record_keys
                },
                bots={k for k in self.truth.bots if k in accounts},
            ),
        )

    def write(self, directory: Path) -> None:
        """Write `records.jsonl` (what to send), `profiles.jsonl` and `truth.json`."""
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "records.jsonl").open("w") as out:
            for r in self.records:
                out.write(json.dumps(asdict(r)) + "\n")
        with (directory / "profiles.jsonl").open("w") as out:
            for p in self.profiles.values():
                out.write(json.dumps(asdict(p)) + "\n")
        truth = {
            "person_of": self.truth.person_of,
            "unit_of_account": self.truth.unit_of_account,
            "groups_of": {k: sorted(v) for k, v in self.truth.groups_of.items()},
            "bots": sorted(self.truth.bots),
        }
        (directory / "truth.json").write_text(json.dumps(truth, indent=1))

    @classmethod
    def read(cls, directory: Path) -> "Dataset":
        """Read a dataset `write` produced."""
        records = [
            BenchRecord(**json.loads(line))
            for line in (directory / "records.jsonl").read_text().splitlines()
            if line
        ]
        profiles = [
            Profile(**json.loads(line))
            for line in (directory / "profiles.jsonl").read_text().splitlines()
            if line
        ]
        raw = json.loads((directory / "truth.json").read_text())
        return cls(
            records=records,
            profiles={p.account: p for p in profiles},
            truth=Truth(
                person_of=raw["person_of"],
                unit_of_account=raw["unit_of_account"],
                groups_of={k: frozenset(v) for k, v in raw["groups_of"].items()},
                bots=set(raw["bots"]),
            ),
        )


@dataclass
class Prediction:
    """What an organizer decided, in the same terms as `Truth`."""

    # Account -> predicted person. An account left out is its own person.
    person_of: dict[AccountKey, str] = field(default_factory=dict)
    # Record -> predicted groupings. A record left out is in no grouping.
    groups_of: dict[RecordKey, frozenset[str]] = field(default_factory=dict)
    bots: set[AccountKey] = field(default_factory=set)
    # Organizer LLM tokens spent (input + output), for cost per ingested token.
    organizer_tokens: int = 0
    # Suggestions a person would be asked to review.
    suggestions: int = 0
