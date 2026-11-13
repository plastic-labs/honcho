"""Synthetic fixtures: the hard cases an organizer must get right, with answers.

Each case is generated several times (instances) from a seed, with different
names and ids, so the held-out split has every case on both sides. Every
instance is its own unit (see `model.BenchRecord.unit`): its people, channels
and records appear nowhere else.

Nothing an organizer sees gives the answers away. External ids, servers,
channels and records get opaque random ids, independent across sources, so
two accounts of one person look no more alike than two strangers' do. Case
and unit labels never reach an organizer (`Dataset.blind`).

The cases:

- `same_name`: two different people with one display name in one server, and a
  third mention of that name with no id, really the first person.
- `shared_handle`: two different people with one username on two platforms,
  and two explicit peers whose names differ only in case.
- `cross_source`: one person on Discord, Slack and GitHub. Sometimes a username
  or email ties the accounts, sometimes only the content does.
- `outside_users`: a channel shared with a customer's users, one of whom has a
  team member's first name.
- `account_change`: a person moves to a new account.
- `edits_deletes`: records revised or deleted after they were sent.
- `explicit_mixed`: an explicit `peer_id` and an account of the same person.
- `no_container`: meeting notes with no channel and no single author.
- `multi_customer_channel`: one channel that discusses several customers, each
  author several of them.
- `plugin_turns`: coding-agent turns with injected system reminders; the agent
  is not a person.
- `bots`: a flagged bot and an unflagged webhook integration in a channel.
"""

import datetime
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tests.bench.auto_workspaces.model import (
    AccountKey,
    BenchRecord,
    Dataset,
    Op,
    Profile,
    account_key,
    explicit_key,
)

_FIRST = [
    "Alex", "Sam", "Jordan", "Riley", "Casey", "Morgan", "Taylor", "Jamie",
    "Avery", "Quinn", "Rowan", "Skyler", "Emerson", "Hayden", "Reese", "Parker",
]  # fmt: skip
_LAST = [
    "Nguyen", "Okafor", "Lindqvist", "Moreau", "Tanaka", "Haddad", "Kowalski",
    "Ferreira", "Iyer", "Sato", "Brennan", "Vasquez", "Adeyemi", "Novak",
]  # fmt: skip
_PROJECTS = ["northstar", "lighthouse", "atlas", "quarry", "meridian", "harbor"]
_CUSTOMERS = ["acme", "globex", "initech", "umbrella", "hooli", "vandelay"]
_LINES = [
    "{p} needs SSO before rollout.",
    "Pushed the {p} migration; please review.",
    "Can we move the {p} demo to Friday?",
    "{p} latency is back under 200ms.",
    "Writing up the {p} runbook today.",
    "Who owns the {p} billing question?",
    "The {p} customer asked for an export.",
    "Blocked on credentials for {p}.",
]
_REMINDER = (
    "<system-reminder>\nThe user's project uses uv. Run tests with `uv run "
    + "pytest`.\n</system-reminder>\n"
)


_EPOCH = datetime.datetime(2026, 9, 1, 9, 0, tzinfo=datetime.UTC)
_WINDOW_MINUTES = 3 * 24 * 60


@dataclass(frozen=True)
class _Account:
    key: AccountKey
    person: str
    author: dict[str, Any] | None
    peer_id: str | None = None


class _Builder:
    """Accumulates one dataset; every helper records its truth as it goes."""

    rng: random.Random
    dataset: Dataset
    _clock: datetime.datetime
    _ids: set[str]
    _handles: set[str]

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.dataset = Dataset()
        self._clock = _EPOCH
        self._ids = set()
        self._handles = set()

    def start_instance(self) -> None:
        """Start an instance at a random time in a two-month window.

        Instances overlap in time, and each record of an instance gets a random
        time in its window, so neither record order nor timestamps reveal which
        records were generated together or in what pattern. Where order is
        real (an account replaced by a new one, an edit after the original),
        the case asks for it with `times` and `after`.
        """
        self._clock = _EPOCH + datetime.timedelta(minutes=self.rng.randint(0, 86_400))

    def _at(self, moment: datetime.datetime) -> str:
        return moment.isoformat().replace("+00:00", "Z")

    def tick(self) -> str:
        """A random time in the current instance's three-day window."""
        offset = datetime.timedelta(minutes=self.rng.randint(0, _WINDOW_MINUTES))
        return self._at(self._clock + offset)

    def times(self, n: int) -> list[str]:
        """`n` random times in the instance's window, in order."""
        return sorted(self.tick() for _ in range(n))

    def after(self, moment: str) -> str:
        """A time up to ten hours after `moment`."""
        start = datetime.datetime.fromisoformat(moment.replace("Z", "+00:00"))
        return self._at(start + datetime.timedelta(minutes=self.rng.randint(1, 600)))

    def id(self) -> str:
        """An opaque id, unique in the dataset, shaped like a Discord snowflake."""
        while True:
            value = str(self.rng.randrange(10**17, 10**18))
            if value not in self._ids:
                self._ids.add(value)
                return value

    def handle(self, first: str, last: str) -> str:
        """A username no other account in the dataset has."""
        while True:
            value = f"{first[0]}{last}".lower() + str(self.rng.randint(1, 9999))
            if value not in self._handles:
                self._handles.add(value)
                return value

    def name(self) -> tuple[str, str]:
        return self.rng.choice(_FIRST), self.rng.choice(_LAST)

    def line(self, topic: str) -> str:
        return self.rng.choice(_LINES).format(p=topic.capitalize())

    def container(self, name: str, parent: str | None = None) -> dict[str, Any]:
        container: dict[str, Any] = {"id": self.id(), "name": name}
        if parent:
            container["parent_id"] = parent
        return container

    def _claim(self, key: AccountKey, unit: str) -> None:
        """Record an account's unit; an account can't belong to two units."""
        owner = self.dataset.truth.unit_of_account.setdefault(key, unit)
        if owner != unit:
            raise ValueError(f"account {key} is in both {owner} and {unit}")

    def account(
        self,
        unit: str,
        person: str,
        source: str,
        instance: str,
        *,
        name: str,
        external_id: str | None = None,
        container: str | None = None,
        username: str | None = None,
        email: str | None = None,
        is_bot: bool = False,
        bot_in_truth: bool | None = None,
    ) -> _Account:
        """An account of `person`. `is_bot` is what the source says; the truth
        follows it unless `bot_in_truth` says otherwise."""
        key = account_key(
            source, instance, external_id=external_id, name=name, container=container
        )
        self._claim(key, unit)
        self.dataset.truth.person_of[key] = person
        if is_bot if bot_in_truth is None else bot_in_truth:
            self.dataset.truth.bots.add(key)
        self.dataset.profiles[key] = Profile(
            account=key,
            display_name=name,
            username=username,
            email=email,
            is_bot=is_bot,
        )
        author: dict[str, Any] = {"name": name}
        if external_id:
            author["id"] = external_id
        return _Account(key=key, person=person, author=author)

    def explicit(self, unit: str, person: str, peer_id: str) -> _Account:
        key = explicit_key(peer_id)
        self._claim(key, unit)
        self.dataset.truth.person_of[key] = person
        self.dataset.profiles[key] = Profile(account=key)
        return _Account(key=key, person=person, author=None, peer_id=peer_id)

    def source_peer(self, unit: str, source: str, instance: str) -> _Account:
        """The author of author-less records: the source itself, not a person."""
        key = account_key(source, instance)
        self._claim(key, unit)
        self.dataset.truth.person_of[key] = f"{unit}/source"
        self.dataset.truth.bots.add(key)
        self.dataset.profiles[key] = Profile(account=key)
        return _Account(key=key, person=f"{unit}/source", author=None)

    def record(
        self,
        case: str,
        unit: str,
        source: str,
        instance: str,
        author: _Account,
        content: str,
        *,
        groups: set[str],
        container: dict[str, Any] | None = None,
        participants: list[_Account] | None = None,
        source_id: str | None = None,
        revised_at: str | None = None,
        created_at: str | None = None,
        op: Op = "create",
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "source": source,
            "source_instance": instance,
            "source_id": source_id or self.id(),
            "created_at": created_at or self.tick(),
            "content": content,
        }
        if revised_at:
            record["revised_at"] = revised_at
        if container is not None:
            record["container"] = container
        if author.peer_id:
            record["peer_id"] = author.peer_id
        elif author.author is not None:
            record["author"] = author.author
        if participants:
            record["participants"] = [p.author for p in participants if p.author]
        key = self.id()
        self.dataset.records.append(
            BenchRecord(key=key, case=case, unit=unit, record=record, op=op)
        )
        self.dataset.truth.groups_of[key] = frozenset(groups)
        return record


def same_name(b: _Builder, i: int) -> None:
    case, unit = "same_name", f"same_name#{i}"
    guild, notes_instance = b.id(), b.id()
    first = b.rng.choice(_FIRST)
    topic_a, topic_b = b.rng.sample(_PROJECTS, 2)
    chan_a, chan_b = b.container(topic_a), b.container(topic_b)
    one = b.account(
        unit, f"{unit}/one", "discord", guild, name=first, external_id=b.id(),
        username=b.handle(first, "a"),
    )  # fmt: skip
    two = b.account(
        unit, f"{unit}/two", "discord", guild, name=first, external_id=b.id(),
        username=b.handle(first, "b"),
    )  # fmt: skip
    for _ in range(4):
        b.record(case, unit, "discord", guild, one, b.line(topic_a),
                 container=chan_a, groups={f"{unit}/{topic_a}"})  # fmt: skip
        b.record(case, unit, "discord", guild, two, b.line(topic_b),
                 container=chan_b, groups={f"{unit}/{topic_b}"})  # fmt: skip
    # Meeting notes name the first one with no id: a provisional person, really
    # `one`, which only the content can tell.
    notes = b.source_peer(unit, "granola", notes_instance)
    mention = b.account(unit, f"{unit}/one", "granola", notes_instance, name=first)
    b.record(
        case, unit, "granola", notes_instance, notes,
        f"Sync notes: {first} will finish the {topic_a} work.",
        participants=[mention], groups={f"{unit}/{topic_a}"},
    )  # fmt: skip


def shared_handle(b: _Builder, i: int) -> None:
    case, unit = "shared_handle", f"shared_handle#{i}"
    guild, org, app = b.id(), b.id(), b.id()
    first, last = b.name()
    handle = b.handle(first, last)
    topic_a, topic_b = b.rng.sample(_PROJECTS, 2)
    # One username, two people: it's free on each platform separately.
    on_discord = b.account(
        unit, f"{unit}/discord-person", "discord", guild, name=first,
        external_id=b.id(), username=handle,
    )  # fmt: skip
    on_github = b.account(
        unit, f"{unit}/github-person", "github", org, name=handle,
        external_id=b.id(), username=handle,
    )  # fmt: skip
    chan = b.container(topic_a)
    repo = b.id()
    for _ in range(3):
        b.record(case, unit, "discord", guild, on_discord, b.line(topic_a),
                 container=chan, groups={f"{unit}/{topic_a}"})  # fmt: skip
        b.record(case, unit, "github", org, on_github, f"Review: {b.line(topic_b)}",
                 container=b.container(f"{topic_b} PR", parent=repo),
                 groups={f"{unit}/{topic_b}"})  # fmt: skip
    # Peer names are case-sensitive: these are two different people.
    peer_id = f"{first}-{last}-{b.id()[-6:]}"
    upper = b.explicit(unit, f"{unit}/upper", peer_id)
    lower = b.explicit(unit, f"{unit}/lower", peer_id.lower())
    session = b.container(topic_a)
    for author in (upper, lower, upper):
        b.record(case, unit, "app", app, author, b.line(topic_a),
                 container=session, groups={f"{unit}/{topic_a}"})  # fmt: skip


def cross_source(b: _Builder, i: int) -> None:
    case, unit = "cross_source", f"cross_source#{i}"
    guild, team, org = b.id(), b.id(), b.id()
    first, last = b.name()
    handle = b.handle(first, last)
    email = f"{handle}@example.com"
    topic = b.rng.choice(_PROJECTS)
    group = f"{unit}/{topic}"
    person = f"{unit}/person"
    # 0: a username and an email tie the accounts; 1: only the email ties Slack
    # and GitHub; 2: nothing but the content does.
    hints = i % 3
    discord = b.account(
        unit, person, "discord", guild, name=first, external_id=b.id(),
        username=handle if hints == 0 else b.handle(first, last),
    )  # fmt: skip
    slack = b.account(
        unit, person, "slack", team, name=f"{first} {last[0]}.", external_id=b.id(),
        email=email if hints < 2 else None,
    )  # fmt: skip
    github = b.account(
        unit, person, "github", org, name=handle, external_id=b.id(),
        username=handle, email=email if hints < 2 else None,
    )  # fmt: skip
    other_first, other_last = b.name()
    other = b.account(
        unit, f"{unit}/other", "discord", guild, name=other_first,
        external_id=b.id(), username=b.handle(other_first, other_last),
    )  # fmt: skip
    chan, slack_chan = b.container(topic), b.container(f"{topic}-eng")
    repo = b.id()
    for _ in range(3):
        b.record(case, unit, "discord", guild, discord, b.line(topic),
                 container=chan, groups={group})  # fmt: skip
        b.record(case, unit, "discord", guild, other, b.line(topic),
                 container=chan, groups={group})  # fmt: skip
        b.record(case, unit, "slack", team, slack, b.line(topic),
                 container=slack_chan, groups={group})  # fmt: skip
        b.record(case, unit, "github", org, github, f"Review: {b.line(topic)}",
                 container=b.container(f"{topic} PR", parent=repo),
                 groups={group})  # fmt: skip
    b.record(
        case, unit, "discord", guild, discord,
        f"Opened the {topic} PR on GitHub as {handle}.",
        container=chan, groups={group},
    )  # fmt: skip


def outside_users(b: _Builder, i: int) -> None:
    case, unit = "outside_users", f"outside_users#{i}"
    team = b.id()
    customer = b.rng.choice(_CUSTOMERS)
    shared_first = b.rng.choice(_FIRST)

    def member(role: str, name: str, domain: str) -> _Account:
        return b.account(
            unit, f"{unit}/{role}", "slack", team, name=name, external_id=b.id(),
            email=f"{role}{b.id()[-6:]}@{domain}.example",
        )  # fmt: skip

    teammate = member("teammate", f"{shared_first} {b.rng.choice(_LAST)}", "team")
    colleague = member("colleague", " ".join(b.name()), "team")
    guest = member("guest", f"{shared_first} {b.rng.choice(_LAST)}", customer)
    guest2 = member("guest2", " ".join(b.name()), customer)
    shared, internal = b.container(f"ext-{customer}"), b.container("eng")
    topic = b.rng.choice(_PROJECTS)
    for author in (teammate, guest, colleague, guest2, guest, teammate):
        b.record(case, unit, "slack", team, author, f"[{customer}] {b.line(topic)}",
                 container=shared, groups={f"{unit}/{customer}"})  # fmt: skip
    for author in (teammate, colleague):
        b.record(case, unit, "slack", team, author, b.line(topic),
                 container=internal, groups={f"{unit}/{topic}"})  # fmt: skip


def account_change(b: _Builder, i: int) -> None:
    case, unit = "account_change", f"account_change#{i}"
    guild = b.id()
    first, last = b.name()
    person = f"{unit}/person"
    # Half the time both accounts carry the person's email; otherwise only
    # their own message says the new account is them.
    email = f"{b.handle(first, last)}@example.com" if i % 2 == 0 else None
    old = b.account(
        unit, person, "discord", guild, name=first, external_id=b.id(),
        username=b.handle(first, last), email=email,
    )  # fmt: skip
    new = b.account(
        unit, person, "discord", guild, name=f"{first} {last}", external_id=b.id(),
        username=b.handle(first, last), email=email,
    )  # fmt: skip
    topic = b.rng.choice(_PROJECTS)
    chan = b.container(topic)
    # The old account's messages, then the switch, then the new account's.
    when = iter(b.times(7))
    for _ in range(3):
        b.record(case, unit, "discord", guild, old, b.line(topic), container=chan,
                 groups={f"{unit}/{topic}"}, created_at=next(when))  # fmt: skip
    b.record(case, unit, "discord", guild, new,
             f"New account, my old one ({first}) got locked out.", container=chan,
             groups={f"{unit}/{topic}"}, created_at=next(when))  # fmt: skip
    for _ in range(3):
        b.record(case, unit, "discord", guild, new, b.line(topic), container=chan,
                 groups={f"{unit}/{topic}"}, created_at=next(when))  # fmt: skip


def edits_deletes(b: _Builder, i: int) -> None:
    case, unit = "edits_deletes", f"edits_deletes#{i}"
    team = b.id()
    first, _last = b.name()
    author = b.account(
        unit, f"{unit}/person", "slack", team, name=first, external_id=b.id()
    )
    topic = b.rng.choice(_PROJECTS)
    chan = b.container(topic)
    group = {f"{unit}/{topic}"}
    originals = [
        b.record(
            case,
            unit,
            "slack",
            team,
            author,
            b.line(topic),
            container=chan,
            groups=group,
        )  # fmt: skip
        for _ in range(4)
    ]
    edited, deleted = originals[1], originals[2]
    b.record(
        case, unit, "slack", team, author,
        edited["content"] + " (edited: moved to Monday)", container=chan,
        groups=group, source_id=edited["source_id"],
        created_at=edited["created_at"], revised_at=b.after(edited["created_at"]),
        op="edit",
    )  # fmt: skip
    b.record(
        case, unit, "slack", team, author, deleted["content"], container=chan,
        groups=group, source_id=deleted["source_id"],
        created_at=deleted["created_at"], op="delete",
    )  # fmt: skip


def explicit_mixed(b: _Builder, i: int) -> None:
    case, unit = "explicit_mixed", f"explicit_mixed#{i}"
    guild, app = b.id(), b.id()
    first, last = b.name()
    peer_id = f"{first.lower()}-{last.lower()}-{b.id()[-6:]}"
    person = f"{unit}/person"
    explicit = b.explicit(unit, person, peer_id)
    # Half the time the account's username is the peer name.
    account = b.account(
        unit, person, "discord", guild, name=first, external_id=b.id(),
        username=peer_id if i % 2 == 0 else b.handle(first, last),
    )  # fmt: skip
    topic = b.rng.choice(_PROJECTS)
    session, chan = b.container(topic), b.container(topic)
    for _ in range(3):
        b.record(case, unit, "app", app, explicit, b.line(topic),
                 container=session, groups={f"{unit}/{topic}"})  # fmt: skip
        b.record(case, unit, "discord", guild, account, b.line(topic),
                 container=chan, groups={f"{unit}/{topic}"})  # fmt: skip


def no_container(b: _Builder, i: int) -> None:
    case, unit = "no_container", f"no_container#{i}"
    team, notes_instance = b.id(), b.id()
    topic = b.rng.choice(_PROJECTS)
    people: list[tuple[_Account, _Account]] = []
    for n in range(3):
        first, last = b.name()
        email = f"{first}.{last}{b.id()[-6:]}@team.example".lower()
        person = f"{unit}/p{n}"
        slack = b.account(
            unit, person, "slack", team, name=f"{first} {last}", external_id=b.id(),
            email=email,
        )  # fmt: skip
        # Meeting tools identify participants by email.
        attendee = b.account(
            unit, person, "granola", notes_instance, name=f"{first} {last}",
            external_id=email, email=email,
        )  # fmt: skip
        people.append((slack, attendee))
    notes = b.source_peer(unit, "granola", notes_instance)
    for _ in range(2):
        b.record(
            case, unit, "granola", notes_instance, notes,
            f"Meeting notes. Decisions: {b.line(topic)}",
            participants=[attendee for _, attendee in people],
            groups={f"{unit}/{topic}"},
        )  # fmt: skip
    chan = b.container(topic)
    for slack, _ in people:
        b.record(case, unit, "slack", team, slack, b.line(topic),
                 container=chan, groups={f"{unit}/{topic}"})  # fmt: skip


def multi_customer_channel(b: _Builder, i: int) -> None:
    case, unit = "multi_customer_channel", f"multi_customer_channel#{i}"
    team = b.id()
    customers = b.rng.sample(_CUSTOMERS, 3)
    authors = [
        b.account(
            unit,
            f"{unit}/p{n}",
            "slack",
            team,
            name=" ".join(b.name()),
            external_id=b.id(),
        )  # fmt: skip
        for n in range(3)
    ]
    chan = b.container("customer-success")
    # Authors are drawn independently of customers: who wrote a record says
    # nothing about which customer it is about.
    for n in range(12):
        customer = customers[n % 3]
        b.record(
            case, unit, "slack", team, b.rng.choice(authors),
            f"{customer.capitalize()} renewal: {b.line(customer)}",
            container=chan, groups={f"{unit}/{customer}"},
        )  # fmt: skip
    both = customers[:2]
    b.record(
        case, unit, "slack", team, b.rng.choice(authors),
        f"{both[0].capitalize()} and {both[1].capitalize()} both asked for SSO.",
        container=chan, groups={f"{unit}/{c}" for c in both},
    )  # fmt: skip


def plugin_turns(b: _Builder, i: int) -> None:
    case, unit = "plugin_turns", f"plugin_turns#{i}"
    machine, org = b.id(), b.id()
    first, last = b.name()
    handle = b.handle(first, last)
    email = f"{handle}@example.com"
    person = f"{unit}/person"
    user = b.account(
        unit, person, "claude-code", machine, name=handle, external_id=b.id(),
        email=email,
    )  # fmt: skip
    # The coding agent is not a person. The plugin knows which turns are its own.
    agent = b.account(
        unit, f"{unit}/agent", "claude-code", machine, name="assistant",
        external_id=b.id(), is_bot=True,
    )  # fmt: skip
    github = b.account(
        unit, person, "github", org, name=handle, external_id=b.id(),
        username=handle, email=email,
    )  # fmt: skip
    for repo in b.rng.sample(_PROJECTS, 2):
        conversation = b.container(repo)
        repo_id = b.id()
        for turn in range(3):
            reminder = _REMINDER if turn % 2 == 0 else ""
            b.record(case, unit, "claude-code", machine, user,
                     f"{reminder}Fix the failing {repo} test.",
                     container=conversation, groups={f"{unit}/{repo}"})  # fmt: skip
            b.record(case, unit, "claude-code", machine, agent,
                     f"Updated the {repo} fixture; the test passes now.",
                     container=conversation, groups={f"{unit}/{repo}"})  # fmt: skip
        b.record(
            case, unit, "github", org, github, f"Merged the {repo} test fix.",
            container=b.container(f"{repo} PR", parent=repo_id),
            groups={f"{unit}/{repo}"},
        )  # fmt: skip


def bots(b: _Builder, i: int) -> None:
    case, unit = "bots", f"bots#{i}"
    guild = b.id()
    flagged = b.account(
        unit, f"{unit}/alerts", "discord", guild, name="Alerts", external_id=b.id(),
        is_bot=True,
    )  # fmt: skip
    # A webhook integration: a bot, but the source doesn't flag it.
    webhook = b.account(
        unit, f"{unit}/tracker", "discord", guild, name="Tracker", external_id=b.id(),
        bot_in_truth=True,
    )  # fmt: skip
    human = b.account(
        unit, f"{unit}/human", "discord", guild, name=" ".join(b.name()),
        external_id=b.id(),
    )  # fmt: skip
    topic = b.rng.choice(_PROJECTS)
    chan = b.container(f"{topic}-alerts")
    for _ in range(2):
        b.record(case, unit, "discord", guild, flagged,
                 f"[ALERT] {topic} error rate above 2%", container=chan,
                 groups={f"{unit}/{topic}"})  # fmt: skip
        b.record(case, unit, "discord", guild, webhook,
                 f"Issue updated: {b.line(topic)}", container=chan,
                 groups={f"{unit}/{topic}"})  # fmt: skip
        b.record(case, unit, "discord", guild, human, b.line(topic),
                 container=chan, groups={f"{unit}/{topic}"})  # fmt: skip


CASES: dict[str, Callable[[_Builder, int], None]] = {
    "same_name": same_name,
    "shared_handle": shared_handle,
    "cross_source": cross_source,
    "outside_users": outside_users,
    "account_change": account_change,
    "edits_deletes": edits_deletes,
    "explicit_mixed": explicit_mixed,
    "no_container": no_container,
    "multi_customer_channel": multi_customer_channel,
    "plugin_turns": plugin_turns,
    "bots": bots,
}


def generate(seed: int = 7, instances: int = 30) -> Dataset:
    """Every case, `instances` times each, deterministically from `seed`."""
    builder = _Builder(seed)
    for i in range(instances):
        for make in CASES.values():
            builder.start_instance()
            make(builder, i)
    return builder.dataset
