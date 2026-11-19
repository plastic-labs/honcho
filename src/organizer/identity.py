"""Which accounts are one person: rules over names, emails and what people write.

Pure: no database and no model calls, so the server (`crud.ingest_suggestions`)
and the benchmark (`tests/bench/auto_workspaces/rules_identity.py`) run the
same rules on the same inputs.

A pair of accounts gets at most one level of evidence from each family:

- **email:** the same email digest (strong).
- **names:** two different names agree, one of them specific (strong); one
  specific name agrees, or a handle matches an initial and surname, or two
  handles share a stem (medium); only a short or common name, a first or last
  name, or the start of a name agrees (weak). A name, stem or initial and
  surname is specific when it has at least 6 characters and no third account
  holds it: full names repeat as much as first names do. An explicit peer ID equal to an account's username,
  case and all, is strong.
- **shared work:** one account links, in another source, to records the other
  created there, within 6 hours of their creation: two or more distinct
  records (strong), or one, or links later on (weak).
- **self-reference:** one account gives the other's handle as its own, in a
  form that names the writer, the other's source and the handle together:
  "I'm on GitHub as alexdev", "my GitHub username is alexdev", "I'm alexdev
  on GitHub", "alexdev on GitHub is me" (strong). Negated sentences, quoted
  lines, code and injected system reminders don't count.

A `high` suggestion has evidence a person can accept at a glance: an email, a
self-reference, two agreeing names, a specific name plus shared work, or a
rare first name plus strong shared work. A `low` one needs a closer look; a
later judge stage reviews those. Nothing is applied here.

What keeps two people apart:

- A handle alone is never enough: a username is unique only on its own
  platform, so two people can hold the same one on two platforms.
- Two accounts in one source instance are usually two people. They are
  suggested only on an email or a self-reference.
- Shared work alone is never enough: someone who posts a colleague's pull
  requests links to work that isn't theirs.
- No transitive merging: accepting every `high` suggestion, together with the
  links that already exist, must not put two accounts of one source instance
  on one person without direct evidence between them, nor two accounts that
  talk to each other. Suggestions that would are `low`.
"""

import datetime
import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

RULES_VERSION = 1

Confidence = Literal["high", "low"]
Signal = Literal["email", "names", "shared_work", "self_reference", "conflict"]
Level = Literal["strong", "medium", "weak"]
AccountKind = Literal["external", "provisional", "explicit", "source"]

FRESH = datetime.timedelta(hours=6)
"""How soon after a record was created a link to it counts as sharing your own work."""

MAX_BLOCK = 50
"""A name held by more accounts than this tells nobody apart; it proposes no pairs."""

_WEIGHT: dict[tuple[Signal, Level], float] = {
    ("email", "strong"): 4.0,
    ("self_reference", "strong"): 3.0,
    ("names", "strong"): 3.0,
    ("names", "medium"): 2.0,
    ("names", "weak"): 0.5,
    ("shared_work", "strong"): 2.0,
    ("shared_work", "weak"): 1.0,
}


@dataclass(frozen=True)
class Account:
    """One account in one source, and what is known about it."""

    key: str
    source: str
    instance: str
    kind: AccountKind = "external"
    names: tuple[str, ...] = ()
    """Display name first, then the other names it goes by."""
    handle: str | None = None
    """Its username in the source; for an explicit peer, the peer ID."""
    email_digest: str | None = None
    bot: bool = False
    peer: str | None = None
    """The peer it maps to now: accounts on one peer are already one person."""


@dataclass(frozen=True)
class Post:
    """One record, by its author: what it says and what it points at."""

    author: str
    at: datetime.datetime
    text: str = ""
    url: str | None = None
    """The record's own address in its source, such as a pull request's URL."""
    participants: tuple[str, ...] = ()
    """Accounts it addresses, mentions or replies to."""


@dataclass(frozen=True)
class Evidence:
    signal: Signal
    level: Level
    detail: str
    rare: bool = False
    """A weak name that no third account holds: rare enough to back strong shared work."""


@dataclass(frozen=True)
class Suggestion:
    """Two accounts that look like one person, and why."""

    accounts: tuple[str, str]
    confidence: Confidence
    score: float
    evidence: tuple[Evidence, ...]


def email_digest(tenant_id: str, workspace_name: str, email: str) -> str:
    """What ingest keeps of an email address: a digest scoped to one workspace.

    The address is trimmed and casefolded first, so `Alex@Example.com ` and
    `alex@example.com` match. The tenant and workspace go into the digest, so the
    same address digests differently in two workspaces and digests can't be
    compared across them. `v1` names the scheme, so it can change later.
    """
    scope = f"{len(tenant_id)}:{tenant_id}:{len(workspace_name)}:{workspace_name}"
    value = f"honcho-ingest-email:v1:{scope}:{email.strip().casefold()}"
    # surrogatepass: never raise, so an address can't reach an error report.
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[0-9]+|[^\W\d_]+")
_TRAILING = re.compile(r"[\d_.\-]+$")


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def normalize(name: str) -> str:
    """A name with case, accents, spaces and punctuation removed."""
    return re.sub(r"[\W_]+", "", _fold(name))


def _tokens(name: str) -> list[str]:
    """A name's words, split on spaces, punctuation and case changes."""
    return [_fold(w) for w in _WORD.findall(name)]


def _full_name(name: str) -> bool:
    """At least two words of two or more letters: "David Braun", not "Alex N."."""
    words = [w for w in re.split(r"[\s._-]+", name.strip()) if w]
    return len(words) >= 2 and all(sum(c.isalpha() for c in w) >= 2 for w in words)


def _stem(name: str) -> str:
    """A handle without its trailing digits: "bubjafrig422" → "bubjafrig"."""
    return normalize(_TRAILING.sub("", name.strip()))


def _initial_surname(name: str) -> str | None:
    """ "Vineeth Voruganti" → "vvoruganti"; None unless it is a full name."""
    if not _full_name(name):
        return None
    words = [w for w in _tokens(name) if w.isalpha()]
    return normalize(words[0][0] + words[-1]) if len(words) >= 2 else None


def _all_names(account: Account) -> list[str]:
    names = list(account.names)
    if account.handle:
        names.append(account.handle)
    return [n for n in names if n and n.strip()]


def _name_values(account: Account) -> set[str]:
    """Every normalized form of an account's names: whole, stem, initial and surname."""
    values: set[str] = set()
    for name in _all_names(account):
        values.update({normalize(name), _stem(name), _initial_surname(name) or ""})
    return values - {""}


def _words(account: Account) -> set[str]:
    """The words of an account's names that can stand for a first or last name."""
    return {
        t
        for n in _all_names(account)
        for t in _tokens(n)
        if t.isalpha() and len(t) >= 4
    }


@dataclass(frozen=True)
class _NameIndex:
    """How many accounts hold each name form and word: a pair's own, or common."""

    counts: Counter[str]
    words: Counter[str]

    def distinctive(self, value: str) -> bool:
        """At least 6 characters, not all digits, and held by these two accounts only."""
        return len(value) >= 6 and not value.isdigit() and self.counts[value] <= 2

    def rare(self, *values: str) -> bool:
        """Every value held by at most two accounts, as a name form or a word."""
        return all(self.counts[v] <= 2 and self.words[v] <= 2 for v in values)


def _name_evidence(a: Account, b: Account, index: _NameIndex) -> Evidence | None:
    if (a.kind == "explicit") != (b.kind == "explicit"):
        explicit, other = (a, b) if a.kind == "explicit" else (b, a)
        if other.handle and other.handle == explicit.handle:
            return Evidence(
                "names",
                "strong",
                f"the peer ID {explicit.handle} is their {other.source} username",
            )
    a_names, b_names = _all_names(a), _all_names(b)
    a_norm = {normalize(n) for n in a_names} - {""}
    b_norm = {normalize(n) for n in b_names} - {""}
    same = a_norm & b_norm
    # A full name or a long handle is specific only when no third account has it:
    # common names repeat, and a full name is no exception.
    specific = {s for s in same if index.distinctive(s)}
    shown = ", ".join(sorted(same))
    if len(same) >= 2 and specific:
        return Evidence("names", "strong", f"same names: {shown}")
    if specific:
        return Evidence("names", "medium", f"same name: {shown}")

    a_stems = {_stem(n) for n in a_names} - {""}
    b_stems = {_stem(n) for n in b_names} - {""}
    a_initials = {i for n in a_names if (i := _initial_surname(n))}
    b_initials = {i for n in b_names if (i := _initial_surname(n))}
    for x, y in ((a_initials, b_norm | b_stems), (b_initials, a_norm | a_stems)):
        if hit := sorted(v for v in x & y if index.distinctive(v)):
            return Evidence("names", "medium", f"initial and surname match: {hit[0]}")
    stems = {s for s in a_stems & b_stems if index.distinctive(s)}
    if stems:
        return Evidence("names", "medium", f"same handle stem: {sorted(stems)[0]}")

    if same:
        return Evidence(
            "names", "weak", f"same short name: {shown}", rare=index.rare(*same)
        )
    a_words, b_words = _words(a), _words(b)
    if shared := sorted(a_words & b_words):
        return Evidence(
            "names",
            "weak",
            f"same first or last name: {shared[0]}",
            rare=index.rare(shared[0]),
        )
    for x in sorted(a_words):
        for y in sorted(b_words):
            if x.startswith(y) or y.startswith(x):
                return Evidence(
                    "names",
                    "weak",
                    f"one name starts the other: {x}, {y}",
                    rare=index.rare(x, y),
                )
    return None


def _name_keys(account: Account) -> set[str]:
    """Blocking keys: accounts sharing one are compared."""
    keys: set[str] = set()
    for name in _all_names(account):
        if value := normalize(name):
            keys.add("n:" + value)
        if (stem := _stem(name)) and len(stem) >= 4:
            keys.add("n:" + stem)
        if initials := _initial_surname(name):
            keys.add("n:" + initials)
        keys.update("t:" + t for t in _tokens(name) if t.isalpha() and len(t) >= 4)
    return keys


# ---------------------------------------------------------------------------
# What a person wrote themselves
# ---------------------------------------------------------------------------

_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
_FENCED = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]*`")


def own_words(text: str) -> str:
    """The text without quoted lines, code or injected system reminders."""
    text = _FENCED.sub(" ", _REMINDER.sub(" ", text))
    text = _INLINE_CODE.sub(" ", text)
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith(">")
    )


# ---------------------------------------------------------------------------
# Shared work
# ---------------------------------------------------------------------------

_URL = re.compile(r"https?://[^\s<>()\[\]{}\"'`|]+", re.IGNORECASE)


def normalize_url(url: str) -> str | None:
    """A URL without its query or trailing slash; scheme and host lowercased.

    The fragment stays: on GitHub it names a comment inside a pull request.
    """
    try:
        parts = urlsplit(url.strip().rstrip(".,;:!?)>]"))
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        return None
    path = parts.path.rstrip("/")
    return urlunsplit(("https", parts.netloc.lower(), path, "", parts.fragment))


def _addresses(url: str) -> Iterator[str]:
    """The URL, then up to two path segments shorter.

    A link to a pull request's `/files` page is the pull request's. A link
    with a fragment names one item, such as a comment: it is that item's or
    no one's, never the page's author's.
    """
    yield url
    if "#" in url:
        return
    base = url
    for _ in range(2):
        head = base.rpartition("/")[0]
        if head.count("/") < 3:
            return
        base = head
        yield base


def _shared_work(
    accounts: Mapping[str, Account], posts: Sequence[Post]
) -> dict[tuple[str, str], tuple[int, int]]:
    """(sharer, owner) → (distinct records of the owner linked, of them within FRESH)."""
    owners: dict[str, tuple[datetime.datetime, str]] = {}
    for post in posts:
        url = normalize_url(post.url) if post.url else None
        if url and (url not in owners or post.at < owners[url][0]):
            owners[url] = (post.at, post.author)
    linked: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    for post in posts:
        sharer = accounts.get(post.author)
        if sharer is None or not post.text:
            continue
        for match in _URL.finditer(own_words(post.text)):
            url = normalize_url(match.group(0))
            if url is None:
                continue
            address = next((u for u in _addresses(url) if u in owners), None)
            if address is None:
                continue
            created, owner = owners[address]
            owned = accounts.get(owner)
            if owner == post.author or owned is None or owned.source == sharer.source:
                continue
            fresh = datetime.timedelta(0) <= post.at - created <= FRESH
            seen = linked[(post.author, owner)]
            seen[address] = seen.get(address, False) or fresh
    return {pair: (len(seen), sum(seen.values())) for pair, seen in linked.items()}


# ---------------------------------------------------------------------------
# Self-reference
# ---------------------------------------------------------------------------

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_ACCOUNT_WORDS = r"(?:account|handle|username|user\s*name|login|name|id)"
_NEGATION = re.compile(r"\b(?:not|never|no\s+longer)\b|n't\b", re.IGNORECASE)


def _claims_handle(sentence: str, handle: str, source: str) -> bool:
    """Does the sentence give `handle` on `source` as the writer's own?

    Only in a form that names the writer and the account together, and not
    negated: "My friend Bob is on GitHub as bobdev" and "I'm not on GitHub as
    bobdev" say nothing about the writer.
    """
    if _NEGATION.search(sentence):
        return False
    h = r"@?" + re.escape(handle) + r"(?![\w-])"
    s = re.escape(source)
    patterns = (
        # "I'm on GitHub as alexdev", "I am on GitHub as alexdev"
        rf"\bi(?:'m|\s+am)\s+(?:on|at|in)\s+{s}\s+as\s+{h}",
        # "my GitHub (username) is alexdev", "my GitHub: alexdev"
        rf"\bmy\s+{s}(?:\s+{_ACCOUNT_WORDS})?\s*(?:is|:|=)\s*{h}",
        # "I'm alexdev on GitHub", "I am alexdev on GitHub"
        rf"\bi(?:'m|\s+am)\s+{h}\s+(?:on|at|in)\s+{s}\b",
        # "alexdev on GitHub is me"
        rf"(?<![\w@]){h}\s+(?:on|at|in)\s+{s}\s+is\s+me\b",
    )
    return any(re.search(p, sentence, re.IGNORECASE) for p in patterns)


def _self_references(
    accounts: Mapping[str, Account], posts: Sequence[Post]
) -> dict[tuple[str, str], int]:
    """(writer, account) → sentences where the writer gives that handle as theirs."""
    by_handle: dict[str, list[Account]] = defaultdict(list)
    for account in accounts.values():
        if account.handle and len(account.handle) >= 4 and account.kind == "external":
            by_handle[account.handle.casefold()].append(account)
    if not by_handle:
        return {}
    alternatives = sorted(by_handle, key=len, reverse=True)
    pattern = re.compile(
        r"(?<![\w./-])@?(" + "|".join(map(re.escape, alternatives)) + r")(?![\w-])",
        re.IGNORECASE,
    )
    found: Counter[tuple[str, str]] = Counter()
    for post in posts:
        writer = accounts.get(post.author)
        if writer is None or not post.text or not pattern.search(post.text):
            continue
        for sentence in _SENTENCE.split(own_words(post.text)):
            for match in pattern.finditer(sentence):
                for account in by_handle[match.group(1).casefold()]:
                    if account.key != writer.key and _claims_handle(
                        sentence, account.handle or "", account.source
                    ):
                        found[(writer.key, account.key)] += 1
    return dict(found)


# ---------------------------------------------------------------------------
# Pairs, confidence and consistency
# ---------------------------------------------------------------------------


def _instance(account: Account) -> tuple[str, str]:
    return (account.source, account.instance)


def _confidence(evidence: Sequence[Evidence]) -> Confidence | None:
    level: dict[Signal, Level] = {e.signal: e.level for e in evidence}
    names, work = level.get("names"), level.get("shared_work")
    rare_name = any(e.signal == "names" and e.rare for e in evidence)
    if "email" in level or "self_reference" in level or names == "strong":
        return "high"
    if names == "medium" and work is not None:
        return "high"
    # A rare first name and a habit of linking your own work right away: a
    # colleague who posts your pull requests rarely has your first name too.
    if names == "weak" and rare_name and work == "strong":
        return "high"
    if names == "medium" or work == "strong" or (names == "weak" and work):
        return "low"
    return None


class _Groups:
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


def _without_conflicts(
    suggestions: list[Suggestion],
    accounts: Mapping[str, Account],
    talks: set[tuple[str, str]],
    retained: Iterable[tuple[str, str]] = (),
) -> tuple[list[Suggestion], dict[tuple[str, str], str]]:
    """Make `low` every `high` suggestion of a group accepting them would break.

    The group is what accepting every `high` suggestion would make, the ones
    made now and the `retained` ones still open from earlier, together with
    the accounts that already share a peer. It breaks when it would hold two
    accounts of one source instance with no direct `high` suggestion or shared
    peer between them, or two accounts that talk to each other.

    Returns the suggestions made now, and the retained pairs that must drop to
    `low` too, with why.
    """
    made = {s.accounts for s in suggestions}
    kept = {
        pair
        for pair in retained
        if pair[0] in accounts and pair[1] in accounts and pair not in made
    }
    direct = {s.accounts for s in suggestions if s.confidence == "high"} | kept
    groups = _Groups()
    by_peer: dict[str, list[str]] = defaultdict(list)
    for key, account in accounts.items():
        if account.peer is not None:
            by_peer[account.peer].append(key)
    for keys in by_peer.values():
        for other in keys[1:]:
            groups.union(keys[0], other)
    for a, b in direct:
        groups.union(a, b)

    def joined(x: str, y: str) -> bool:
        pair = (min(x, y), max(x, y))
        px, py = accounts[x].peer, accounts[y].peer
        return pair in direct or (px is not None and px == py)

    members: dict[str, list[str]] = defaultdict(list)
    for key in {k for pair in direct for k in pair}:
        members[groups.find(key)] = []
    for key in accounts:
        root = groups.find(key)
        if root in members:
            members[root].append(key)
    broken: dict[str, str] = {}
    for root, keys in members.items():
        for x, y in combinations(sorted(keys), 2):
            if (x, y) in talks:
                broken[root] = "two of the accounts it would join talk to each other"
                break
            if _instance(accounts[x]) == _instance(accounts[y]) and not joined(x, y):
                broken[root] = (
                    f"it would join two {accounts[x].source} accounts with nothing"
                    + " linking them directly"
                )
                break
    if not broken:
        return suggestions, {}
    demoted = {
        pair: broken[groups.find(pair[0])]
        for pair in kept
        if groups.find(pair[0]) in broken
    }
    return [
        Suggestion(
            s.accounts,
            "low",
            s.score,
            (*s.evidence, Evidence("conflict", "strong", reason)),
        )
        if s.confidence == "high" and (reason := broken.get(groups.find(s.accounts[0])))
        else s
        for s in suggestions
    ], demoted


def suggest(accounts: Iterable[Account], posts: Iterable[Post]) -> list[Suggestion]:
    """Pairs of accounts that look like one person, best first."""
    return suggest_with(accounts, posts, retained=())[0]


def suggest_with(
    accounts: Iterable[Account],
    posts: Iterable[Post],
    retained: Iterable[tuple[str, str]],
) -> tuple[list[Suggestion], dict[tuple[str, str], str]]:
    """`suggest`, checked against the `high` pairs still open from earlier.

    Returns the suggestions, best first, and the retained pairs that would
    now join two people (each with why): they must drop to `low` as well.
    Each retained pair is its two account keys in sorted order.
    """
    by_key = {a.key: a for a in accounts}
    people = {k: a for k, a in by_key.items() if not a.bot and a.kind != "source"}
    post_list = [p for p in posts if p.author in by_key]
    index = _NameIndex(
        counts=Counter(v for a in people.values() for v in _name_values(a)),
        words=Counter(w for a in people.values() for w in _words(a)),
    )

    candidates: set[tuple[str, str]] = set()
    blocks: dict[str, list[str]] = defaultdict(list)
    for key, account in people.items():
        for block in _name_keys(account):
            blocks[block].append(key)
        if account.email_digest:
            blocks["e:" + account.email_digest].append(key)
    for block, keys in blocks.items():
        if len(keys) <= MAX_BLOCK or block.startswith("e:"):
            candidates.update(combinations(sorted(set(keys)), 2))
    work = _shared_work(people, post_list)
    claims = _self_references(people, post_list)
    for x, y in (*work, *claims):
        candidates.add((min(x, y), max(x, y)))

    talks: set[tuple[str, str]] = set()
    for post in post_list:
        for other in post.participants:
            if other != post.author:
                talks.add((min(post.author, other), max(post.author, other)))

    suggestions: list[Suggestion] = []
    for a_key, b_key in sorted(candidates):
        a, b = people.get(a_key), people.get(b_key)
        if a is None or b is None or (a.peer is not None and a.peer == b.peer):
            continue
        evidence: list[Evidence] = []
        if a.email_digest and a.email_digest == b.email_digest:
            evidence.append(Evidence("email", "strong", "same email address"))
        for writer, other in ((a, b), (b, a)):
            if claims.get((writer.key, other.key)):
                evidence.append(
                    Evidence(
                        "self_reference",
                        "strong",
                        f"{writer.source} account {_label(writer)} gives"
                        + f" {other.handle} on {other.source} as its own",
                    )
                )
                break
        if names := _name_evidence(a, b, index):
            evidence.append(names)
        shared = [
            (*work[(sharer.key, owner.key)], sharer, owner)
            for sharer, owner in ((a, b), (b, a))
            if (sharer.key, owner.key) in work
        ]
        if shared:
            n, fresh, sharer, owner = max(shared, key=lambda s: (s[1], s[0]))
            evidence.append(
                Evidence(
                    "shared_work",
                    "strong" if fresh >= 2 else "weak",
                    f"{sharer.source} account {_label(sharer)} linked {n}"
                    + f" {owner.source} records the other created, {fresh} within"
                    + f" {int(FRESH.total_seconds() // 3600)} h of their creation",
                )
            )
        if _instance(a) == _instance(b) and not any(
            e.signal in ("email", "self_reference") for e in evidence
        ):
            continue
        confidence = _confidence(evidence)
        if confidence is None:
            continue
        score = sum(_WEIGHT.get((e.signal, e.level), 0.0) for e in evidence)
        suggestions.append(
            Suggestion((a_key, b_key), confidence, score, tuple(evidence))
        )
    suggestions, demoted = _without_conflicts(suggestions, people, talks, retained)
    return sorted(suggestions, key=lambda s: (-s.score, s.accounts)), demoted


def _label(account: Account) -> str:
    return account.handle or (account.names[0] if account.names else account.key)
