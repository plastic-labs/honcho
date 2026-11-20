"""GitHub GraphQL objects to Honcho ingest records.

Pure functions over the JSON GitHub's GraphQL API returns, so they can be
tested without a server. A repository is one container, so one session: each
record opens with a line naming the issue, pull request or commit it belongs
to, and carries its number in metadata.

Text is cleaned before it's kept. CodeRabbit writes its generated summary into
the human author's pull request body, so that block is removed, along with HTML
comments, `Co-Authored-By`/`Signed-off-by` trailers and the Claude Code footer:
a bot's words must never become a person's. A bot's own text also loses its
collapsed `<details>` blocks, where bots keep their machinery.
"""

import datetime
import hashlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

SOURCE = "github"

# The ingest API's limits, so one item can't get a whole batch refused.
MAX_NAME = 512
MAX_CONTENT = 25_000
MAX_PARTICIPANTS = 100

# Context, not derived: an approval with no text says who reviewed, nothing more.
NO_REASONING: dict[str, Any] = {"reasoning": {"enabled": False}}

Json = Mapping[str, Any]
Person = dict[str, Any]
Resolve = Callable[[str], Person | None]

_CODERABBIT = re.compile(
    r"<!-- This is an auto-generated comment[\s\S]*?"
    + r"<!-- end of auto-generated comment[^>]*-->"
)
_CODERABBIT_SUMMARY = re.compile(r"^## Summary by CodeRabbit[\s\S]*", re.MULTILINE)
_HTML_COMMENT = re.compile(r"<!--[\s\S]*?-->")
_TRAILER = re.compile(
    r"^\s*(co-authored-by|signed-off-by):.*$", re.IGNORECASE | re.MULTILINE
)
_CLAUDE_FOOTER = re.compile(
    r"^.*generated with \[claude code\].*$", re.IGNORECASE | re.MULTILINE
)
_AI_HINT = re.compile(
    r"co-authored-by:.*(claude|codex|cursor|copilot|gpt)|generated with \[claude code\]",
    re.IGNORECASE,
)
_BLANK_RUNS = re.compile(r"\n{3,}")
# A collapsed block. Bots fold their machinery into these (run configuration,
# prompts for AI agents, walkthrough tables): most of a bot's text, little of
# what it said.
_DETAILS = re.compile(r"<details>(?:(?!<details>)[\s\S])*?</details>", re.IGNORECASE)

# A login: letters, digits and single inner hyphens, at most 39 characters,
# optionally an app's `[bot]` account. Not preceded by a word character (an
# email address) or `/` (a path), and not followed by `/` (a team,
# `@org/team`) or any other bracket.
_MENTION = re.compile(
    r"(?<![\w`/.@])@([A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38})(\[bot\])?"
    + r"(?![\w/\[-])"
)
_FENCED = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~")
_INLINE_CODE = re.compile(r"`[^`\n]*`")

_MERGE_SUBJECT = re.compile(r"^Merge pull request #(\d+) from \S+")
_BRANCH_SYNC = re.compile(r"^Merge (remote-tracking )?branch ")

_REVIEW_STATES: Mapping[str, str] = {
    "APPROVED": "approved",
    "CHANGES_REQUESTED": "changes requested",
    "COMMENTED": "commented",
    "DISMISSED": "dismissed",
}


def items(value: Any) -> list[Json]:
    """A JSON list of objects, or [] when the field is missing or null."""
    return cast(list[Json], value) if isinstance(value, list) else []


def obj(value: Any) -> Json:
    """A JSON object, or {} when the field is missing or null."""
    return cast(Json, value) if isinstance(value, Mapping) else {}


def nodes(connection: Any) -> list[Json]:
    """A GraphQL connection's nodes, skipping nulls."""
    return [n for n in items(obj(connection).get("nodes")) if n]


def canonical_host(host: str) -> str:
    """A host as one name: lowercase, no trailing dot, `api.github.com` is `github.com`."""
    name = host.strip().lower().rstrip(".")
    return "github.com" if name in ("", "api.github.com") else name


def instance_for(host: str) -> str:
    """The ingest `source_instance` for a GitHub host.

    The host, not the organization: an account is the same person in every
    organization on a host, so linking must see it as one account. `github.com`
    is `github-com`; any other host is a readable slug plus a hash of the host,
    so two hosts that slug alike (`a.b-c`, `a-b.c`) stay apart and the name
    fits ingest's pattern and length.
    """
    name = canonical_host(host)
    if name == "github.com":
        return "github-com"
    slug = re.sub(r"[^a-z0-9_-]", "-", name)[:100]
    return f"{slug}-{hashlib.sha256(name.encode()).hexdigest()[:12]}"


def clean(text: Any) -> str:
    """The text a person wrote, without generated blocks, comments or trailers."""
    t = text if isinstance(text, str) else ""
    t = _CODERABBIT.sub("", t)
    t = _CODERABBIT_SUMMARY.sub("", t)
    t = _HTML_COMMENT.sub("", t)
    t = _TRAILER.sub("", t)
    t = _CLAUDE_FOOTER.sub("", t)
    t = t.replace("\x00", "").replace("\r\n", "\n")
    return _BLANK_RUNS.sub("\n\n", t).strip()


def without_collapsed(text: str) -> str:
    """A bot's text without its collapsed `<details>` blocks, innermost first."""
    previous = None
    while previous != text:
        previous = text
        text = _DETAILS.sub("", text)
    return _BLANK_RUNS.sub("\n\n", text).strip()


def ai_assisted(*texts: Any) -> bool:
    """A coding assistant is credited: a co-author trailer or a generated-with footer."""
    return any(isinstance(t, str) and _AI_HINT.search(t) for t in texts)


def mentions(text: str) -> list[str]:
    """The logins a text @-mentions, outside code, in the order first named.

    An app's account keeps its suffix (`dependabot[bot]`), so it's never read
    as a person with the same base name.
    """
    prose = _INLINE_CODE.sub("", _FENCED.sub("", text))
    seen: dict[str, str] = {}
    for login, bot in _MENTION.findall(prose):
        seen.setdefault((login + bot).casefold(), login + bot)
    return list(seen.values())


def login_key(p: Person) -> str:
    """How a known account is found by a mention: a bot's login carries `[bot]`."""
    name = str(p["name"]).casefold()
    if p.get("bot") and not name.endswith("[bot]"):
        return f"{name}[bot]"
    return name


def clean_name(value: Any) -> str | None:
    """A usable display name: NUL-free, trimmed, at most 512 characters."""
    if not isinstance(value, str):
        return None
    name = value.replace("\x00", "").strip()
    return name[:MAX_NAME] or None


def is_bot(actor: Json) -> bool:
    """A GitHub App (`Bot`), or an account named like one (`dependabot[bot]`)."""
    login = str(actor.get("login") or actor.get("name") or "")
    return actor.get("__typename") == "Bot" or login.endswith("[bot]")


def person(actor: Any) -> Person | None:
    """An author or participant: the numeric ID, the login as the name.

    None for a deleted account (GitHub's ghost). The numeric ID survives a
    rename; the login is what GitHub shows on every comment.
    """
    actor = obj(actor)
    name = clean_name(actor.get("login"))
    if name is None:
        return None
    found: Person = {"name": name}
    if actor.get("databaseId") is not None:
        found["id"] = str(actor["databaseId"])
    if is_bot(actor):
        found["bot"] = True
    return found


def account(actor: Any) -> dict[str, Any] | None:
    """What an export keeps about one account, for identity checks."""
    actor = obj(actor)
    found = person(actor)
    if found is None:
        return None
    return {
        "id": found.get("id"),
        "login": found["name"],
        "name": clean_name(actor.get("name")),
        "type": actor.get("__typename"),
        "bot": bool(found.get("bot")),
    }


@dataclass(frozen=True)
class Repo:
    """A repository, as a record's container: one session per repository."""

    id: str
    name_with_owner: str
    owner_id: str | None
    url: str
    private: bool = False

    @classmethod
    def from_node(cls, node: Json) -> "Repo":
        return cls(
            id=str(node["id"]),
            name_with_owner=str(node["nameWithOwner"]),
            owner_id=str(obj(node.get("owner")).get("id") or "") or None,
            url=str(node.get("url") or ""),
            private=bool(node.get("isPrivate")),
        )

    def as_container(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name_with_owner,
            "parent_id": self.owner_id,
        }


@dataclass(frozen=True)
class Thread:
    """An issue or pull request, which every record about it names."""

    kind: str  # "issue" or "pull_request"
    number: int
    title: str
    url: str
    state: str

    @classmethod
    def from_node(cls, node: Json, kind: str) -> "Thread":
        state = str(node.get("state") or "").lower()
        if kind == "pull_request" and node.get("merged"):
            state = "merged"
        return cls(
            kind=kind,
            number=int(node["number"]),
            title=" ".join(str(node.get("title") or "").split()),
            url=str(node.get("url") or ""),
            state=state,
        )

    @property
    def label(self) -> str:
        """`PR #12` or `issue #3`: what a record about it says it's about.

        Only the number, never the title: a title can change without changing
        the record, and ingest would then see the record's text conflict.
        """
        noun = "PR" if self.kind == "pull_request" else "issue"
        return f"{noun} #{self.number}"

    def metadata(self) -> dict[str, Any]:
        """Its number, its state, and `thread`: the conversation inside the
        repository's session, which scopes who a mention is talking to."""
        key = "pr" if self.kind == "pull_request" else "issue"
        return {key: self.number, "thread": f"{key}:{self.number}", "state": self.state}


def _participants(
    text: str, author: Person | None, resolve: Resolve, *extra: Person | None
) -> tuple[list[Person], int]:
    """Who a record names: extra people first, then resolved @mentions.

    One entry per account, never the author. Returns the list, cut at the
    ingest limit, and how many were dropped.
    """
    by_key: dict[str, Person] = {}
    author_key = _key(author) if author else None
    candidates = [*extra, *(resolve(login) for login in mentions(text))]
    for candidate in candidates:
        if candidate is None:
            continue
        key = _key(candidate)
        if key == author_key:
            continue
        kept = by_key.setdefault(key, dict(candidate))
        if candidate.get("bot"):
            kept["bot"] = True
    found = list(by_key.values())
    return found[:MAX_PARTICIPANTS], max(0, len(found) - MAX_PARTICIPANTS)


def utc(timestamp: str) -> str:
    """A GitHub timestamp in UTC, `2026-11-04T10:00:00Z`.

    Commit times (`GitTimestamp`) keep the committer's offset; records are
    sorted by this text, so every timestamp must be in one zone.
    """
    parsed = datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.UTC)
    return parsed.astimezone(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _key(p: Person) -> str:
    return f"id:{p['id']}" if p.get("id") else f"name:{str(p['name']).casefold()}"


def _record(
    repo: Repo,
    instance: str,
    *,
    source_id: str,
    created_at: str,
    revised_at: Any,
    author: Person | None,
    header: str,
    body: str,
    resolve: Resolve,
    metadata: dict[str, Any],
    replied_to: Person | None = None,
    configuration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if author is not None and author.get("bot"):
        body = without_collapsed(body)
    content = f"{header}\n{body}" if body else header
    if len(content) > MAX_CONTENT:
        content = content[:MAX_CONTENT]
        metadata["truncated"] = True
    participants, dropped = _participants(body, author, resolve, replied_to)
    if dropped:
        metadata["participants_dropped"] = dropped
    record: dict[str, Any] = {
        "source": SOURCE,
        "source_instance": instance,
        "source_id": source_id,
        "created_at": utc(created_at),
        "revised_at": utc(revised_at) if isinstance(revised_at, str) else None,
        "container": repo.as_container(),
        "author": author,
        "participants": participants,
        "content": content,
        "metadata": {"repo": repo.name_with_owner, **metadata},
    }
    if configuration is not None:
        record["configuration"] = configuration
    return record


def thread_record(
    node: Json, thread: Thread, repo: Repo, instance: str, resolve: Resolve
) -> dict[str, Any]:
    """An issue or pull request's opening post: its title, then its body."""
    noun = "PR" if thread.kind == "pull_request" else "issue"
    raw = node.get("body")
    return _record(
        repo,
        instance,
        source_id=str(node["id"]),
        created_at=str(node["createdAt"]),
        revised_at=node.get("lastEditedAt"),
        author=person(node.get("author")),
        header=f"Opened {noun} #{thread.number}: {thread.title}",
        body=clean(raw),
        resolve=resolve,
        metadata={
            "kind": thread.kind,
            "url": thread.url,
            **thread.metadata(),
            **({"ai_assisted": True} if ai_assisted(raw) else {}),
        },
    )


def comment_record(
    node: Json, thread: Thread, repo: Repo, instance: str, resolve: Resolve
) -> dict[str, Any] | None:
    """A comment on an issue or pull request, or None if it has no text."""
    raw = node.get("body")
    body = clean(raw)
    if not body:
        return None
    return _record(
        repo,
        instance,
        source_id=str(node["id"]),
        created_at=str(node["createdAt"]),
        revised_at=node.get("lastEditedAt"),
        author=person(node.get("author")),
        header=f"Comment on {thread.label}:",
        body=body,
        resolve=resolve,
        metadata={
            "kind": "comment",
            "url": str(node.get("url") or ""),
            **thread.metadata(),
            **({"ai_assisted": True} if ai_assisted(raw) else {}),
        },
    )


def review_record(
    node: Json, thread: Thread, repo: Repo, instance: str, resolve: Resolve
) -> dict[str, Any] | None:
    """A submitted review: its verdict, then its text.

    An approval or change request with no text is kept as context with
    reasoning off. A pending review, or a comment-only review with no text
    (its inline comments are records of their own), is None.
    """
    state = str(node.get("state") or "")
    verdict = _REVIEW_STATES.get(state)
    if verdict is None:  # PENDING: not submitted, visible only to its author
        return None
    raw = node.get("body")
    body = clean(raw)
    configuration: dict[str, Any] | None = None
    if not body:
        if state not in ("APPROVED", "CHANGES_REQUESTED"):
            return None
        configuration = NO_REASONING
    return _record(
        repo,
        instance,
        source_id=str(node["id"]),
        created_at=str(node.get("submittedAt") or node["createdAt"]),
        revised_at=node.get("lastEditedAt"),
        author=person(node.get("author")),
        header=f"Review of {thread.label}: {verdict}",
        body=body,
        resolve=resolve,
        metadata={
            "kind": "review",
            "review_state": verdict,
            "url": str(node.get("url") or ""),
            **thread.metadata(),
            **({"ai_assisted": True} if ai_assisted(raw) else {}),
        },
        configuration=configuration,
    )


def review_comment_record(
    node: Json,
    thread: Thread,
    repo: Repo,
    instance: str,
    resolve: Resolve,
    replied_to: Json | None = None,
) -> dict[str, Any] | None:
    """An inline review comment, or None if it has no text.

    `replied_to` is the comment it answers, whose author becomes a participant.
    """
    raw = node.get("body")
    body = clean(raw)
    if not body:
        return None
    path = str(node.get("path") or "")
    # The line in the diff the comment was made on: fixed, unlike `line`,
    # which moves as later commits change the file.
    line = node.get("originalLine") or node.get("line")
    where = f"{path}:{line}" if path and line else path
    metadata: dict[str, Any] = {
        "kind": "review_comment",
        "url": str(node.get("url") or ""),
        **thread.metadata(),
    }
    if path:
        metadata["path"] = path
    if line:
        metadata["line"] = int(line)
    reply_to = obj(node.get("replyTo")).get("id")
    if reply_to:
        metadata["reply_to"] = str(reply_to)
    if ai_assisted(raw):
        metadata["ai_assisted"] = True
    return _record(
        repo,
        instance,
        source_id=str(node["id"]),
        created_at=str(node["createdAt"]),
        revised_at=node.get("lastEditedAt"),
        author=person(node.get("author")),
        header=f"Review comment on {thread.label}" + (f", {where}:" if where else ":"),
        body=body,
        resolve=resolve,
        metadata=metadata,
        replied_to=person(obj(replied_to).get("author")) if replied_to else None,
    )


def commit_author(node: Json) -> Person | None:
    """A commit's author: its linked account, or a bot named by git.

    A commit with no linked account and no bot name (an email that matches no
    account) has no author: it can't be tied to a person.
    """
    git_author = obj(node.get("author"))
    user = obj(git_author.get("user"))
    if user:
        return person({"__typename": "User", **user})
    name = clean_name(git_author.get("name"))
    if name and name.endswith("[bot]"):
        return {"name": name, "bot": True}
    return None


def commit_record(
    node: Json,
    repo: Repo,
    instance: str,
    resolve: Resolve,
) -> dict[str, Any] | None:
    """A default-branch commit: what shipped. None for branch syncs or no author.

    A merge commit (`Merge pull request #12 from …`) is named by the title
    line git recorded. `touched` lists the top directories of the commit's own
    pull request, read with the commit, so the text is the same however the
    export was scoped.
    """
    raw = str(node.get("message") or "")
    subject, _, rest = raw.partition("\n")
    subject = subject.strip()
    if not subject or _BRANCH_SYNC.match(subject):
        return None
    author = commit_author(node)
    if author is None:
        return None
    body = clean(rest)
    found = nodes(node.get("associatedPullRequests"))
    associated: Json = found[0] if found else {}
    pr = int(associated["number"]) if associated.get("number") else None
    merged = _MERGE_SUBJECT.match(subject)
    if merged:
        title, _, tail = body.partition("\n")
        subject = f"{title.strip() or 'Merge pull request'} (#{merged.group(1)})"
        body = tail.strip()
        pr = int(merged.group(1))
    text = subject + (f"\n\n{body}" if body else "")
    dirs = touched_dirs([str(f["path"]) for f in nodes(associated.get("files"))])
    if dirs:
        text += f"\n\ntouched: {', '.join(dirs)}"
    metadata: dict[str, Any] = {
        "kind": "commit",
        "sha": str(node.get("oid") or "")[:12],
        "url": str(node.get("url") or ""),
    }
    if pr is not None:
        metadata["pr"] = pr
        metadata["thread"] = f"pr:{pr}"
    if ai_assisted(raw):
        metadata["ai_assisted"] = True
    return _record(
        repo,
        instance,
        source_id=str(node["id"]),
        created_at=str(node["committedDate"]),
        revised_at=None,
        author=author,
        # The repository is in `metadata.repo`, not the text: a commit has no
        # revision, so a renamed repository would change its text for good.
        header="Commit" + (f" (PR #{pr}):" if pr is not None else ":"),
        body=text,
        resolve=resolve,
        metadata=metadata,
    )


def touched_dirs(paths: list[str], limit: int = 6) -> list[str]:
    """A change's most-touched top directories (two levels deep), most first."""
    counts: dict[str, int] = {}
    for path in paths:
        parts = path.split("/")
        # A file at the top is itself; anything deeper is its directory, at
        # most two levels down.
        key = (
            parts[0]
            if len(parts) == 1
            else "/".join(parts[: min(2, len(parts) - 1)]) + "/"
        )
        counts[key] = counts.get(key, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [d for d, _ in ranked[:limit]]
