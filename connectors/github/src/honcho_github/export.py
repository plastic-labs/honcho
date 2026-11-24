"""Walk GitHub repositories and hand each one's records, oldest first, to a sink.

A repository is read whole before any of it is sent: its issues and pull
requests (with their comments, reviews and review comments, however many pages
they take) and its default-branch commits. Its records are then sorted by time
and sent in order, since the ingest API keeps arrival order within a session.
A repository that can't be read completely sends nothing.
"""

import contextlib
import datetime
import json
import logging
import os
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, final

from honcho import BotRecords, filter_bot_records

from honcho_github.client import GitHubClient, GitHubError
from honcho_github.mapping import (
    Json,
    Person,
    Repo,
    Thread,
    account,
    clean,
    comment_record,
    commit_author,
    commit_record,
    instance_for,
    items,
    login_key,
    mentions,
    nodes,
    obj,
    person,
    review_comment_record,
    review_record,
    thread_record,
)

logger = logging.getLogger(__name__)

_ACTOR = (
    "author { __typename login ... on User { databaseId name }"
    + " ... on Bot { databaseId } ... on Mannequin { databaseId } }"
)
_PAGE = "pageInfo { hasNextPage endCursor }"
_COMMIT = (
    "id oid url committedDate message"
    + " author { name user { databaseId login name } }"
    + " associatedPullRequests(first: 1) {"
    + " nodes { number files(first: 100) { nodes { path } } } }"
)
_COMMENT = f"id createdAt lastEditedAt url body {_ACTOR}"
_REVIEW = f"id state body createdAt submittedAt lastEditedAt url {_ACTOR}"
_THREAD_COMMENT = f"id state createdAt lastEditedAt url body path line originalLine replyTo {{ id }} {_ACTOR}"
_THREAD = f"id comments(first: 50) {{ {_PAGE} nodes {{ {_THREAD_COMMENT} }} }}"
_REPO = (
    "id nameWithOwner url isPrivate isArchived isFork owner { id login }"
    + " pullRequests { totalCount } issues { totalCount }"
)

REPO_QUERY = f"""
query($owner: String!, $name: String!) {{
  rateLimit {{ cost remaining }}
  repository(owner: $owner, name: $name) {{ {_REPO} }}
}}"""

OWNER_REPOS_QUERY = f"""
query($login: String!, $after: String) {{
  rateLimit {{ cost remaining }}
  repositoryOwner(login: $login) {{
    repositories(first: 100, after: $after, privacy: PUBLIC,
                 ownerAffiliations: [OWNER], orderBy: {{field: NAME, direction: ASC}}) {{
      {_PAGE} nodes {{ {_REPO} }}
    }}
  }}
}}"""

ISSUES_QUERY = f"""
query($owner: String!, $name: String!, $after: String) {{
  rateLimit {{ cost remaining }}
  repository(owner: $owner, name: $name) {{
    issues(first: 50, after: $after, orderBy: {{field: UPDATED_AT, direction: DESC}}) {{
      {_PAGE}
      nodes {{
        id number title url state createdAt updatedAt lastEditedAt body {_ACTOR}
        comments(first: 100) {{ {_PAGE} nodes {{ {_COMMENT} }} }}
      }}
    }}
  }}
}}"""

PULL_REQUESTS_QUERY = f"""
query($owner: String!, $name: String!, $after: String) {{
  rateLimit {{ cost remaining }}
  repository(owner: $owner, name: $name) {{
    pullRequests(first: 20, after: $after, orderBy: {{field: UPDATED_AT, direction: DESC}}) {{
      {_PAGE}
      nodes {{
        id number title url state merged createdAt updatedAt lastEditedAt body {_ACTOR}
        comments(first: 50) {{ {_PAGE} nodes {{ {_COMMENT} }} }}
        reviews(first: 50) {{ {_PAGE} nodes {{ {_REVIEW} }} }}
        reviewThreads(first: 50) {{ {_PAGE} nodes {{ {_THREAD} }} }}
      }}
    }}
  }}
}}"""

COMMITS_QUERY = f"""
query($owner: String!, $name: String!, $after: String, $since: GitTimestamp) {{
  rateLimit {{ cost remaining }}
  repository(owner: $owner, name: $name) {{
    defaultBranchRef {{
      target {{
        ... on Commit {{
          history(first: 100, after: $after, since: $since) {{
            {_PAGE}
            nodes {{ {_COMMIT} }}
          }}
        }}
      }}
    }}
  }}
}}"""

HEAD_QUERY = """
query($owner: String!, $name: String!) {
  rateLimit { cost remaining }
  repository(owner: $owner, name: $name) { defaultBranchRef { target { oid } } }
}"""


def _rest_query(typename: str, connection: str, fields: str) -> str:
    return f"""
query($id: ID!, $after: String) {{
  rateLimit {{ cost remaining }}
  node(id: $id) {{
    ... on {typename} {{
      {connection}(first: 100, after: $after) {{ {_PAGE} nodes {{ {fields} }} }}
    }}
  }}
}}"""


class Sink(Protocol):
    """Where exported records go."""

    def container(self, repo: Repo) -> None:
        """A repository's records follow."""
        ...

    def record(self, record: dict[str, Any]) -> None: ...

    def close(
        self, accounts: Iterable[dict[str, Any]], summary: dict[str, Any]
    ) -> None:
        """The export finished: flush, and keep the accounts and summary."""
        ...


class SelectionError(ValueError):
    """The requested repositories can't be exported as asked."""


@dataclass(frozen=True)
class Listed:
    """A repository an export would read, with its size."""

    repo: Repo
    pull_requests: int
    issues: int
    archived: bool
    fork: bool


def _listed(node: Json) -> Listed:
    return Listed(
        repo=Repo.from_node(node),
        pull_requests=int(obj(node.get("pullRequests")).get("totalCount") or 0),
        issues=int(obj(node.get("issues")).get("totalCount") or 0),
        archived=bool(node.get("isArchived")),
        fork=bool(node.get("isFork")),
    )


def _split(full_name: str) -> tuple[str, str]:
    owner, sep, name = full_name.strip().partition("/")
    if not sep or not owner or not name or "/" in name:
        raise SelectionError(f"{full_name!r} isn't owner/name")
    return owner, name


def select(
    client: GitHubClient, repos: Sequence[str], owners: Sequence[str]
) -> list[Listed]:
    """The repositories to export: each named one, and each owner's public ones.

    An owner's listing skips private, archived and forked repositories; a
    private repository is read only when it's named.

    Raises:
        SelectionError: nothing was asked for, a name isn't `owner/name`, or a
            repository or owner doesn't exist or can't be read with this token.
    """
    if not repos and not owners:
        raise SelectionError("Name a repository (--repo) or an owner (--org)")
    chosen: dict[str, Listed] = {}
    for full_name in repos:
        owner, name = _split(full_name)
        data = client.query(
            REPO_QUERY, {"owner": owner, "name": name}, allow_not_found=True
        )
        node = obj(data.get("repository"))
        if not node:
            raise SelectionError(
                f"{full_name} doesn't exist or this token can't read it"
            )
        listed = _listed(node)
        chosen.setdefault(listed.repo.id, listed)
    for login in owners:
        after: str | None = None
        found_owner = False
        while True:
            data = client.query(
                OWNER_REPOS_QUERY,
                {"login": login, "after": after},
                allow_not_found=True,
            )
            owner_node = obj(data.get("repositoryOwner"))
            if not owner_node:
                break
            found_owner = True
            connection = obj(owner_node.get("repositories"))
            for node in nodes(connection):
                listed = _listed(node)
                if listed.archived or listed.fork or listed.repo.private:
                    continue
                chosen.setdefault(listed.repo.id, listed)
            page = obj(connection.get("pageInfo"))
            if not page.get("hasNextPage"):
                break
            after = str(page["endCursor"])
        if not found_owner:
            raise SelectionError(f"No GitHub account or organization named {login}")
    return sorted(
        chosen.values(), key=lambda listed: listed.repo.name_with_owner.lower()
    )


@dataclass
class Fetched:
    """Everything one repository holds, read in full."""

    threads: list[
        tuple[Thread, Json, list[Json], list[Json], list[tuple[Json, list[Json]]]]
    ]
    commits: list[Json]


def _all(
    client: GitHubClient,
    node_id: str,
    typename: str,
    connection: str,
    fields: str,
    first: Any,
) -> list[Json]:
    """A connection's nodes: the first page given, then any further pages."""
    found = nodes(first)
    page = obj(obj(first).get("pageInfo"))
    query = _rest_query(typename, connection, fields)
    while page.get("hasNextPage"):
        data = client.query(query, {"id": node_id, "after": page["endCursor"]})
        conn = obj(obj(data.get("node")).get(connection))
        found.extend(nodes(conn))
        page = obj(conn.get("pageInfo"))
    return found


def _paged(
    client: GitHubClient,
    query: str,
    variables: dict[str, Any],
    pick: Callable[[dict[str, Any]], Json],
    keep: Callable[[Json], bool] = lambda _node: True,
) -> list[Json]:
    """Every node of a top-level connection, stopping at the first one `keep` refuses.

    The connections paged here are ordered newest first, so the first node
    older than `--since` ends the walk.
    """
    found: list[Json] = []
    after: str | None = None
    while True:
        conn = pick(client.query(query, {**variables, "after": after}))
        for node in nodes(conn):
            if not keep(node):
                return found
            found.append(node)
        page = obj(conn.get("pageInfo"))
        if not page.get("hasNextPage"):
            return found
        after = str(page["endCursor"])


def read_commits(
    client: GitHubClient,
    repo: Repo,
    *,
    since: str | None = None,
    limit: int | None = None,
) -> tuple[list[Json], bool]:
    """Default-branch commits, newest first, and whether `limit` cut them short.

    All of them, or those made since `since`.
    """
    owner, name = _split(repo.name_with_owner)
    count = 0
    cut = False

    def keep(_node: Json) -> bool:
        nonlocal count, cut
        count += 1
        if limit is not None and count > limit:
            cut = True
            return False
        return True

    commits = _paged(
        client,
        COMMITS_QUERY,
        {"owner": owner, "name": name, "since": since},
        lambda d: obj(
            obj(
                obj(obj(obj(d.get("repository")).get("defaultBranchRef")).get("target"))
            ).get("history")
        ),
        keep,
    )
    return commits, cut


def head_commit(client: GitHubClient, repo: Repo) -> str | None:
    """The SHA at the head of the default branch; None for an empty repository."""
    owner, name = _split(repo.name_with_owner)
    data = client.query(HEAD_QUERY, {"owner": owner, "name": name})
    target = obj(obj(obj(data.get("repository")).get("defaultBranchRef")).get("target"))
    return str(target["oid"]) if target.get("oid") else None


_COMPARE_PAGE = 100
_COMMIT_BATCH = 25


def commits_between(
    client: GitHubClient, repo: Repo, base: str, head: str, *, limit: int
) -> tuple[list[Json], bool] | None:
    """The commits `head` has that `base` hasn't, and whether `limit` cut them short.

    That's what `git log base..head` lists: the commits pushed since, including
    those of a branch merged since, however long ago they were made. The newest
    `limit` are kept. None when GitHub doesn't know `base` (history rewritten).
    """
    owner, name = _split(repo.name_with_owner)
    path = f"/repos/{owner}/{name}/compare/{base}...{head}"
    first = client.rest(path, {"per_page": _COMPARE_PAGE, "page": 1})
    if first is None:
        return None
    total = int(obj(first).get("total_commits") or 0)
    last_page = max(1, -(-total // _COMPARE_PAGE))
    from_page = max(0, total - limit) // _COMPARE_PAGE + 1
    shas: list[str] = []
    # Pages run oldest first: read the last ones, for the newest commits.
    for page in range(from_page, last_page + 1):
        body = (
            first
            if page == 1
            else client.rest(path, {"per_page": _COMPARE_PAGE, "page": page})
        )
        if body is None:
            return None
        shas.extend(
            str(c["sha"]) for c in items(obj(body).get("commits")) if c.get("sha")
        )
    shas = shas[-limit:]
    return commits_by_sha(client, repo, shas[::-1]), total > len(shas)


def commits_by_sha(client: GitHubClient, repo: Repo, shas: Sequence[str]) -> list[Json]:
    """These commits, as the commit history query reads them, in the same order."""
    owner, name = _split(repo.name_with_owner)
    found: list[Json] = []
    for start in range(0, len(shas), _COMMIT_BATCH):
        batch = shas[start : start + _COMMIT_BATCH]
        params = ", ".join(f"$o{i}: GitObjectID!" for i in range(len(batch)))
        fields = " ".join(
            f"c{i}: object(oid: $o{i}) {{ ... on Commit {{ {_COMMIT} }} }}"
            for i in range(len(batch))
        )
        data = client.query(
            f"query($owner: String!, $name: String!, {params}) {{"
            + " rateLimit { cost remaining }"
            + f" repository(owner: $owner, name: $name) {{ {fields} }} }}",
            {"owner": owner, "name": name}
            | {f"o{i}": sha for i, sha in enumerate(batch)},
        )
        repository = obj(data.get("repository"))
        for i in range(len(batch)):
            node = obj(repository.get(f"c{i}"))
            if node:
                found.append(node)
    return found


def fetch(
    client: GitHubClient, repo: Repo, since: str | None, *, with_commits: bool = True
) -> Fetched:
    """Read a repository's issues, pull requests and (unless not wanted)
    default-branch commits."""
    owner, name = _split(repo.name_with_owner)
    where = {"owner": owner, "name": name}

    def updated_since(node: Json) -> bool:
        return since is None or str(node.get("updatedAt") or "") >= since

    issues = _paged(
        client,
        ISSUES_QUERY,
        where,
        lambda d: obj(obj(d.get("repository")).get("issues")),
        updated_since,
    )
    pulls = _paged(
        client,
        PULL_REQUESTS_QUERY,
        where,
        lambda d: obj(obj(d.get("repository")).get("pullRequests")),
        updated_since,
    )
    threads: list[
        tuple[Thread, Json, list[Json], list[Json], list[tuple[Json, list[Json]]]]
    ] = []
    for node in issues:
        node_id = str(node["id"])
        comments = _all(
            client, node_id, "Issue", "comments", _COMMENT, node.get("comments")
        )
        threads.append((Thread.from_node(node, "issue"), node, comments, [], []))
    for node in pulls:
        node_id = str(node["id"])
        thread = Thread.from_node(node, "pull_request")
        comments = _all(
            client, node_id, "PullRequest", "comments", _COMMENT, node.get("comments")
        )
        # A pending review, and its inline comments, are drafts visible only to
        # their author (and to an export run with that author's token).
        reviews = [
            r
            for r in _all(
                client, node_id, "PullRequest", "reviews", _REVIEW, node.get("reviews")
            )
            if r.get("state") != "PENDING"
        ]
        review_threads: list[tuple[Json, list[Json]]] = []
        for review_thread in _all(
            client,
            node_id,
            "PullRequest",
            "reviewThreads",
            _THREAD,
            node.get("reviewThreads"),
        ):
            thread_comments = _all(
                client,
                str(review_thread["id"]),
                "PullRequestReviewThread",
                "comments",
                _THREAD_COMMENT,
                review_thread.get("comments"),
            )
            submitted = [c for c in thread_comments if c.get("state") != "PENDING"]
            if submitted:
                review_threads.append((review_thread, submitted))
        threads.append((thread, node, comments, reviews, review_threads))

    commits = read_commits(client, repo, since=since)[0] if with_commits else []
    return Fetched(threads=threads, commits=commits)


_USER_BATCH = 50


def resolve_logins(client: GitHubClient, logins: Iterable[str]) -> dict[str, Json]:
    """Look up accounts by login, 50 to a query. Unknown logins are left out."""
    wanted = sorted({login.casefold(): login for login in logins}.values())
    found: dict[str, Json] = {}
    for start in range(0, len(wanted), _USER_BATCH):
        batch = wanted[start : start + _USER_BATCH]
        params = ", ".join(f"$l{i}: String!" for i in range(len(batch)))
        fields = " ".join(
            f"u{i}: user(login: $l{i}) {{ __typename login databaseId name }}"
            for i in range(len(batch))
        )
        data = client.query(
            f"query({params}) {{ rateLimit {{ cost remaining }} {fields} }}",
            {f"l{i}": login for i, login in enumerate(batch)},
            allow_not_found=True,
        )
        for i in range(len(batch)):
            user = obj(data.get(f"u{i}"))
            if user.get("login"):
                found[str(user["login"]).casefold()] = user
    return found


def records_for(
    client: GitHubClient,
    repo: Repo,
    fetched: Fetched,
    instance: str,
    accounts: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """A repository's records, oldest first, and what was skipped.

    Adds every author and resolved participant to `accounts`.
    """
    known: dict[str, Person] = {}

    def note(actor: Any) -> None:
        found = account(actor)
        if found is None:
            return
        p = person(actor)
        if p is not None:
            known.setdefault(login_key(p), p)
        accounts.setdefault(found["id"] or f"login:{found['login']}", found)

    texts: list[str] = []
    for thread, node, comments, reviews, review_threads in fetched.threads:
        del thread
        note(node.get("author"))
        texts.append(clean(node.get("body")))
        for item in [*comments, *reviews, *(c for _, cs in review_threads for c in cs)]:
            note(item.get("author"))
            texts.append(clean(item.get("body")))
    for commit in fetched.commits:
        user = obj(obj(commit.get("author")).get("user"))
        if user:
            note({"__typename": "User", **user})
        texts.append(clean(commit.get("message")))

    def resolve(login: str) -> Person | None:
        """A mention's account: `x[bot]` only as a bot; `x` as a person, else an
        app whose slug is `x` (`@coderabbitai`)."""
        key = login.casefold()
        if key.endswith("[bot]"):
            return known.get(key)
        return known.get(key) or known.get(f"{key}[bot]")

    # Only a person can be looked up by login; a bot is known from what it
    # wrote. A plain `@x` is looked up unless a person `x` is already known,
    # even when an app `x` is: the person comes first, the app is the fallback.
    unknown = {
        login
        for text in texts
        for login in mentions(text)
        if not login.endswith("[bot]") and login.casefold() not in known
    }
    for user in resolve_logins(client, unknown).values():
        note(user)

    records: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()

    def keep(record: dict[str, Any] | None, why: str) -> None:
        if record is None:
            skipped[why] += 1
        else:
            records.append(record)

    for thread, node, comments, reviews, review_threads in fetched.threads:
        records.append(thread_record(node, thread, repo, instance, resolve))
        for comment in comments:
            keep(comment_record(comment, thread, repo, instance, resolve), "empty")
        for review in reviews:
            keep(review_record(review, thread, repo, instance, resolve), "empty_review")
        for _, thread_comments in review_threads:
            by_id = {str(c["id"]): c for c in thread_comments}
            for comment in thread_comments:
                replied = by_id.get(str(obj(comment.get("replyTo")).get("id")))
                keep(
                    review_comment_record(
                        comment, thread, repo, instance, resolve, replied
                    ),
                    "empty",
                )
    for commit in fetched.commits:
        if commit_author(commit) is None and str(commit.get("message") or "").strip():
            skipped["commit_without_account"] += 1
            continue
        keep(
            commit_record(
                commit,
                repo,
                instance,
                resolve,
            ),
            "branch_sync",
        )
    for p in (r["author"] for r in records if r["author"] and r["author"].get("bot")):
        # A bot git names without an account (`dependabot[bot]` on a commit).
        accounts.setdefault(
            p.get("id") or f"login:{p['name']}",
            {
                "id": p.get("id"),
                "login": p["name"],
                "name": None,
                "type": "Bot",
                "bot": True,
            },
        )
    records.sort(key=lambda r: (r["created_at"], r["source_id"]))
    return records, skipped


@dataclass
class Stats:
    records: int = 0
    repos: int = 0
    skipped: Counter[str] = field(default_factory=Counter)
    unreadable: dict[str, str] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return not self.unreadable


def export(
    client: GitHubClient,
    repos: Sequence[str],
    owners: Sequence[str],
    make_sink: Callable[[], Sink],
    *,
    since: str | None = None,
    progress: Callable[[str], None] = logger.info,
) -> tuple[Stats, Sink]:
    """Read the selected repositories into a sink, each one oldest first.

    The sink is made only once the selection is known to be valid, so a bad
    selection never creates a workspace or an output file. `since` (ISO 8601)
    keeps the issues and pull requests updated since then, each in full, and
    the commits made since then.
    """
    selected = select(client, repos, owners)
    sink = make_sink()
    instance = instance_for(client.host)
    stats = Stats()
    accounts: dict[str, dict[str, Any]] = {}
    for listed in selected:
        repo = listed.repo
        try:
            fetched = fetch(client, repo, since)
            records, skipped = records_for(client, repo, fetched, instance, accounts)
        except GitHubError as error:
            logger.warning("Couldn't read %s: %s", repo.name_with_owner, error)
            stats.unreadable[repo.name_with_owner] = str(error)
            continue
        sink.container(repo)
        for record in records:
            sink.record(record)
        stats.repos += 1
        stats.records += len(records)
        stats.skipped.update(skipped)
        progress(f"{repo.name_with_owner}: {len(records)} records")

    summary = {
        "source": "github",
        "source_instance": instance,
        "host": client.host,
        "exported_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "since": since,
        "records": stats.records,
        "repos": [listed.repo.name_with_owner for listed in selected],
        "skipped": dict(stats.skipped),
        "unreadable_repos": stats.unreadable,
        "complete": stats.complete,
        "graphql_cost": client.cost,
    }
    sink.close(accounts.values(), summary)
    return stats, sink


def inside_git_worktree(path: Path) -> bool:
    """Whether `path` is inside a git checkout, where exported data could get committed."""
    for parent in [path.resolve(), *path.resolve().parents]:
        if (parent / ".git").exists():
            return True
    return False


@final
class JsonlSink:
    """Writes `records.jsonl`, `accounts.jsonl`, `containers.jsonl` and `export.json`.

    Refuses a directory inside a git checkout unless `allow_git_dir`, so exported
    text doesn't end up committed. The directory is opened once, refusing a
    symlink, and every file is created inside that open directory without
    following symlinks, owner-only, so nothing redirects the writes afterwards.
    """

    def __init__(self, directory: Path, *, allow_git_dir: bool = False):
        if inside_git_worktree(directory) and not allow_git_dir:
            raise SelectionError(
                f"{directory} is inside a git checkout, where exported text"
                + " could be committed. Choose another directory."
            )
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory: Path = directory
        self._dir: int = os.open(
            directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        self._records: Any = self._create("records.jsonl")
        self._containers: Any = self._create("containers.jsonl")

    def _create(self, name: str) -> Any:
        # A file left by an earlier export is unlinked, not truncated: it could
        # be a hard link to a file elsewhere, which truncating would overwrite.
        # The new file is created exclusively, so nothing can be swapped in
        # between.
        with contextlib.suppress(FileNotFoundError):
            os.unlink(name, dir_fd=self._dir)
        fd = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=self._dir,
        )
        return os.fdopen(fd, "w", encoding="utf-8")

    def container(self, repo: Repo) -> None:
        self._containers.write(
            json.dumps(
                {**repo.as_container(), "url": repo.url, "private": repo.private}
            )
            + "\n"
        )

    def record(self, record: dict[str, Any]) -> None:
        self._records.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(
        self, accounts: Iterable[dict[str, Any]], summary: dict[str, Any]
    ) -> None:
        self._records.close()
        self._containers.close()
        with self._create("accounts.jsonl") as f:
            for item in sorted(
                accounts, key=lambda a: (a["id"] is None, a["id"] or "", a["login"])
            ):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        with self._create("export.json") as f:
            f.write(json.dumps(summary, indent=2) + "\n")
        os.close(self._dir)


class IngestError(Exception):
    """The ingest API refused records."""


class _Ingester(Protocol):
    def ingest(self, records: Iterable[Any], *, derive: Any = ...) -> Any: ...


@final
class IngestSink:
    """Sends records through the Python SDK's `ingest()`, deriving later.

    The workspace is created as an auto workspace if it doesn't exist; ingest
    writes only to auto workspaces. Records are sent with `derive: "later"`,
    a repository at a time, so nothing is derived until someone calls
    `POST /ingest/derive` (or `honcho ingest derive`).

    `bots` picks the bot records sent (the SDK's `filter_bot_records`): by
    default only those a person interacted with. A repository is filtered as a
    whole, since a reply can come after what it answers.
    """

    def __init__(
        self,
        base_url: str,
        workspace: str,
        *,
        api_key: str | None = None,
        client: _Ingester | None = None,
        bots: BotRecords = "interacted",
    ):
        if client is None:
            from honcho import Honcho

            client = Honcho(
                base_url=base_url, api_key=api_key, workspace_id=workspace, auto=True
            )
        self._client: _Ingester = client
        self.workspace: str = workspace
        self.outcomes: Counter[str] = Counter()
        self.rejections: Counter[str] = Counter()
        self.held_back: int = 0
        self._bots: BotRecords = bots
        self._batch: list[dict[str, Any]] = []

    def container(self, repo: Repo) -> None:
        del repo
        self._flush()

    def record(self, record: dict[str, Any]) -> None:
        self._batch.append(record)

    def _flush(self) -> None:
        if not self._batch:
            return
        from honcho import HonchoError

        pending, self._batch = self._batch, []
        try:
            batch, held_back = filter_bot_records(pending, keep=self._bots)
            self.held_back += held_back
            if not batch:
                return
            response = self._client.ingest(batch, derive="later")
        except HonchoError as error:
            # Only the status: the SDK's message can quote the records it sent.
            raise IngestError(
                f"Ingest refused a batch: {type(error).__name__} {error.status or ''}".rstrip()
            ) from None
        except ValueError as error:
            # The filter and the SDK validate records; their messages quote them.
            raise IngestError(
                f"A record didn't validate: {type(error).__name__}"
            ) from None
        for result in response.records:
            self.outcomes[str(result.outcome)] += 1
            if result.reason:
                self.rejections[str(result.reason)] += 1

    def close(
        self, accounts: Iterable[dict[str, Any]], summary: dict[str, Any]
    ) -> None:
        # The server keeps what ingest built; accounts and the summary are for
        # file exports.
        del accounts, summary
        self._flush()
