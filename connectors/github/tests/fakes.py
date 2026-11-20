"""A fake GitHub GraphQL API serving fixed data, paging where asked, for the tests."""

import json
from typing import Any, final

import httpx
from honcho_github.client import GitHubClient
from honcho_github.mapping import utc

ALEX = {"__typename": "User", "login": "alex", "databaseId": 11, "name": "Alex Doe"}
SAM = {"__typename": "User", "login": "sam", "databaseId": 22, "name": None}
RABBIT = {"__typename": "Bot", "login": "coderabbitai", "databaseId": 99}
TOKEN = "ghp_secret_token_value"


def page(nodes: list[Any], cursor: str | None = None) -> dict[str, Any]:
    return {
        "pageInfo": {"hasNextPage": cursor is not None, "endCursor": cursor},
        "nodes": nodes,
    }


def repo(name: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": f"R_{name}",
        "nameWithOwner": f"acme/{name}",
        "url": f"https://github.com/acme/{name}",
        "isPrivate": False,
        "isArchived": False,
        "isFork": False,
        "owner": {"id": "O_acme", "login": "acme"},
        "pullRequests": {"totalCount": 2},
        "issues": {"totalCount": 1},
        **extra,
    }


def item(item_id: str, at: str, author: Any, body: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": item_id,
        "createdAt": at,
        "lastEditedAt": None,
        "url": f"https://github.com/x/{item_id}",
        "author": author,
        "body": body,
        **extra,
    }


@final
class FakeGitHub:
    """Serves the connector's queries from fixed data, paging where asked."""

    def __init__(self) -> None:
        self.queries: list[tuple[str, dict[str, Any]]] = []
        self.repos = {
            "web": repo("web"),
            "api": repo("api"),
            "secret": repo("secret", isPrivate=True),
            "old": repo("old", isArchived=True),
            "fork": repo("fork", isFork=True),
        }
        self.forbidden: set[str] = set()
        self.issues = [
            {
                **item("I_1", "2026-10-15T09:00:00Z", ALEX, "It crashes."),
                "number": 3,
                "title": "Crash",
                "state": "OPEN",
                "updatedAt": "2026-11-05T00:00:00Z",
                "comments": page(
                    [item("IC_1", "2026-10-20T10:00:00Z", SAM, "Same here.")], "c1"
                ),
            }
        ]
        self.issue_comments_rest = [
            item("IC_2", "2026-11-02T10:00:00Z", ALEX, "Fixed by @newbie, not @nobody.")
        ]
        thread_comments = [
            item(
                "RC_1",
                "2026-11-03T11:00:00Z",
                SAM,
                "Why this?",
                path="src/auth/a.py",
                line=4,
                originalLine=4,
                replyTo=None,
                state="SUBMITTED",
            ),
            item(
                "RC_2",
                "2026-11-03T12:00:00Z",
                ALEX,
                "Because.",
                path="src/auth/a.py",
                line=4,
                originalLine=4,
                replyTo={"id": "RC_1"},
                state="SUBMITTED",
            ),
            item(
                "RC_P",
                "2026-11-03T12:30:00Z",
                ALEX,
                "Draft: ask @secretuser",
                path="src/auth/a.py",
                line=4,
                originalLine=4,
                replyTo=None,
                state="PENDING",
            ),
        ]
        self.thread_comments_rest = [
            item(
                "RC_3",
                "2026-11-03T12:45:00Z",
                ALEX,
                "Done.",
                path="src/auth/a.py",
                line=4,
                originalLine=4,
                replyTo={"id": "RC_1"},
                state="SUBMITTED",
            )
        ]
        self.looked_up: list[str] = []
        self.pulls = [
            {
                **item("PR_12", "2026-11-03T08:00:00Z", ALEX, "Adds login."),
                "number": 12,
                "title": "Add login",
                "state": "MERGED",
                "merged": True,
                "updatedAt": "2026-11-06T00:00:00Z",
                "comments": page(
                    [
                        item(
                            "PC_1",
                            "2026-11-03T08:01:00Z",
                            RABBIT,
                            "Summary.\n<details><summary>x</summary>machinery</details>",
                        ),
                        item(
                            "PC_2",
                            "2026-11-03T08:02:00Z",
                            SAM,
                            "@coderabbitai review please; thanks @sam[bot]",
                        ),
                    ]
                ),
                "reviews": page(
                    [
                        {
                            **item("PRR_1", "2026-11-03T13:00:00Z", SAM, ""),
                            "state": "APPROVED",
                            "submittedAt": "2026-11-03T13:00:00Z",
                        },
                        {
                            **item(
                                "PRR_P",
                                "2026-11-03T14:00:00Z",
                                ALEX,
                                "draft @secretuser",
                            ),
                            "state": "PENDING",
                            "submittedAt": None,
                        },
                    ]
                ),
                "reviewThreads": page(
                    [{"id": "T_1", "comments": page(thread_comments, "t1")}]
                ),
            },
            {
                **item("PR_13", "2026-09-01T08:00:00Z", SAM, "Old work."),
                "number": 13,
                "title": "Old",
                "state": "CLOSED",
                "merged": False,
                "updatedAt": "2026-10-01T00:00:00Z",
                "comments": page([]),
                "reviews": page([]),
                "reviewThreads": page([]),
            },
        ]
        self.commits: list[dict[str, Any]] = [
            {
                "id": "C_1",
                "oid": "abcdef0123456789",
                "url": "https://github.com/acme/web/commit/abc",
                "committedDate": "2026-11-06T09:00:00Z",
                "message": "feat: add login (#12)\n\nAdds login.",
                "author": {"name": "Alex", "user": {"login": "alex", "databaseId": 11}},
                "associatedPullRequests": {
                    "nodes": [
                        {
                            "number": 12,
                            "files": {
                                "nodes": [
                                    {"path": "src/auth/a.py"},
                                    {"path": "tests/t.py"},
                                ]
                            },
                        }
                    ]
                },
            },
            {
                "id": "C_0",
                "oid": "0000000000000000",
                "url": "u",
                "committedDate": "2026-11-06T10:00:00Z",
                "message": "chore: tidy",
                "author": {"name": "Unknown", "user": None},
                "associatedPullRequests": {"nodes": []},
            },
            {
                "id": "C_old",
                "oid": "1111111111111111",
                "url": "u",
                "committedDate": "2026-10-01T11:00:00+02:00",
                "message": "chore: old work",
                "author": {"name": "Sam", "user": {"login": "sam", "databaseId": 22}},
                "associatedPullRequests": {"nodes": []},
            },
        ]
        # What compare answers for (base, head), as SHAs oldest first, where
        # the history's order alone would be wrong (a merged branch).
        self.compared: dict[tuple[str, str], list[str]] = {}
        # Commits reachable only through a merge: not in `commits`' order.
        self.merged: list[dict[str, Any]] = []
        self.users: dict[str, dict[str, Any]] = {
            "newbie": {
                "__typename": "User",
                "login": "newbie",
                "databaseId": 33,
                "name": "New",
            }
        }

    @staticmethod
    def _project(query: str, comments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Inline comments as GitHub returns them: `state` only if the query asks."""
        if "id state createdAt" in query:
            return comments
        return [{k: v for k, v in c.items() if k != "state"} for c in comments]

    def _project_pull(self, query: str, pull: dict[str, Any]) -> dict[str, Any]:
        threads = pull["reviewThreads"]
        return {
            **pull,
            "reviewThreads": {
                **threads,
                "nodes": [
                    {
                        **t,
                        "comments": {
                            **t["comments"],
                            "nodes": self._project(query, t["comments"]["nodes"]),
                        },
                    }
                    for t in threads["nodes"]
                ],
            },
        }

    def _compare(self, request: httpx.Request) -> httpx.Response:
        """REST compare: the commits `head` has that `base` hasn't, oldest first."""
        self.queries.append((request.url.path, dict(request.url.params)))
        base, _, head = request.url.path.rsplit("/", 1)[1].partition("...")
        shas = [c["oid"] for c in self.commits]
        if (base, head) in self.compared:
            ahead = self.compared[(base, head)]
        elif base in shas and head in shas:
            ahead = list(reversed(shas[shas.index(head) : shas.index(base)]))
        else:
            return httpx.Response(404, json={"message": "Not Found"})
        size = int(request.url.params.get("per_page", 250))
        number = int(request.url.params.get("page", 1))
        listed = ahead[(number - 1) * size : number * size]
        return httpx.Response(
            200,
            json={
                "status": "ahead",
                "total_commits": len(ahead),
                "commits": [{"sha": sha} for sha in listed],
            },
        )

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and "/compare/" in request.url.path:
            return self._compare(request)
        body = json.loads(request.content)
        query: str = body["query"]
        variables: dict[str, Any] = body["variables"]
        self.queries.append((query, variables))
        name = variables.get("name")
        if (
            name in self.forbidden
            and "repository(owner" in query
            and "{ id nameWithOwner" not in query
        ):
            return httpx.Response(
                200,
                json={"data": None, "errors": [{"type": "FORBIDDEN", "message": "no"}]},
            )
        data: dict[str, Any] = {"rateLimit": {"cost": 1, "remaining": 4000}}
        if "repositoryOwner(" in query:
            if variables["login"] != "acme":
                return httpx.Response(
                    200,
                    json={
                        "data": {"repositoryOwner": None},
                        "errors": [{"type": "NOT_FOUND"}],
                    },
                )
            data["repositoryOwner"] = {"repositories": page(list(self.repos.values()))}
        elif "user(login:" in query:
            errors: list[dict[str, Any]] = []
            for key, login in variables.items():
                self.looked_up.append(login)
                user = self.users.get(login)
                data[f"u{key[1:]}"] = user
                if user is None:
                    errors.append({"type": "NOT_FOUND", "path": [f"u{key[1:]}"]})
            return httpx.Response(200, json={"data": data, "errors": errors or None})
        elif "node(id: $id)" in query:
            if variables["id"] == "I_1":
                assert variables["after"] == "c1"
                assert "... on Issue {" in query
                data["node"] = {"comments": page(self.issue_comments_rest)}
            else:
                assert variables == {"id": "T_1", "after": "t1"}
                assert "... on PullRequestReviewThread {" in query
                rest = self._project(query, self.thread_comments_rest)
                data["node"] = {"comments": page(rest)}
        elif "issues(first:" in query:
            issues = self.issues if name == "web" else []
            data["repository"] = {"issues": page(issues)}
        elif "pullRequests(first:" in query:
            pulls = self.pulls if name == "web" else []
            data["repository"] = {
                "pullRequests": page([self._project_pull(query, p) for p in pulls])
            }
        elif "defaultBranchRef { target { oid } }" in query:
            head = self.commits[0]["oid"] if name == "web" and self.commits else None
            target = {"oid": head} if head else None
            data["repository"] = {
                "defaultBranchRef": {"target": target} if target else None
            }
        elif "object(oid:" in query:
            known = {c["oid"]: c for c in [*self.commits, *self.merged]}
            data["repository"] = {
                f"c{key[1:]}": known.get(sha)
                for key, sha in variables.items()
                if key[:1] == "o" and key[1:].isdigit()
            }
        elif "history(first:" in query:
            # Filter only if the query really passes `since` to GitHub.
            since = variables.get("since") if "since: $since" in query else None
            commits = [
                c
                for c in (self.commits if name == "web" else [])
                if since is None or utc(c["committedDate"]) >= since
            ]
            data["repository"] = {
                "defaultBranchRef": {"target": {"history": page(commits)}}
            }
        else:
            repo = self.repos.get(name or "")
            if repo is None:
                return httpx.Response(
                    200,
                    json={
                        "data": {"repository": None},
                        "errors": [{"type": "NOT_FOUND"}],
                    },
                )
            data["repository"] = repo
        return httpx.Response(200, json={"data": data})


def client_for(fake: FakeGitHub, slept: list[float] | None = None) -> GitHubClient:
    return GitHubClient(
        TOKEN,
        transport=httpx.MockTransport(fake.handle),
        sleep=(slept.append if slept is not None else lambda _s: None),
        now=lambda: 1_000.0,
    )
