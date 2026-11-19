"""GitHub GraphQL objects to ingest records, without a server."""

from typing import Any

from honcho_github.mapping import (
    MAX_CONTENT,
    NO_REASONING,
    Repo,
    Thread,
    ai_assisted,
    clean,
    comment_record,
    commit_record,
    instance_for,
    mentions,
    person,
    review_comment_record,
    review_record,
    thread_record,
    touched_dirs,
    utc,
    without_collapsed,
)

REPO = Repo(
    id="R_1",
    name_with_owner="acme/web",
    owner_id="O_1",
    url="https://github.com/acme/web",
)
PR = Thread(kind="pull_request", number=12, title="Add login", url="u", state="merged")
ISSUE = Thread(kind="issue", number=3, title="Crash", url="u", state="open")
ALEX = {"__typename": "User", "login": "alex", "databaseId": 11, "name": "Alex Doe"}
SAM = {"__typename": "User", "login": "sam", "databaseId": 22}
RABBIT = {"__typename": "Bot", "login": "coderabbitai", "databaseId": 99}
KNOWN = {"alex": person(ALEX), "sam": person(SAM)}


def resolve(login: str) -> dict[str, Any] | None:
    return KNOWN.get(login.casefold())


def _node(**fields: Any) -> dict[str, Any]:
    return {
        "id": "X_1",
        "createdAt": "2026-11-04T10:00:00Z",
        "lastEditedAt": None,
        "url": "https://github.com/acme/web/pull/12",
        "author": ALEX,
        "body": "text",
        **fields,
    }


def test_clean_removes_generated_text_from_a_persons_body() -> None:
    body = (
        "Fixes the login.\n\n"
        + "<!-- This is an auto-generated comment: release notes by coderabbit.ai -->\n"
        + "## Summary by CodeRabbit\n* New feature\n"
        + "<!-- end of auto-generated comment: release notes by coderabbit.ai -->\n\n"
        + "<!-- a hidden note -->\n"
        + "More detail.\n\n\n\n"
        + "Co-Authored-By: Claude <noreply@anthropic.com>\n"
        + "🤖 Generated with [Claude Code](https://claude.com/claude-code)\n"
    )

    assert clean(body) == "Fixes the login.\n\nMore detail."
    assert ai_assisted(body)
    assert not ai_assisted("Co-Authored-By: Sam <sam@example.com>")


def test_a_trailing_coderabbit_summary_without_markers_is_removed() -> None:
    assert clean("Body\n\n## Summary by CodeRabbit\n\n* stuff") == "Body"


def test_mentions_skip_code_emails_and_teams() -> None:
    text = (
        "@alex can you look? cc @Sam and @sam again.\n"
        + "Mail me at me@example.com, ping @acme/backend.\n"
        + "`@decorator` and\n```\n@inside_code\n```\n(@trailing-)"
    )

    assert mentions(text) == ["alex", "Sam"]


def test_a_bot_mention_is_never_a_person_with_the_same_name() -> None:
    assert mentions("Thanks @sam[bot] and @sam[other]") == ["sam[bot]"]


def test_person_uses_the_numeric_id_and_flags_bots() -> None:
    assert person(ALEX) == {
        "name": "alex",
        "handle": "alex",
        "aliases": ["Alex Doe"],
        "id": "11",
    }
    assert person(RABBIT) == {
        "name": "coderabbitai",
        "handle": "coderabbitai",
        "id": "99",
        "bot": True,
    }
    assert person({"__typename": "User", "login": "dependabot[bot]"}) == {
        "name": "dependabot[bot]",
        "handle": "dependabot[bot]",
        "bot": True,
    }
    assert person(None) is None  # a deleted account


def test_instance_is_the_host() -> None:
    assert instance_for("github.com") == "github-com"
    assert instance_for("API.github.com.") == "github-com"
    enterprise = instance_for("ghe.acme.test")
    assert enterprise.startswith("ghe-acme-test-")
    # Hosts that slug alike stay apart, and a long host still fits ingest.
    assert instance_for("ghe-acme.test") != enterprise
    assert len(instance_for("a" * 250 + ".example")) <= 128


def test_an_issue_or_pr_opens_with_its_title() -> None:
    record = thread_record(
        _node(id="I_1", body="It crashes when @sam logs in.", number=3),
        ISSUE,
        REPO,
        "github-com",
        resolve,
    )

    assert record["content"] == "Opened issue #3: Crash\nIt crashes when @sam logs in."
    assert record["container"] == {"id": "R_1", "name": "acme/web", "parent_id": "O_1"}
    assert record["participants"] == [{"name": "sam", "handle": "sam", "id": "22"}]
    assert record["metadata"] == {
        "repo": "acme/web",
        "kind": "issue",
        "url": "u",
        "issue": 3,
        "thread": "issue:3",
        "state": "open",
    }


def test_an_empty_approval_is_context_with_reasoning_off() -> None:
    approved = review_record(
        _node(state="APPROVED", body="", submittedAt="2026-11-04T11:00:00Z"),
        PR,
        REPO,
        "github-com",
        resolve,
    )
    assert approved is not None
    assert approved["content"] == "Review of PR #12: approved"
    assert approved["configuration"] == NO_REASONING
    assert approved["created_at"] == "2026-11-04T11:00:00Z"

    with_text = review_record(
        _node(state="CHANGES_REQUESTED", body="Needs a test."),
        PR,
        REPO,
        "github-com",
        resolve,
    )
    assert with_text is not None
    assert "configuration" not in with_text
    assert with_text["content"].endswith("changes requested\nNeeds a test.")

    for state in ("COMMENTED", "PENDING"):
        assert (
            review_record(_node(state=state, body=""), PR, REPO, "github-com", resolve)
            is None
        )


def test_a_review_reply_names_the_comment_it_answers() -> None:
    first = _node(id="C_1", author=SAM, body="Why?", path="src/a.py", line=4)
    reply = _node(
        id="C_2",
        body="Because.",
        path="src/a.py",
        line=9,  # where the line is now: it moves as the file changes
        originalLine=4,
        replyTo={"id": "C_1"},
    )

    record = review_comment_record(reply, PR, REPO, "github-com", resolve, first)

    assert record is not None
    assert record["content"] == "Review comment on PR #12, src/a.py:4:\nBecause."
    assert record["participants"] == [{"name": "sam", "handle": "sam", "id": "22"}]
    assert record["metadata"]["reply_to"] == "C_1"
    assert record["metadata"]["path"] == "src/a.py"


def test_a_bot_keeps_its_words_but_not_its_collapsed_machinery() -> None:
    body = (
        "**Actionable comments posted: 1**\n\n<details>\n<summary>Prompt</summary>\n"
        + "<details><summary>inner</summary>x</details>\nlots\n</details>\n\nDone."
    )
    assert without_collapsed(body) == "**Actionable comments posted: 1**\n\nDone."

    record = review_record(
        _node(state="COMMENTED", body=body, author=RABBIT),
        PR,
        REPO,
        "github-com",
        resolve,
    )

    assert record is not None
    assert "<details>" not in record["content"]
    assert record["author"] == {
        "name": "coderabbitai",
        "handle": "coderabbitai",
        "id": "99",
        "bot": True,
    }


def test_long_text_is_cut_at_the_ingest_limit() -> None:
    record = thread_record(
        _node(body="x" * (MAX_CONTENT + 10)), PR, REPO, "github-com", resolve
    )

    assert len(record["content"]) == MAX_CONTENT
    assert record["metadata"]["truncated"] is True


def _commit(**fields: Any) -> dict[str, Any]:
    return {
        "id": "C_kw",
        "oid": "0123456789abcdef",
        "url": "https://github.com/acme/web/commit/0123",
        "committedDate": "2026-11-04T12:00:00Z",
        "message": "feat: add login (#12)\n\nBody.\n\nCo-Authored-By: Claude <n@a.com>",
        "author": {"name": "Alex", "user": {"login": "alex", "databaseId": 11}},
        "associatedPullRequests": {
            "nodes": [
                {
                    "number": 12,
                    "files": {
                        "nodes": [{"path": "src/auth/a.py"}, {"path": "tests/t.py"}]
                    },
                }
            ]
        },
        **fields,
    }


def test_a_commit_is_what_shipped_with_what_it_touched() -> None:
    record = commit_record(
        _commit(),
        REPO,
        "github-com",
        resolve,
    )

    assert record is not None
    assert record["content"] == (
        "Commit to acme/web (PR #12):\nfeat: add login (#12)\n\nBody."
        + "\n\ntouched: src/auth/, tests/"
    )
    assert record["author"] == {"name": "alex", "handle": "alex", "id": "11"}
    assert record["metadata"]["ai_assisted"] is True
    assert record["metadata"]["sha"] == "0123456789ab"


def test_a_merge_commit_is_named_by_its_pull_request() -> None:
    record = commit_record(
        _commit(
            message="Merge pull request #12 from alex/login\n\nAdd login\n\nDetails.",
            associatedPullRequests={"nodes": []},
        ),
        REPO,
        "github-com",
        resolve,
    )

    assert record is not None
    assert (
        record["content"] == "Commit to acme/web (PR #12):\nAdd login (#12)\n\nDetails."
    )


def test_commits_without_a_person_or_meaning_are_skipped() -> None:
    def commit(**fields: Any) -> dict[str, Any] | None:
        return commit_record(_commit(**fields), REPO, "github-com", resolve)

    assert commit(message="Merge branch 'main' into login") is None
    assert commit(author={"name": "someone", "user": None}) is None
    bot = commit(author={"name": "dependabot[bot]", "user": None})
    assert bot is not None
    assert bot["author"] == {"name": "dependabot[bot]", "bot": True}


def test_touched_dirs_ranks_top_directories() -> None:
    paths = ["src/a/x.py", "src/a/y.py", "src/b/z.py", "README.md", "tests/t.py"]

    assert touched_dirs(paths) == ["src/a/", "README.md", "src/b/", "tests/"]


def test_a_childs_text_doesnt_change_when_its_pr_is_renamed() -> None:
    renamed = Thread(
        kind="pull_request", number=12, title="Renamed", url="u", state="open"
    )
    node = _node(id="C_9", body="Looks good.")

    before = comment_record(node, PR, REPO, "github-com", resolve)
    after = comment_record(node, renamed, REPO, "github-com", resolve)

    assert before is not None and after is not None
    assert before["content"] == after["content"] == "Comment on PR #12:\nLooks good."


def test_timestamps_are_utc() -> None:
    assert utc("2026-11-02T10:00:00+02:00") == "2026-11-02T08:00:00Z"
    assert utc("2026-11-02T10:00:00Z") == "2026-11-02T10:00:00Z"
