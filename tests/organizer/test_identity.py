"""The identity rules: who looks like one person, and what keeps two people apart."""

import datetime

from src.organizer.identity import (
    Account,
    Post,
    Suggestion,
    normalize_url,
    own_words,
    suggest,
    suggest_with,
)

T0 = datetime.datetime(2026, 11, 3, 9, 0, tzinfo=datetime.UTC)


def _at(minutes: int) -> datetime.datetime:
    return T0 + datetime.timedelta(minutes=minutes)


def _account(
    source: str,
    instance: str,
    key: str,
    names: tuple[str, ...],
    *,
    handle: str | None,
    email_digest: str | None,
    bot: bool,
    peer: str | None,
) -> Account:
    return Account(
        key=key, source=source, instance=instance, names=names, handle=handle,
        email_digest=email_digest, bot=bot, peer=peer or key,
    )  # fmt: skip


def discord(
    key: str,
    *names: str,
    handle: str | None = None,
    email_digest: str | None = None,
    bot: bool = False,
    peer: str | None = None,
) -> Account:
    return _account(
        "discord", "g1", f"d:{key}", names, handle=handle,
        email_digest=email_digest, bot=bot, peer=peer,
    )  # fmt: skip


def github(
    key: str,
    *names: str,
    handle: str | None = None,
    email_digest: str | None = None,
    bot: bool = False,
    peer: str | None = None,
) -> Account:
    return _account(
        "github", "github-com", f"g:{key}", names, handle=handle,
        email_digest=email_digest, bot=bot, peer=peer,
    )  # fmt: skip


def _pairs(
    found: list[Suggestion], confidence: str | None = None
) -> set[tuple[str, str]]:
    return {
        s.accounts for s in found if confidence is None or s.confidence == confidence
    }


def _pr(author: str, n: int, minutes: int) -> Post:
    return Post(
        author=author, at=_at(minutes), text=f"PR {n}",
        url=f"https://github.com/acme/web/pull/{n}",
    )  # fmt: skip


def _share(author: str, n: int, minutes: int, suffix: str = "") -> Post:
    return Post(
        author=author, at=_at(minutes),
        text=f"up for review: https://github.com/acme/web/pull/{n}{suffix}",
    )  # fmt: skip


def test_two_agreeing_names_are_high() -> None:
    found = suggest(
        [
            discord("1", "Marianne", handle="thrialectics"),
            github("1", "Marianne", handle="thrialectics"),
        ],
        [],
    )
    assert _pairs(found, "high") == {("d:1", "g:1")}


def test_a_shared_handle_alone_is_only_low() -> None:
    # One username can belong to two people on two platforms.
    found = suggest(
        [
            discord("1", "Alex", handle="anguyen4821"),
            github("1", "anguyen4821", handle="anguyen4821"),
        ],
        [],
    )
    assert _pairs(found, "high") == set()
    assert _pairs(found, "low") == {("d:1", "g:1")}


def test_a_name_a_third_account_holds_is_not_specific() -> None:
    accounts = [
        discord("1", "Alex Nguyen"),
        github("1", "Alex Nguyen", handle="ax9"),
        github("2", "Alex Nguyen", handle="ax10"),
    ]
    assert suggest(accounts, []) == []


def test_the_same_email_is_high_and_says_so() -> None:
    found = suggest(
        [
            Account("s:1", "slack", "t1", names=("Sam K.",), email_digest="e1"),
            github("1", "samk", handle="samk", email_digest="e1"),
        ],
        [],
    )
    (s,) = found
    assert s.confidence == "high"
    assert s.evidence[0].signal == "email"


def test_a_self_reference_naming_the_source_is_high() -> None:
    accounts = [
        discord("1", "Alex", handle="alex_d"),
        github("1", handle="anguyen4821"),
    ]
    for text in (
        "I opened the northstar PR on GitHub as anguyen4821.",
        "my github username is @anguyen4821",
        "I'm anguyen4821 on GitHub, ping me there",
    ):
        found = suggest(accounts, [Post("d:1", T0, text)])
        assert _pairs(found, "high") == {("d:1", "g:1")}, text


def test_mentions_quotes_and_code_are_not_self_references() -> None:
    accounts = [
        discord("1", "Alex", handle="alex_d"),
        github("1", handle="anguyen4821"),
    ]
    for text in (
        "I reviewed anguyen4821's PR on GitHub",
        "Alex is on GitHub as anguyen4821",
        "I'm not on GitHub as anguyen4821",
        "I never used anguyen4821 on GitHub, my GitHub is something else",
        "same as @anguyen4821 said",
        "> Opened the PR on GitHub as anguyen4821.",
        "run `git push  # on GitHub as anguyen4821`",
        "<system-reminder>The user is on GitHub as anguyen4821.</system-reminder>",
    ):
        found = suggest(accounts, [Post("d:1", T0, text)])
        assert _pairs(found, "high") == set(), text


def test_linking_your_own_work_right_away_backs_a_specific_name() -> None:
    accounts = [
        discord("1", "Dani", handle="danibalcells"),
        github("1", handle="danibalcells"),
    ]
    posts = [_pr("g:1", 1, 0), _share("d:1", 1, 30)]
    assert _pairs(suggest(accounts, posts), "high") == {("d:1", "g:1")}


def test_strong_shared_work_needs_a_rare_name_to_be_high() -> None:
    posts = [
        _pr("g:1", 1, 0), _share("d:1", 1, 10),
        _pr("g:1", 2, 100), _share("d:1", 2, 110),
    ]  # fmt: skip
    rare = [
        discord("1", "Aakash", handle="aakash0"),
        github("1", "Aakash K.", handle="akattelu"),
    ]
    assert _pairs(suggest(rare, posts), "high") == {("d:1", "g:1")}
    # "Alex" is common here: a colleague with your first name who posts your
    # pull requests is not you.
    common = [
        discord("1", "Alex", handle="alex0"),
        github("1", "Alex N.", handle="anx"),
        discord("2", "Alex Brown"),
        github("2", "Alex Lee"),
    ]
    found = suggest(common, posts)
    assert _pairs(found, "high") == set()
    assert ("d:1", "g:1") in _pairs(found, "low")


def test_sharing_someone_elses_work_is_never_enough() -> None:
    accounts = [
        discord("1", "Ben", handle="plasticben"),
        github("1", "Doria", handle="dr-frmr"),
    ]
    posts = [
        p
        for n in range(5)
        for p in (_pr("g:1", n, n * 60), _share("d:1", n, n * 60 + 5))
    ]
    found = suggest(accounts, posts)
    assert _pairs(found, "high") == set()


def test_reposting_one_record_counts_once_and_a_comment_is_its_author_s() -> None:
    accounts = [
        discord("1", "Dani", handle="danibalcells"),
        github("1", handle="danibalcells"),
        github("2", "Lee", handle="leewrites"),
    ]
    comment = Post(
        author="g:2", at=_at(5), text="looks good",
        url="https://github.com/acme/web/pull/1#issuecomment-9",
    )  # fmt: skip
    posts = [
        _pr("g:1", 1, 0),
        comment,
        _share("d:1", 1, 10),
        _share("d:1", 1, 20),
        # A link to Lee's comment is Lee's work, not the pull request author's.
        _share("d:1", 1, 30, "#issuecomment-9"),
    ]
    found = {s.accounts: s for s in suggest(accounts, posts)}
    work = [e for e in found[("d:1", "g:1")].evidence if e.signal == "shared_work"]
    assert work and work[0].level == "weak" and "linked 1 " in work[0].detail


def test_two_accounts_in_one_instance_need_an_email_or_a_self_reference() -> None:
    old = discord("1", "Riley Okafor", handle="rokafor1")
    new = discord("2", "Riley Okafor", handle="rokafor2")
    assert suggest([old, new], []) == []
    with_email = [
        discord("1", "Riley", handle="rokafor1", email_digest="e"),
        discord("2", "Riley Okafor", handle="rokafor2", email_digest="e"),
    ]
    assert _pairs(suggest(with_email, []), "high") == {("d:1", "d:2")}


def test_no_transitive_merge_of_two_accounts_of_one_instance() -> None:
    # Both Discord accounts give the same GitHub account as theirs: accepting
    # both would make two Discord people one, with nothing between them.
    accounts = [
        discord("1", "Alex", handle="alex1"),
        discord("2", "Sam", handle="sam2"),
        github("1", handle="anguyen4821"),
    ]
    posts = [
        Post("d:1", T0, "I'm anguyen4821 on GitHub"),
        Post("d:2", T0, "I'm anguyen4821 on GitHub"),
    ]
    found = suggest(accounts, posts)
    assert _pairs(found, "high") == set()
    assert all(any(e.signal == "conflict" for e in s.evidence) for s in found)


def test_accounts_that_talk_to_each_other_are_not_joined() -> None:
    accounts = [
        discord("1", "Marianne", handle="thrialectics"),
        github("1", "Marianne", handle="thrialectics"),
        discord("2", "Marianne T", handle="mt"),
        github("2", handle="mt"),
    ]
    # g:1 and g:2 are high pairs with d:1 / d:2 ... and d:1 talks to d:2.
    posts = [Post("d:1", T0, "hi", participants=("d:2",))]
    found = {s.accounts: s for s in suggest(accounts, posts)}
    assert found[("d:1", "g:1")].confidence == "high"


def test_explicit_peer_id_equal_to_a_username_case_and_all() -> None:
    explicit = Account("explicit:alex-nguyen-4f2a", "explicit", "", kind="explicit",
                       handle="alex-nguyen-4f2a", peer="alex-nguyen-4f2a")  # fmt: skip
    same = discord("1", "Alex", handle="alex-nguyen-4f2a")
    other_case = discord("2", "Alex", handle="Alex-Nguyen-4f2a")
    found = suggest([explicit, same], [])
    assert _pairs(found, "high") == {("d:1", "explicit:alex-nguyen-4f2a")}
    found = suggest([explicit, other_case], [])
    assert _pairs(found, "high") == set()


def test_bots_source_peers_and_linked_accounts_are_left_out() -> None:
    accounts = [
        discord("1", "Marianne", handle="thrialectics", bot=True),
        github("1", "Marianne", handle="thrialectics"),
        Account("d:src", "discord", "g1", kind="source", names=("thrialectics",)),
        # Already one person: nothing to suggest.
        discord("3", "Dani", handle="danibalcells", peer="p"),
        github("3", "Dani", handle="danibalcells", peer="p"),
    ]
    assert suggest(accounts, []) == []


def test_helpers() -> None:
    assert (
        normalize_url("HTTPS://GitHub.com/acme/web/pull/12/?x=1#issuecomment-3")
        == "https://github.com/acme/web/pull/12#issuecomment-3"
    )
    assert normalize_url("ftp://example.com/x") is None
    assert own_words("> quoted\nmine\n```\ncode\n```") == "mine\n "


def test_a_link_to_a_comment_that_was_not_imported_is_no_ones() -> None:
    # The comment isn't in the import: its link must not count for the PR author.
    accounts = [
        discord("1", "Dani", handle="danibalcells"),
        github("1", handle="danibalcells"),
    ]
    posts = [
        _pr("g:1", 1, 0),
        _share("d:1", 1, 10, "#issuecomment-9"),
        _share("d:1", 1, 20, "#issuecomment-10"),
    ]
    found = {s.accounts: s for s in suggest(accounts, posts)}
    assert all(e.signal != "shared_work" for e in found[("d:1", "g:1")].evidence)


def test_a_retained_high_pair_joins_the_conflict_check() -> None:
    # An earlier refresh left d:1 + g:1 open and high; now d:2 claims g:1 too.
    accounts = [
        discord("1", "Alex", handle="alex1"),
        discord("2", "Sam", handle="sam2"),
        github("1", handle="anguyen4821"),
    ]
    posts = [Post("d:2", T0, "I'm anguyen4821 on GitHub")]
    found, demoted = suggest_with(accounts, posts, retained=[("d:1", "g:1")])
    assert _pairs(found, "high") == set()
    assert set(demoted) == {("d:1", "g:1")}
    # Without the retained pair, the new one alone is fine.
    found, demoted = suggest_with(accounts, posts, retained=[])
    assert _pairs(found, "high") == {("d:2", "g:1")} and demoted == {}
