"""The benchmark itself: fixtures are valid ingest input and give nothing away,
the split holds, the scorer counts what it says, and the baseline is what the
README says it is."""

import json
from collections import Counter
from itertools import pairwise
from math import isclose
from pathlib import Path
from typing import Any

import pytest

from src import schemas
from src.crud.ingest import peer_key_for
from tests.bench.auto_workspaces import baseline, fixtures
from tests.bench.auto_workspaces.model import (
    BenchRecord,
    Dataset,
    Profile,
    account_key,
    record_accounts,
)
from tests.bench.auto_workspaces.run import main
from tests.bench.auto_workspaces.scoring import (
    count_tokens,
    score,
    score_bots,
    score_grouping,
    score_identity,
)
from tests.bench.auto_workspaces.split import (
    HELDOUT_FRACTION,
    missing_cases,
    select,
    side,
)


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return fixtures.generate()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def test_fixtures_are_deterministic(dataset: Dataset) -> None:
    again = fixtures.generate()
    assert [r.record for r in again.records] == [r.record for r in dataset.records]
    assert again.truth.person_of == dataset.truth.person_of


def test_every_record_is_a_valid_ingest_record(dataset: Dataset) -> None:
    for r in dataset.records:
        schemas.IngestRecordCreate.model_validate(r.record)


def test_every_case_has_records_and_operations(dataset: Dataset) -> None:
    assert set(Counter(r.case for r in dataset.records)) == set(fixtures.CASES)
    ops = Counter(r.op for r in dataset.records)
    assert ops["edit"] > 0 and ops["delete"] > 0


def test_every_account_and_record_has_an_answer(dataset: Dataset) -> None:
    truth = dataset.truth
    for r in dataset.records:
        assert truth.groups_of[r.key]
        for account in record_accounts(r.record):
            assert account in truth.person_of, account
            assert truth.unit_of_account[account] == r.unit
            assert account in dataset.profiles
    assert truth.bots <= set(truth.person_of)


def test_what_an_organizer_sees_gives_no_answers_away(dataset: Dataset) -> None:
    """No case or unit label, and no id shared or derived across accounts."""
    blind = dataset.blind()
    assert {(r.case, r.unit) for r in blind.records} == {("", "")}
    assert blind.truth.person_of == {}
    seen = json.dumps([r.record for r in blind.records]) + json.dumps(
        [r.key for r in blind.records]
    )
    for label in {r.unit for r in dataset.records} | set(fixtures.CASES):
        assert label not in seen, label
    # Every external id is its own opaque number: one person's accounts on
    # different sources share no id or id stem.
    ids = [
        a.split("/id:")[1]
        for a in dataset.truth.person_of
        if "/id:" in a and "@" not in a
    ]
    assert len(ids) == len(set(ids))
    assert all(i.isdigit() and len(i) == 18 for i in ids)


def test_the_hard_cases_are_hard(dataset: Dataset) -> None:
    """The traps exist: one display name for different people, one username
    for different people, and people whose accounts share no handle."""
    truth = dataset.truth
    names: dict[tuple[str, str], set[str]] = {}
    handles: dict[tuple[str, str], set[str]] = {}
    for account, person in truth.person_of.items():
        profile = dataset.profiles[account]
        unit = truth.unit_of_account[account]
        if profile.display_name:
            names.setdefault((unit, profile.display_name), set()).add(person)
        if profile.username:
            handles.setdefault((unit, profile.username), set()).add(person)
    assert any(len(people) > 1 for people in names.values())
    assert any(len(people) > 1 for people in handles.values())

    prediction = baseline.organize(dataset.blind())
    assert score_identity(truth.person_of, prediction.person_of).false_splits > 0


def test_coding_agents_are_not_people(dataset: Dataset) -> None:
    agents = {
        a
        for a, p in dataset.truth.person_of.items()
        if p.startswith("plugin_turns#") and p.endswith("/agent")
    }
    assert agents and agents <= dataset.truth.bots


def test_provisional_keys_match_ingest_and_cannot_collide() -> None:
    """A container and name can't spell another pair's key, as in ingest."""
    one = account_key("s", "i", name="c", container="a:b")
    two = account_key("s", "i", name="b:c", container="a")
    assert one != two
    record = schemas.IngestRecordCreate.model_validate(
        {
            "source": "s",
            "source_instance": "i",
            "created_at": "2026-11-04T20:47:00Z",
            "content": "x",
            "container": {"id": "a:b"},
            "author": {"name": "C"},
        }
    )
    assert record.author is not None
    ingest_key = peer_key_for(record, record.author)
    assert one == f"s/i/name:{ingest_key.value}"
    assert record_accounts(record.model_dump(mode="json")) == [one]


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


def test_split_keeps_units_whole_and_every_case_on_both_sides(
    dataset: Dataset,
) -> None:
    dev, heldout = select(dataset, "dev"), select(dataset, "heldout")
    assert {r.unit for r in dev.records}.isdisjoint({r.unit for r in heldout.records})
    assert len(dev.records) + len(heldout.records) == len(dataset.records)
    assert not missing_cases(dataset, dev) and not missing_cases(dataset, heldout)
    assert set(dev.truth.person_of).isdisjoint(heldout.truth.person_of)

    units = {r.unit for r in dataset.records}
    held = sum(1 for u in units if side(u) == "heldout") / len(units)
    assert abs(held - HELDOUT_FRACTION) < 0.1


def test_a_split_missing_a_case_is_refused() -> None:
    small = fixtures.generate(instances=2)
    assert missing_cases(small, select(small, "heldout"))
    with pytest.raises(SystemExit):
        main(["--instances", "2", "--split", "heldout"])


def test_a_unit_stays_on_its_side() -> None:
    assert side("same_name#3") == side("same_name#3")
    assert {side(f"unit#{n}") for n in range(50)} == {"dev", "heldout"}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_TRUTH = {"a1": "alice", "a2": "alice", "a3": "alice", "b1": "bob", "c1": "carol"}


def test_identity_perfect() -> None:
    result = score_identity(_TRUTH, _TRUTH)
    assert (result.precision, result.recall, result.f05) == (1.0, 1.0, 1.0)
    assert (result.false_merges, result.false_splits, result.true_pairs) == (0, 0, 3)


def test_identity_counts_false_merges_and_names_them() -> None:
    merged = dict.fromkeys(_TRUTH, "everyone")
    result = score_identity(_TRUTH, merged)
    # 10 predicted pairs, 3 of them right.
    assert (result.false_merges, result.false_splits) == (7, 0)
    assert isclose(result.precision, 3 / 10)
    assert result.recall == 1.0
    assert ("a1", "b1") in result.merged_wrongly


def test_identity_counts_false_splits_and_weights_merges_worse() -> None:
    split_apart = score_identity(_TRUTH, {})
    assert (split_apart.false_merges, split_apart.false_splits) == (0, 3)
    assert split_apart.precision == 1.0 and split_apart.recall == 0.0

    # Splitting off one of alice's accounts loses 2 of her 3 pairs (precision 1,
    # recall 1/3); merging bob into alice keeps all of hers but adds 3 wrong
    # pairs (precision 1/2, recall 1). F0.5 ranks the split higher.
    one_split = score_identity(_TRUTH, {**_TRUTH, "a3": "alone"})
    one_merge = score_identity(_TRUTH, {**_TRUTH, "b1": "alice"})
    assert one_split.precision == 1.0 and isclose(one_split.recall, 1 / 3)
    assert (one_merge.precision, one_merge.recall) == (0.5, 1.0)
    assert isclose(one_split.f05, 0.714, abs_tol=1e-3)
    assert isclose(one_merge.f05, 0.556, abs_tol=1e-3)


def test_identity_leaves_excluded_accounts_out() -> None:
    truth = {**_TRUTH, "bot": "bot"}
    result = score_identity(truth, dict.fromkeys(truth, "x"), exclude={"bot"})
    assert result.false_merges == 7


def test_a_merge_across_cases_counts_against_both() -> None:
    """Focusing on one case still counts its accounts' merges with outsiders."""
    truth = {"x1": "x", "x2": "x", "y1": "y"}
    predicted = {"x1": "p", "x2": "p", "y1": "p"}
    for focus in ({"x1", "x2"}, {"y1"}):
        result = score_identity(truth, predicted, focus=focus)
        assert result.false_merges == 2, focus
        assert result.precision < 1


def test_grouping_perfect_and_one_big_group() -> None:
    truth = {
        "r1": frozenset({"acme"}),
        "r2": frozenset({"acme"}),
        "r3": frozenset({"globex"}),
        "r4": frozenset({"acme", "globex"}),
    }
    assert score_grouping(truth, truth).f1 == 1.0
    # One group for the whole channel matches acme (F1 6/7); globex is left
    # unmatched.
    lumped = score_grouping(truth, dict.fromkeys(truth, frozenset({"channel"})))
    assert isclose(lumped.recall, (3 * 6 / 7) / 5)
    assert isclose(lumped.precision, 6 / 7)
    assert score_grouping(truth, {}).f1 == 0.0


def test_copies_of_a_group_earn_nothing() -> None:
    truth = {f"r{n}": frozenset({"a" if n < 5 else "b"}) for n in range(10)}
    honest = score_grouping(truth, truth).f1
    copies = {
        r: frozenset({*(f"{g}{c}" for c in range(100) for g in groups), "all"})
        for r, groups in truth.items()
    }
    padded = score_grouping(truth, copies)
    assert padded.precision < 0.05
    assert padded.f1 < honest / 2


def test_a_group_that_takes_in_another_case_loses_precision() -> None:
    case_truth = {"r1": frozenset({"a"}), "r2": frozenset({"a"})}
    clean = score_grouping(case_truth, {"r1": frozenset({"p"}), "r2": frozenset({"p"})})
    spill = score_grouping(
        case_truth, {r: frozenset({"p"}) for r in ("r1", "r2", "other1", "other2")}
    )
    # The spilled group is twice the true one: matched at F1 2/3.
    assert clean.f1 == 1.0
    assert isclose(spill.precision, 2 / 3) and isclose(spill.f1, 2 / 3)


def test_bots() -> None:
    result = score_bots({"x", "y"}, {"x", "z"})
    assert (result.precision, result.recall) == (0.5, 0.5)


def test_score_reports_every_case(dataset: Dataset) -> None:
    report = score(dataset, baseline.organize(dataset.blind()))
    assert set(report.identity_f05_by_case) == set(fixtures.CASES)
    assert report.ingested_tokens > 0
    assert report.organizer_tokens_per_million == 0


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def test_the_baseline_merges_only_where_handles_collide(dataset: Dataset) -> None:
    report = score(dataset, baseline.organize(dataset.blind()))
    merges = {c: n for c, n in report.false_merges_by_case.items() if n}
    assert set(merges) == {"shared_handle"}
    assert 0 < report.identity.recall < 1


def test_the_baseline_keeps_explicit_peers_case_sensitive() -> None:
    small = fixtures.generate(instances=1).subset(["shared_handle#0"])
    prediction = baseline.organize(small.blind())
    explicit = [a for a in small.truth.person_of if a.startswith("explicit:")]
    assert len(explicit) == 2
    assert prediction.person_of[explicit[0]] != prediction.person_of[explicit[1]]


def test_the_baseline_reads_no_labels_or_truth(dataset: Dataset) -> None:
    assert baseline.organize(dataset.blind()) == baseline.organize(dataset)


# ---------------------------------------------------------------------------
# Round trip and the runner
# ---------------------------------------------------------------------------


def test_a_written_dataset_reads_back_the_same(
    dataset: Dataset, tmp_path: Path
) -> None:
    dataset.write(tmp_path)
    again = Dataset.read(tmp_path)
    assert again.records == dataset.records
    assert again.profiles == dataset.profiles
    assert again.truth == dataset.truth


def test_the_runner_scores_both_splits(capsys: pytest.CaptureFixture[str]) -> None:
    for split in ("dev", "heldout"):
        assert main(["--split", split]) == 0
        out = capsys.readouterr().out
        assert "identity  F0.5" in out and "shared_handle" in out


# ---------------------------------------------------------------------------
# Shortcuts an organizer must not have
# ---------------------------------------------------------------------------


def test_record_order_says_nothing_about_which_records_go_together(
    dataset: Dataset,
) -> None:
    """Instances interleave in time, so neighbours in the blind feed are
    usually from different units; keys are opaque."""
    unit_of = {r.key: r.unit for r in dataset.records}
    blind = dataset.blind().records
    same = sum(1 for a, b in pairwise(blind) if unit_of[a.key] == unit_of[b.key])
    assert same / (len(blind) - 1) < 0.2
    assert [r.key for r in blind] != [r.key for r in dataset.records]
    assert all(r.key.isdigit() for r in blind)

    # Within a case, the order in which its accounts post (numbered by first
    # appearance) differs per instance: no fixed offset says who is who.
    by_unit: dict[str, list[str]] = {}
    for r in blind:
        by_unit.setdefault(unit_of[r.key], []).append(record_accounts(r.record)[0])
    patterns: dict[str, Counter[tuple[int, ...]]] = {}
    for unit, authors in by_unit.items():
        number = {a: n for n, a in enumerate(dict.fromkeys(authors))}
        patterns.setdefault(unit.split("#")[0], Counter())[
            tuple(number[a] for a in authors)
        ] += 1
    for case, seen in patterns.items():
        # One account has no order to vary; an account replaced by a new one
        # really does stop before the new one starts.
        if case in ("edits_deletes", "account_change"):
            continue
        _, commonest = seen.most_common(1)[0]
        assert commonest / seen.total() < 0.5, case


def test_grouping_doesnt_depend_on_group_names() -> None:
    truth = {
        "r0": frozenset({"A"}),
        "r1": frozenset({"A"}),
        "r2": frozenset({"B"}),
        "r3": frozenset({"B"}),
    }

    def predicted(small: str, big: str) -> dict[str, frozenset[str]]:
        return {
            "r0": frozenset({small, big}),
            "r1": frozenset({big}),
            "r2": frozenset({big}),
            "r3": frozenset({big}),
        }

    one = score_grouping(truth, predicted("a", "z"))
    other = score_grouping(truth, predicted("z", "a"))
    assert (one.precision, one.recall) == (other.precision, other.recall)


def test_a_correct_scope_across_two_cases_scores_perfectly_in_each() -> None:
    truth = {"r1": frozenset({"g"}), "r2": frozenset({"g"})}
    for focus in ({"r1"}, {"r2"}):
        assert score_grouping(truth, truth, focus=focus).f1 == 1.0


def test_only_the_real_source_peer_is_taken_for_a_bot() -> None:
    record = {
        "source": "s",
        "source_instance": "i",
        "source_id": "1",
        "created_at": "2026-11-04T20:47:00Z",
        "content": "hi",
        "author": {"id": "users/source", "name": "A person"},
    }
    human = Dataset(records=[BenchRecord(key="1", case="", unit="", record=record)])
    assert baseline.organize(human).bots == set()


def test_profile_order_says_nothing_about_who_is_who(dataset: Dataset) -> None:
    profiles = list(dataset.blind().profiles)
    assert profiles == sorted(profiles)
    person = dataset.truth.person_of
    same = sum(1 for a, b in pairwise(profiles) if person[a] == person[b])
    assert same / (len(profiles) - 1) < 0.05


def test_an_edit_or_delete_comes_after_its_original(dataset: Dataset) -> None:
    position: dict[tuple[str, str, str, str], int] = {}
    for n, r in enumerate(dataset.blind().records):
        rec = r.record
        identity = (rec["source"], rec["source_instance"], rec["source_id"], r.op)
        position[identity] = n
    for (source, instance, source_id, op), n in position.items():
        if op != "create":
            assert n > position[(source, instance, source_id, "create")]


def test_tied_grouping_assignments_dont_depend_on_names() -> None:
    truth = {
        "r0": frozenset({"A", "B"}),
        "r1": frozenset({"A"}),
        "r2": frozenset({"B"}),
        "r3": frozenset({"B"}),
    }

    def predicted(x: str, y: str) -> dict[str, frozenset[str]]:
        return {
            "r0": frozenset({x}),
            "r1": frozenset({y}),
            "r2": frozenset({x, y}),
            "r3": frozenset({y}),
        }

    one, other = (
        score_grouping(truth, predicted("a", "z")),
        score_grouping(truth, predicted("z", "a")),
    )
    assert (one.precision, one.recall) == (other.precision, other.recall)


def test_a_left_out_account_never_shares_a_predicted_label() -> None:
    """Whatever an organizer names its one person, the left-out account stays
    alone: no label merges it."""
    truth = {"a": "x", "b": "y"}
    for label in ("group", "__alone__:b", "alone", "b"):
        result = score_identity(truth, {"a": label})
        assert (result.false_merges, result.f05) == (0, 1.0), label


def test_tokens_are_counted_as_stored() -> None:
    assert count_tokens("hi\x00 there") == count_tokens("hi there")


@pytest.mark.parametrize(
    ("created", "revised"),
    [
        ("2026-11-01T10:00:00Z", "2026-11-01T10:00:00.001Z"),
        ("2026-11-01T10:00:00+02:00", "2026-11-01T09:00:00Z"),
    ],
)
def test_an_edit_follows_its_original_however_times_are_written(
    created: str, revised: str
) -> None:
    base = {
        "source": "s",
        "source_instance": "i",
        "source_id": "1",
        "created_at": created,
        "content": "x",
        "author": {"id": "a", "name": "A"},
    }
    original = BenchRecord(key="9", case="", unit="", record=base)
    edit = BenchRecord(
        key="1", case="", unit="", record={**base, "revised_at": revised}, op="edit"
    )
    feed = Dataset(records=[edit, original]).blind().records
    assert [r.op for r in feed] == ["create", "edit"]


def test_groups_invented_where_there_are_none_score_zero() -> None:
    truth: dict[str, frozenset[str]] = {"r1": frozenset()}
    assert score_grouping(truth, {"r1": frozenset({"made-up"})}).precision == 0.0
    assert score_grouping(truth, {"r1": frozenset()}).f1 == 1.0


def test_an_explicit_peer_matches_a_username_exactly() -> None:
    def record(n: int, **who: Any) -> BenchRecord:
        return BenchRecord(
            key=str(n),
            case="",
            unit="",
            record={
                "source": "s",
                "source_instance": "i",
                "source_id": str(n),
                "created_at": "2026-11-01T10:00:00Z",
                "content": "x",
                **who,
            },
        )

    dataset = Dataset(
        records=[
            record(1, author={"id": "u1", "name": "A"}),
            record(2, peer_id="Alice"),
            record(3, peer_id="alice"),
        ]
    )
    account = account_key("s", "i", external_id="u1")
    dataset.profiles[account] = Profile(account=account, username="Alice")
    person = baseline.organize(dataset).person_of
    assert person[account] == person["explicit:Alice"]
    assert person[account] != person["explicit:alice"]
