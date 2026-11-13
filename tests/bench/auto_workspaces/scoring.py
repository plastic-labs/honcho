"""Scoring an organizer's prediction against the truth.

Identity is scored over pairs of accounts:
- a **false merge** puts two different people's accounts together (the costly
  mistake: one person's messages and memory become another's);
- a **false split** leaves one person's accounts apart (the conservative
  default: two partial memories).
Precision counts false merges and recall counts false splits. The headline is
F0.5, which weights precision twice as heavily as recall.

Grouping matches true and predicted groups one to one, by F1 of their record
overlap, with the assignment that maximizes the size-weighted total (so copies
of a group earn nothing, and renaming groups changes nothing); recall weights
the matched F1s by true group size, precision by predicted group size, and an
unmatched group counts as 0. A record can belong to several groups.

Per-case scores count every pair, and every predicted group, that touches the
case, including what reaches outside it: merging an account of one case with
an account of another is a false merge in both.

Bots are scored as a plain classification. Cost is organizer tokens per million
ingested tokens, and review load is suggestions per thousand records.
"""

import math
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import cast

import numpy as np
import numpy.typing as npt
import tiktoken
from scipy.optimize import (
    linear_sum_assignment,  # pyright: ignore[reportUnknownVariableType]
)

from src.utils.sanitization import strip_nul
from tests.bench.auto_workspaces.model import (
    AccountKey,
    Dataset,
    Prediction,
    RecordKey,
    record_accounts,
)

# F-beta for identity: beta 0.5 weights precision (false merges) 2x recall.
IDENTITY_BETA = 0.5


def _pairs(n: int) -> int:
    return n * (n - 1) // 2


def _f(precision: float, recall: float, beta: float = 1.0) -> float:
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


@dataclass
class IdentityScore:
    precision: float
    recall: float
    f05: float
    true_pairs: int
    false_merges: int
    false_splits: int
    # The account pairs wrongly merged, for reading what went wrong.
    merged_wrongly: list[tuple[AccountKey, AccountKey]] = field(default_factory=list)


def _pair_counts(
    accounts: Collection[AccountKey],
    truth: Mapping[AccountKey, str],
    pred: Mapping[AccountKey, tuple[str, str]],
) -> tuple[int, int, int]:
    """(pairs truly together, pairs predicted together, pairs both) among accounts."""
    true_same = sum(_pairs(n) for n in Counter(truth[a] for a in accounts).values())
    pred_same = sum(_pairs(n) for n in Counter(pred[a] for a in accounts).values())
    both = sum(
        _pairs(n) for n in Counter((truth[a], pred[a]) for a in accounts).values()
    )
    return true_same, pred_same, both


def score_identity(
    truth: Mapping[AccountKey, str],
    predicted: Mapping[AccountKey, str],
    *,
    exclude: Iterable[AccountKey] = (),
    focus: Collection[AccountKey] | None = None,
    examples: int = 20,
) -> IdentityScore:
    """Pairwise precision and recall of person assignment.

    Accounts the prediction leaves out are their own person. Excluded accounts
    (bots) take no part. With `focus`, only pairs with at least one account in
    it count.
    """
    skip = set(exclude)
    accounts = [a for a in truth if a not in skip]
    # An account the prediction leaves out is alone: a label no predicted
    # string can equal.
    pred: dict[AccountKey, tuple[str, str]] = {
        a: ("person", predicted[a]) if a in predicted else ("alone", a)
        for a in accounts
    }
    true_same, pred_same, both = _pair_counts(accounts, truth, pred)
    if focus is not None:
        inside = set(focus)
        rest = [a for a in accounts if a not in inside]
        out_true, out_pred, out_both = _pair_counts(rest, truth, pred)
        true_same, pred_same, both = (
            true_same - out_true,
            pred_same - out_pred,
            both - out_both,
        )
    false_merges = pred_same - both
    false_splits = true_same - both
    precision = both / pred_same if pred_same else 1.0
    recall = both / true_same if true_same else 1.0

    wrong: list[tuple[AccountKey, AccountKey]] = []
    if false_merges:
        members: dict[tuple[str, str], list[AccountKey]] = defaultdict(list)
        for a in accounts:
            members[pred[a]].append(a)
        for group in members.values():
            for x, y in combinations(sorted(group), 2):
                if focus is not None and x not in focus and y not in focus:
                    continue
                if truth[x] != truth[y] and len(wrong) < examples:
                    wrong.append((x, y))
    return IdentityScore(
        precision=precision,
        recall=recall,
        f05=_f(precision, recall, IDENTITY_BETA),
        true_pairs=true_same,
        false_merges=false_merges,
        false_splits=false_splits,
        merged_wrongly=wrong,
    )


@dataclass
class GroupingScore:
    precision: float
    recall: float
    f1: float
    true_groups: int
    predicted_groups: int


def _members(
    groups_of: Mapping[RecordKey, frozenset[str]],
) -> dict[str, set[RecordKey]]:
    members: dict[str, set[RecordKey]] = defaultdict(set)
    for record, groups in groups_of.items():
        for group in groups:
            members[group].add(record)
    return members


def _weighted(
    groups: Mapping[str, set[RecordKey]], match: Mapping[str, float]
) -> float:
    total = sum(len(m) for m in groups.values())
    if not total:
        return 0.0
    return sum(match.get(g, 0.0) * len(m) for g, m in groups.items()) / total


def _max_assignment(weight: npt.NDArray[np.float64]) -> list[tuple[int, int]]:
    """Row/column pairs of the maximum-weight one-to-one assignment."""
    rows, cols = cast(
        tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]],
        linear_sum_assignment(weight, maximize=True),
    )
    return [(int(i), int(j)) for i, j in zip(rows, cols, strict=True)]


def _assign(
    pairs: dict[tuple[str, str], float],
    true_members: Mapping[str, set[RecordKey]],
    pred_members: Mapping[str, set[RecordKey]],
) -> tuple[dict[str, float], dict[str, float]]:
    """One-to-one assignment of true to predicted groups maximizing total weight.

    A pair's weight is its F1 times the two groups' sizes, so the assignment
    maximizes exactly the sum the size-weighted precision and recall add up.
    It is solved per connected component of the overlap graph, with groups
    ordered by their members rather than their names, so even a tie between
    assignments resolves the same way whatever the groups are called.
    """

    def canonical(node: str) -> tuple[str, ...]:
        members = true_members if node.startswith("t:") else pred_members
        return (node[:2], *sorted(members[node[2:]]))

    neighbours: dict[str, set[str]] = defaultdict(set)
    for t, p in pairs:
        neighbours["t:" + t].add("p:" + p)
        neighbours["p:" + p].add("t:" + t)
    matched_t: dict[str, float] = {}
    matched_p: dict[str, float] = {}
    seen: set[str] = set()
    for start in sorted(neighbours, key=canonical):
        if start in seen:
            continue
        component: list[str] = []
        stack = [start]
        seen.add(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for nxt in neighbours[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        ts = [n[2:] for n in sorted(component, key=canonical) if n.startswith("t:")]
        ps = [n[2:] for n in sorted(component, key=canonical) if n.startswith("p:")]
        weight = np.zeros((len(ts), len(ps)))
        for i, t in enumerate(ts):
            for j, p in enumerate(ps):
                f1 = pairs.get((t, p), 0.0)
                weight[i, j] = f1 * (len(true_members[t]) + len(pred_members[p]))
        for i, j in _max_assignment(weight):
            f1 = pairs.get((ts[i], ps[j]), 0.0)
            if f1 > 0:
                matched_t[ts[i]] = matched_p[ps[j]] = f1
    return matched_t, matched_p


def score_grouping(
    truth: Mapping[RecordKey, frozenset[str]],
    predicted: Mapping[RecordKey, frozenset[str]],
    *,
    focus: Collection[RecordKey] | None = None,
) -> GroupingScore:
    """One-to-one best-match F1 between true and predicted groups.

    Groups are compared at their full size. With `focus`, only true and
    predicted groups holding a focus record take part, so a predicted group
    that also takes in records from elsewhere is penalized, and a true group
    that spans elsewhere is still matched whole.
    """
    true_all = _members(truth)
    pred_all = _members(predicted)
    if focus is not None:
        inside = set(focus)
        true_members = {g: m for g, m in true_all.items() if m & inside}
        pred_members = {g: m for g, m in pred_all.items() if m & inside}
    else:
        true_members, pred_members = true_all, pred_all
    if not true_members:
        # Nothing to find: perfect only if nothing was invented either.
        precision = 0.0 if pred_members else 1.0
        return GroupingScore(precision, 1.0, _f(precision, 1.0), 0, len(pred_members))
    pairs: dict[tuple[str, str], float] = {}
    for t, records in true_members.items():
        overlaps: Counter[str] = Counter()
        for record in records:
            for p in predicted.get(record, ()):
                if p in pred_members:
                    overlaps[p] += 1
        for p, overlap in overlaps.items():
            pairs[(t, p)] = _f(overlap / len(pred_members[p]), overlap / len(records))
    matched_t, matched_p = _assign(pairs, true_members, pred_members)
    recall = _weighted(true_members, matched_t)
    precision = _weighted(pred_members, matched_p)
    return GroupingScore(
        precision=precision,
        recall=recall,
        f1=_f(precision, recall),
        true_groups=len(true_members),
        predicted_groups=len(pred_members),
    )


@dataclass
class BotScore:
    precision: float
    recall: float
    true_bots: int
    predicted_bots: int


def score_bots(truth: set[AccountKey], predicted: set[AccountKey]) -> BotScore:
    hit = len(truth & predicted)
    return BotScore(
        precision=hit / len(predicted) if predicted else 1.0,
        recall=hit / len(truth) if truth else 1.0,
        true_bots=len(truth),
        predicted_bots=len(predicted),
    )


_ENCODING = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    """Tokens in a message's content as stored: NUL stripped, as ingest does."""
    return len(_ENCODING.encode(strip_nul(text), disallowed_special=()))


def ingested_tokens(dataset: Dataset) -> int:
    """Tokens of every created record's content."""
    return sum(
        count_tokens(r.record["content"]) for r in dataset.records if r.op == "create"
    )


@dataclass
class Report:
    records: int
    accounts: int
    ingested_tokens: int
    identity: IdentityScore
    grouping: GroupingScore
    bots: BotScore
    organizer_tokens_per_million: float
    suggestions_per_thousand: float
    # Per case, to see where an organizer gains or loses.
    identity_f05_by_case: dict[str, float]
    false_merges_by_case: dict[str, int]
    grouping_f1_by_case: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def score(dataset: Dataset, prediction: Prediction) -> Report:
    """Score a prediction on a dataset, overall and per case."""
    truth = dataset.truth
    # Grouping is scored on created records; edits and deletes change content,
    # not where a record belongs.
    created = {r.key for r in dataset.records if r.op == "create"}
    groups_truth = {k: v for k, v in truth.groups_of.items() if k in created}
    groups_pred = {k: v for k, v in prediction.groups_of.items() if k in created}
    case_accounts: dict[str, set[AccountKey]] = defaultdict(set)
    case_records: dict[str, set[RecordKey]] = defaultdict(set)
    for r in dataset.records:
        if r.key in created:
            case_records[r.case].add(r.key)
        case_accounts[r.case].update(record_accounts(r.record))

    by_case = {
        case: score_identity(
            truth.person_of, prediction.person_of, exclude=truth.bots, focus=accounts
        )
        for case, accounts in sorted(case_accounts.items())
    }
    by_case_grouping = {
        case: score_grouping(groups_truth, groups_pred, focus=keys).f1
        for case, keys in sorted(case_records.items())
    }
    tokens = ingested_tokens(dataset)
    return Report(
        records=len(created),
        accounts=len(truth.person_of),
        ingested_tokens=tokens,
        identity=score_identity(
            truth.person_of, prediction.person_of, exclude=truth.bots
        ),
        grouping=score_grouping(groups_truth, groups_pred),
        bots=score_bots(truth.bots, prediction.bots),
        organizer_tokens_per_million=(
            prediction.organizer_tokens / tokens * 1_000_000 if tokens else math.nan
        ),
        suggestions_per_thousand=(
            prediction.suggestions / len(created) * 1000 if created else math.nan
        ),
        identity_f05_by_case={case: s.f05 for case, s in by_case.items()},
        false_merges_by_case={case: s.false_merges for case, s in by_case.items()},
        grouping_f1_by_case=by_case_grouping,
    )
