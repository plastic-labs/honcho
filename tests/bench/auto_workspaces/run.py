"""Score an organizer on the auto-workspaces benchmark.

    python -m tests.bench.auto_workspaces.run                    # synthetic, dev split
    python -m tests.bench.auto_workspaces.run --split heldout
    python -m tests.bench.auto_workspaces.run --data DIR         # a dataset on disk
    python -m tests.bench.auto_workspaces.run --write DIR        # save the fixtures

Only the rules-only baseline exists so far; an organizer plugs in as another
`organize(dataset) -> Prediction`.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import get_args

from tests.bench.auto_workspaces import baseline, fixtures
from tests.bench.auto_workspaces.model import Dataset
from tests.bench.auto_workspaces.scoring import Report, score
from tests.bench.auto_workspaces.split import Split, missing_cases, select


def _table(report: Report) -> str:
    i, g, b = report.identity, report.grouping, report.bots
    lines = [
        f"records {report.records}  accounts {report.accounts}  "
        + f"ingested tokens {report.ingested_tokens}",
        "",
        f"identity  F0.5 {i.f05:.3f}  precision {i.precision:.3f}  "
        + f"recall {i.recall:.3f}  false merges {i.false_merges}  "
        + f"false splits {i.false_splits} (of {i.true_pairs} true pairs)",
        f"grouping  F1 {g.f1:.3f}  precision {g.precision:.3f}  "
        + f"recall {g.recall:.3f}  groups {g.predicted_groups} predicted, "
        + f"{g.true_groups} true",
        f"bots      precision {b.precision:.3f}  recall {b.recall:.3f}",
        f"cost      {report.organizer_tokens_per_million:.0f} organizer tokens "
        + "per 1M ingested",
        f"review    {report.suggestions_per_thousand:.1f} suggestions per 1,000 "
        + "records",
        "",
        f"{'case':<24} {'identity F0.5':>14} {'false merges':>13} {'grouping F1':>12}",
    ]
    for case, f05 in report.identity_f05_by_case.items():
        merges = report.false_merges_by_case.get(case, 0)
        grouping = report.grouping_f1_by_case.get(case, float("nan"))
        lines.append(f"{case:<24} {f05:>14.3f} {merges:>13} {grouping:>12.3f}")
    if i.merged_wrongly:
        lines += ["", "false merges (first few):"]
        lines += [f"  {x}  +  {y}" for x, y in i.merged_wrongly[:10]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score an organizer on the auto-workspaces benchmark."
    )
    parser.add_argument("--split", choices=get_args(Split), default="dev")
    parser.add_argument("--data", type=Path, help="score a dataset written to DIR")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--instances", type=int, default=30)
    parser.add_argument("--write", type=Path, help="write the dataset to DIR and exit")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)

    dataset = (
        Dataset.read(args.data)
        if args.data
        else fixtures.generate(seed=args.seed, instances=args.instances)
    )
    if args.write:
        dataset.write(args.write)
        print(f"wrote {len(dataset.records)} records to {args.write}")
        return 0

    part = select(dataset, args.split)
    if missing := missing_cases(dataset, part):
        parser.error(
            f"the {args.split} split has no {', '.join(sorted(missing))}: "
            + "generate more --instances"
        )
    # The organizer sees records and profiles, never labels or truth.
    report = score(part, baseline.organize(part.blind()))
    print(json.dumps(report.as_dict(), indent=1) if args.json else _table(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
