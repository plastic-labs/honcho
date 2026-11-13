"""The held-out split: dev units to build and tune on, held-out units to judge.

Whole units are assigned, never single records: a unit is a fixture instance,
or for replayed data a person's accounts and the channels they post in. A
stable hash decides, so a unit stays on its side as the dataset grows, and no
organizer can be tuned on the people or channels it is judged on.
"""

import hashlib
from typing import Literal

from tests.bench.auto_workspaces.model import Dataset

Split = Literal["dev", "heldout", "all"]

# Fraction of units held out.
HELDOUT_FRACTION = 0.3
# Changing the salt reshuffles every unit; never change it once thresholds are set.
_SALT = "auto-workspaces-v1"


def side(unit: str) -> Literal["dev", "heldout"]:
    """Which side a unit is on."""
    digest = hashlib.sha256(f"{_SALT}:{unit}".encode()).digest()
    fraction = int.from_bytes(digest[:8]) / 2**64
    return "heldout" if fraction < HELDOUT_FRACTION else "dev"


def select(dataset: Dataset, split: Split) -> Dataset:
    """The part of the dataset on one side (or all of it)."""
    if split == "all":
        return dataset
    units = {r.unit for r in dataset.records} | set(
        dataset.truth.unit_of_account.values()
    )
    return dataset.subset(u for u in units if side(u) == split)


def missing_cases(dataset: Dataset, part: Dataset) -> set[str]:
    """Cases in the dataset that the selected part has no records of."""
    return {r.case for r in dataset.records} - {r.case for r in part.records}
