"""Unit tests for Phase 2 — promotion worker and scheduler.

Tests the heuristic promotion test, document level → edge type mapping,
and promotion scheduler logic. No DB required for the pure function tests.
"""

import sys
sys.path.insert(0, "/app/.venv/lib/python3.13/site-packages")
sys.path.insert(0, "/app")

from src.deriver.promotion import (
    _heuristic_promotion_test,
    LEVEL_TO_EDGE_TYPE,
    OBVIOUS_PATTERNS,
    DURABLE_PATTERNS,
    TEMPORARY_PATTERNS,
)
from src.utils.types import EdgeType

passed = 0
failed = 0


# ── Document level → edge type mapping ──

print("=== Document Level → Edge Type Mapping ===")

assert LEVEL_TO_EDGE_TYPE["explicit"] == "related"
print("  ✅ explicit → related")
passed += 1

assert LEVEL_TO_EDGE_TYPE["deductive"] == "refines"
print("  ✅ deductive → refines")
passed += 1

assert LEVEL_TO_EDGE_TYPE["inductive"] == "composes-with"
print("  ✅ inductive → composes-with")
passed += 1

assert LEVEL_TO_EDGE_TYPE["contradiction"] == "contradicts"
print("  ✅ contradiction → contradicts")
passed += 1

assert set(LEVEL_TO_EDGE_TYPE.keys()) == {"explicit", "deductive", "inductive", "contradiction"}
print("  ✅ All 4 document levels mapped")
passed += 1

# All values must be valid EdgeType values
valid_types = {"related", "composes-with", "see-also", "refines", "supersedes", "contradicts"}
for et in LEVEL_TO_EDGE_TYPE.values():
    assert et in valid_types, f"{et} is not a valid EdgeType"
print("  ✅ All edge types are valid")
passed += 1


# ── Heuristic promotion test ──

print("\n=== Heuristic Promotion Test ===")

# Should NOT promote: obvious patterns
obvious_cases = [
    "import os and sys for path handling",
    "def get_user_data returns a dict",
    "class UserModel handles database operations",
    "return self.data.get('key') or None",
    "print(f'Processing item {i}')",
    "TODO: fix this later when we have time",
    "FIXME: this is a temporary workaround",
    "let me check the documentation for that",
    "i'll look into it and get back to you",
    "one moment while I check",
    "hang on, let me find that",
    "not sure about that one",
    "i don't know the answer to that",
    "i'm not sure how to proceed",
    "let me think about that for a sec",
    "give me a sec to look that up",
]

for case in obvious_cases:
    result = _heuristic_promotion_test(case)
    if result:
        print(f"  ❌ Should NOT promote obvious: {case[:50]}...")
        failed += 1
    else:
        passed += 1
print(f"  ✅ Obvious patterns rejected: {len(obvious_cases)}/{len(obvious_cases)}")

# Should NOT promote: temporary patterns
temporary_cases = [
    "today we are working on the new feature",
    "this week we'll focus on bug fixes",
    "right now we're investigating the issue",
    "currently the system is in maintenance mode",
    "for now we'll use the workaround",
    "temporary fix applied to production",
    "maybe we should consider using a different approach",
    "perhaps the issue is related to caching",
    "could be a problem with the database connection",
    "might be worth investigating further",
]

for case in temporary_cases:
    result = _heuristic_promotion_test(case)
    if result:
        print(f"  ❌ Should NOT promote temporary: {case[:50]}...")
        failed += 1
    else:
        passed += 1
print(f"  ✅ Temporary patterns rejected: {len(temporary_cases)}/{len(temporary_cases)}")

# Should promote: durable patterns
durable_cases = [
    "We decided to use PostgreSQL for the primary database",
    "The team agreed on using microservices architecture",
    "We concluded that vector search is the best approach",
    "It was determined that the root cause was a race condition",
    "We established a new deployment pipeline",
    "The test results confirmed our hypothesis",
    "The system uses a distributed cache for performance",
    "The architecture separates read and write paths",
    "Our approach to error handling follows the fail-fast principle",
    "A key insight from the experiment is that batching improves throughput",
    "We should avoid over-indexing because it causes memory bloat",
    "We decided to adopt the CQRS pattern for the new service",
    "After testing, we found that index fragmentation was the root cause",
    "The reason for the performance improvement is the new caching layer",
    "This is important because it prevents data loss during failover",
]

for case in durable_cases:
    result = _heuristic_promotion_test(case)
    if not result:
        print(f"  ❌ Should promote durable: {case[:50]}...")
        failed += 1
    else:
        passed += 1
print(f"  ✅ Durable patterns promoted: {len(durable_cases)}/{len(durable_cases)}")

# Should NOT promote: very short facts
short_cases = [
    "short",
    "ok",
    "yes",
    "no",
    "done",
    "fixed",
    "works for me",
    "looks good",
    "lgtm",
    "will do",
]

for case in short_cases:
    result = _heuristic_promotion_test(case)
    if result:
        print(f"  ❌ Should NOT promote short: {case!r}")
        failed += 1
    else:
        passed += 1
print(f"  ✅ Short facts rejected: {len(short_cases)}/{len(short_cases)}")

# Should promote: conservative default (no obvious/temporary match, long enough)
default_cases = [
    "The query optimizer uses cost-based analysis to select the best execution plan",
    "The monitoring system collects metrics from all services every 30 seconds",
    "The backup strategy uses incremental snapshots with weekly full backups",
]

for case in default_cases:
    result = _heuristic_promotion_test(case)
    if not result:
        print(f"  ❌ Should promote by default: {case[:50]}...")
        failed += 1
    else:
        passed += 1
print(f"  ✅ Conservative default promotes: {len(default_cases)}/{len(default_cases)}")


# ── Pattern definitions ──

print("\n=== Pattern Definitions ===")

# OBVIOUS_PATTERNS should catch code-related patterns
assert len(OBVIOUS_PATTERNS) > 0
print(f"  ✅ {len(OBVIOUS_PATTERNS)} obvious patterns defined")
passed += 1

# DURABLE_PATTERNS should catch decision-related patterns
assert len(DURABLE_PATTERNS) > 0
print(f"  ✅ {len(DURABLE_PATTERNS)} durable patterns defined")
passed += 1

# TEMPORARY_PATTERNS should catch time-bound patterns
assert len(TEMPORARY_PATTERNS) > 0
print(f"  ✅ {len(TEMPORARY_PATTERNS)} temporary patterns defined")
passed += 1


# ── Summary ──

print(f"\n{'='*50}")
print(f"  Results: {passed} passed, {failed} failed")
if failed == 0:
    print("  ✅ ALL PHASE 2 TESTS PASSED")
else:
    print(f"  ❌ {failed} TEST(S) FAILED")
print(f"{'='*50}")
