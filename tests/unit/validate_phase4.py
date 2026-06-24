"""Phase 4 validation — eviction cold storage tests."""
import sys
sys.path.insert(0, "/app/.venv/lib/python3.13/site-packages")
sys.path.insert(0, "/app")

from src.crud.graph_memory import (
    EVICTION_THRESHOLD,
    REHYDRATE_RESTORE,
    LOG_RETENTION_HALF_LIVES,
    ACTIVATION_HALF_LIFE_HOURS,
)

passed = 0
failed = 0

print("=== Eviction Constants ===")
assert EVICTION_THRESHOLD == 0.12
print("  ✅ EVICTION_THRESHOLD = 0.12")
passed += 1

assert REHYDRATE_RESTORE == 0.60
print("  ✅ REHYDRATE_RESTORE = 0.60 (hysteresis gap)")
passed += 1

print("\n=== Cold Storage Schema ===")
from src.models import DocumentCold
assert DocumentCold.__tablename__ == "documents_cold"
print("  ✅ documents_cold table exists")
passed += 1

# Check columns
import inspect
cols = [c.name for c in DocumentCold.__table__.columns]
expected_cols = {"id", "workspace_name", "collection_name", "content", "level",
                 "metadata", "internal_metadata", "embedding", "evicted_at",
                 "edge_snapshot", "access_log_tail", "rehydrated_at"}
missing = expected_cols - set(cols)
if missing:
    print(f"  ❌ Missing columns: {missing}")
    failed += 1
else:
    print(f"  ✅ All {len(expected_cols)} columns present")
    passed += 1

print("\n=== Hysteresis Gap ===")
# The hysteresis gap prevents thrashing: evict at 0.12, restore at 0.60
assert REHYDRATE_RESTORE > EVICTION_THRESHOLD
print(f"  ✅ Hysteresis gap: evict at {EVICTION_THRESHOLD}, restore at {REHYDRATE_RESTORE}")
passed += 1

gap = REHYDRATE_RESTORE - EVICTION_THRESHOLD
assert gap > 0.4
print(f"  ✅ Gap = {gap:.2f} (sufficient to prevent thrashing)")
passed += 1

print("\n=== Edge Snapshot ===")
from src.crud.graph_memory import _snapshot_edges, _snapshot_access_log
import inspect
assert callable(_snapshot_edges)
print("  ✅ _snapshot_edges function exists")
passed += 1
assert callable(_snapshot_access_log)
print("  ✅ _snapshot_access_log function exists")
passed += 1

print("\n=== Rehydration ===")
from src.crud.graph_memory import rehydrate_observation, list_cold_observations
assert callable(rehydrate_observation)
print("  ✅ rehydrate_observation function exists")
passed += 1
assert callable(list_cold_observations)
print("  ✅ list_cold_observations function exists")
passed += 1

print(f"\n{'='*50}")
print(f"  Results: {passed} passed, {failed} failed")
if failed == 0:
    print("  ✅ ALL PHASE 4 TESTS PASSED")
else:
    print(f"  ❌ {failed} TEST(S) FAILED")
print(f"{'='*50}")
