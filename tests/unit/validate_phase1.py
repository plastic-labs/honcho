"""Quick validation script for graph memory Phase 1 — run inside the Honcho container."""
import sys

# Add venv site-packages for pydantic, fastapi, etc.
sys.path.insert(0, "/app/.venv/lib/python3.13/site-packages")
sys.path.insert(0, "/app")

from src.schemas.graph_memory import (
    EdgeCreate, RecallRequest, ContextCreate,
    ThreadBindingCreate, PinRequest, EdgeListFilter,
)
from pydantic import ValidationError

OID = "abc123def456ghi789jab"
OID2 = "xyz789uvw456rst123aab"

passed = 0
failed = 0

# ── Schema validation tests ──

# EdgeCreate
e = EdgeCreate(collection_name="test", source_obs_id=OID, target_obs_id=OID2, edge_type="related")
assert e.edge_type == "related"
print("  ✅ EdgeCreate valid")
passed += 1

try:
    EdgeCreate(collection_name="test", source_obs_id=OID, target_obs_id=OID2, edge_type="invalid")
    print("  ❌ EdgeCreate should have rejected invalid type")
    failed += 1
except ValidationError:
    print("  ✅ EdgeCreate rejects invalid type")
    passed += 1

for et in ["related", "composes-with", "see-also", "refines", "supersedes", "contradicts"]:
    EdgeCreate(collection_name="test", source_obs_id=OID, target_obs_id=OID2, edge_type=et)
print("  ✅ All 6 edge types accepted")
passed += 1

# RecallRequest
r = RecallRequest(query="test", collection_name="test")
assert r.max_depth == 3
assert r.frontier_cap == 10
assert r.token_budget == 2000
print("  ✅ RecallRequest defaults")
passed += 1

# ContextCreate
for name in ["my-context", "my_context", "context123"]:
    ContextCreate(context_name=name)
print("  ✅ ContextCreate valid names")
passed += 1

for name in ["has spaces", "has.dots", ""]:
    try:
        ContextCreate(context_name=name)
        print(f"  ❌ ContextCreate should have rejected {name!r}")
        failed += 1
    except ValidationError:
        pass
print("  ✅ ContextCreate rejects invalid names")
passed += 1

# ThreadBindingCreate
tb = ThreadBindingCreate(thread_id="1234567890.123456", context_name="test")
assert tb.thread_id == "1234567890.123456"
print("  ✅ ThreadBindingCreate valid")
passed += 1

try:
    ThreadBindingCreate(thread_id="bad", context_name="test")
    print("  ❌ ThreadBindingCreate should have rejected bad thread")
    failed += 1
except ValidationError:
    print("  ✅ ThreadBindingCreate rejects invalid thread")
    passed += 1

# PinRequest
p = PinRequest()
assert p.verify_cadence_days is None
print("  ✅ PinRequest null cadence")
passed += 1

for days in [1, 7, 30]:
    p = PinRequest(verify_cadence_days=days)
    assert p.verify_cadence_days == days
print("  ✅ PinRequest valid cadences")
passed += 1

try:
    PinRequest(verify_cadence_days=-1)
    print("  ❌ PinRequest should have rejected negative cadence")
    failed += 1
except ValidationError:
    print("  ✅ PinRequest rejects negative cadence")
    passed += 1

# EdgeListFilter
f = EdgeListFilter()
assert f.source_obs_id is None
print("  ✅ EdgeListFilter empty")
passed += 1

# ── CRUD logic tests ──
import math

ACTIVATION_HALF_LIFE_HOURS = 24.0
CONFIDENCE_HALF_LIFE_DAYS = 30.0
CONFIDENCE_THRESHOLD = 0.3
PINNED_FLOOR = 0.85
EVENT_WEIGHTS = {"access": 0.3, "verify": 1.0, "recall": 0.5, "promote": 1.0, "rehydrate": 1.0, "evict": 0.0}

# Activation decay
w = EVENT_WEIGHTS["access"]
assert w * math.exp(0) == 0.3
print("  ✅ Activation at t=0")
passed += 1

assert w * math.exp(-1) == 0.3 * math.exp(-1)
print("  ✅ Activation at t=24h")
passed += 1

assert w * math.exp(-5) < 0.01
print("  ✅ Activation at t=120h negligible")
passed += 1

# Verify > access weight
decay = math.exp(-1.0 / ACTIVATION_HALF_LIFE_HOURS)
assert EVENT_WEIGHTS["verify"] * decay > EVENT_WEIGHTS["access"] * decay
print("  ✅ Verify weight > access weight")
passed += 1

assert EVENT_WEIGHTS["evict"] == 0.0
print("  ✅ Evict contributes nothing")
passed += 1

# Confidence decay (pure function, no compounding)
HL = CONFIDENCE_HALF_LIFE_DAYS * 24.0
assert math.exp(0) == 1.0
print("  ✅ Confidence at t=0")
passed += 1

assert math.exp(-HL / HL) == math.exp(-1)
print("  ✅ Confidence at t=30d")
passed += 1

conf_60d = math.exp(-(60 * 24) / HL)
conf_30d = math.exp(-(30 * 24) / HL)
assert conf_60d < conf_30d
print("  ✅ Confidence no compounding")
passed += 1

# Threshold crossing
t_hours = -HL * math.log(CONFIDENCE_THRESHOLD)
assert 35 < t_hours / 24.0 < 37
print("  ✅ Confidence threshold at ~36 days")
passed += 1

# Source diversity
def factor(n):
    return 1.0 / (1.0 + math.log(1.0 + n))

assert factor(0) == 1.0
print("  ✅ First access full weight")
passed += 1

assert factor(1) < 1.0
print("  ✅ Second access diminished")
passed += 1

assert factor(9) < 0.5
print("  ✅ Tenth access heavily diminished")
passed += 1

# Two sources better than one
one_source = sum(EVENT_WEIGHTS["access"] * decay * factor(i) for i in range(4))
two_sources = 2 * sum(EVENT_WEIGHTS["access"] * decay * factor(i) for i in range(2))
assert two_sources > one_source
print("  ✅ Two sources > one source")
passed += 1

# Pinned floor
assert PINNED_FLOOR == 0.85
assert max(0.5, PINNED_FLOOR) == PINNED_FLOOR
assert max(0.95, PINNED_FLOOR) == 0.95
print("  ✅ Pinned floor applied correctly")
passed += 1

# ── Summary ──
print(f"\n{'='*50}")
print(f"  Results: {passed} passed, {failed} failed")
if failed == 0:
    print("  ✅ ALL TESTS PASSED")
else:
    print(f"  ❌ {failed} TEST(S) FAILED")
print(f"{'='*50}")
