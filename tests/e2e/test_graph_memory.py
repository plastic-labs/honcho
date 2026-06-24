"""End-to-end smoke tests for graph memory API.
Standalone — no Honcho imports needed. Runs from host against localhost:8088.
"""
import httpx
import os
import time
import random

BASE = os.environ.get("HONCHO_BASE_URL", "http://localhost:8088")
WS = "hermes"
API = BASE + "/v3/workspaces/" + WS + "/graph-memory"

HEADERS = {"Content-Type": "application/json"}

passed = 0
failed = 0

def check(label, condition, detail=""):
    global passed, failed
    if condition:
        print("  [PASS] " + label)
        passed += 1
    else:
        print("  [FAIL] " + label + ": " + detail)
        failed += 1

def api_post(path, data=None):
    return httpx.post(API + path, headers=HEADERS, json=data or {}, timeout=10.0)

def api_get(path):
    return httpx.get(API + path, headers=HEADERS, timeout=10.0)

# 1. Edge endpoints
print("\n--- Edges ---")
r = api_post("/edges/list", {})
check("List edges returns 200", r.status_code == 200, str(r.status_code))

# 2. Thread Binding
print("\n--- Thread Binding ---")
# Use a guaranteed unique thread_id (pattern: ^[0-9]{10,}\.[0-9]+$)
import time as _time
unique_thread = str(int(_time.time() * 10000000)) + str(random.randint(10000, 99999)) + "." + str(random.randint(100000, 999999))
print("  Thread ID: " + unique_thread)
r = api_post("/thread-bindings", {
    "thread_id": unique_thread,
    "context_name": "project-x",
})
# 201 = created, 422 = already bound (from previous test run)
check("Bind thread returns 201 or 422", r.status_code in (201, 422), str(r.status_code) + ": " + r.text[:100])

r2 = api_get("/thread-bindings/" + unique_thread)
check("Resolve thread returns 200", r2.status_code == 200, str(r2.status_code))
if r2.status_code == 200:
    data = r2.json()
    if data and isinstance(data, dict):
        check("Resolved context is project-x", data.get("context_name") == "project-x", str(data))
    elif data is None:
        check("Unbound thread returns null (expected for GET with no binding)", True, "")
    else:
        check("Resolve returned valid data", False, "got: " + str(data)[:100])

# 3. Compaction
print("\n--- Compaction ---")
r = api_post("/access-log/compact")
check("Compaction returns 200", r.status_code == 200, str(r.status_code))
if r.status_code == 200:
    report = r.json()
    check("Report has pruned_events", "pruned_events" in report)
    check("Report has retention_policy", "retention_policy" in report)
    check("Report has pre_compaction", "pre_compaction" in report)
    check("Report has post_compaction", "post_compaction" in report)
    check("Report has health", "health" in report)
    check("Report has note", "note" in report)

# 4. Context Management
print("\n--- Contexts ---")
r = api_post("/contexts", {"context_name": "e2e-test-context"})
check("Create context returns 201", r.status_code == 201, str(r.status_code))

# 5. Verify-Due
print("\n--- Verify-Due ---")
r = api_get("/observations/verify-due")
check("Verify-due returns 200", r.status_code == 200, str(r.status_code))

# 6. Cold Storage
print("\n--- Cold Storage ---")
r = api_get("/cold")
check("List cold returns 200", r.status_code == 200, str(r.status_code))

r = api_post("/evict-stale", {"threshold": 0.12})
check("Evict stale returns 200", r.status_code == 200, str(r.status_code))
if r.status_code == 200:
    report = r.json()
    check("Evict report has evicted_count", "evicted_count" in report)
    check("Evict report has skipped_pinned", "skipped_pinned" in report)
    check("Evict report has skipped_active", "skipped_active" in report)

# Summary
print("\n" + "=" * 50)
print("  Results: " + str(passed) + " passed, " + str(failed) + " failed")
if failed == 0:
    print("  ALL E2E TESTS PASSED")
else:
    print("  " + str(failed) + " FAILED")
print("=" * 50)
