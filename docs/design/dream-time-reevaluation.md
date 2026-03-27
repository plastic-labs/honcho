# Dream-Time Re-evaluation of Orphaned Reasoning Chains

## Problem

When a document is superseded (replaced by a more informative version),
downstream documents that reference it via `source_ids` become based on
stale premises. The current implementation (v1) handles this at a single
level: the old document gets a `superseded_by` link and chain traversal
follows it one hop. But the downstream deductions derived from the old
premise may no longer hold under the new one.

**Example:**

```
A: "Meeting is on Tuesday"
  -> B (deductive): "User is busy Tuesday afternoon"
    -> C (deductive): "User prefers morning meetings"

A is superseded by A': "Meeting moved to Thursday"
```

B's conclusion ("busy Tuesday") is now wrong. C's conclusion ("prefers
mornings") may still hold but for different reasons. Neither B nor C is
automatically re-evaluated.

## Current Behavior (v1)

- `superseded_by` column on `Document` model (`src/models.py`)
- Dedup path sets the link in `create_documents` (`src/crud/document.py`)
- Dreamer tool sets the link via `delete_observations` with `superseded_by` param (`src/utils/agent_tools.py`)
- Chain traversal follows the link one hop: shows "A (superseded) -> A'" (`src/utils/agent_tools.py`, `_handle_get_reasoning_chain`)
- Reconciler preserves tombstones (superseded docs) that have live dependents (`src/crud/document.py`, `cleanup_soft_deleted_documents`)
- Tombstones are cleaned up once all dependents are gone
- No cascade to downstream documents

## Proposed: Eager Cascade

At the time a document is superseded, mark all descendants for re-evaluation:

1. **Mark phase:** Walk `source_ids` references to find all documents
   that transitively depend on the superseded document. Set a
   `needs_reevaluation` flag (or soft-delete with `superseded_by`
   pointing to the root cause).

2. **Re-evaluation phase:** During the next dream cycle, the deduction
   specialist encounters the flagged documents. For each one:
   - Fetch the new premise (via `superseded_by` chain)
   - Re-derive the conclusion against the updated premise
   - If the conclusion still holds: create a new version, supersede the old
   - If the conclusion no longer holds: delete without replacement

3. **Cascade termination:** Re-evaluation propagates naturally through
   the dream cycle. Each re-derived document may itself trigger
   re-evaluation of its dependents.

### Risks

- **Cascade storms:** In dense graphs, a single supersession at the root
  could trigger re-evaluation of hundreds of documents. Needs batching
  and depth limits.
- **Cycle detection:** If `source_ids` ever form cycles (shouldn't by
  construction, but defensive), cascade would loop infinitely. One-hop
  limit in v1 avoids this entirely.
- **Specialist capacity:** The deduction specialist would need to handle
  re-evaluation as a distinct mode, not just creation. Requires prompt
  changes and potentially a dedicated tool.

## Alternative: In-Place `source_ids` Update

Instead of re-evaluating downstream documents, update their `source_ids`
to point to the replacement document:

1. When A is superseded by A', find all documents with A in their
   `source_ids`
2. Replace A with A' in each document's `source_ids` array
3. The downstream documents now reference the correct premise

### Advantages

- Simpler than full re-evaluation
- No cascade storms (single pass)
- Preserves existing deductions

### Risks

- **Logical integrity:** A deduction derived from premise A may not hold
  when the premise changes to A'. Re-parenting without re-evaluation
  assumes the conclusion is premise-independent, which is often false.
- **Silent corruption:** The deduction looks correct (has valid
  `source_ids`) but its reasoning is based on a premise that no longer
  exists. Worse than a visible gap.

This approach is better suited as a complement to eager cascade: after
re-evaluation confirms the conclusion still holds, update the
`source_ids` to the new premise.

## Proposed: Full Chain Deletion

When a superseded tombstone's entire subtree is also superseded (no live
dependents anywhere in the chain), hard-delete the full chain:

1. During reconciler cleanup, check if a tombstone's dependents are all
   themselves superseded
2. If so, the entire chain is "fully superseded" — no live document
   depends on any node in the chain
3. Hard-delete all nodes in the chain in a single pass

This bounds tombstone accumulation to chains with at least one live
dependent. Fully superseded chains are cleaned up completely rather
than persisting as a linked list of tombstones.

## Why Not Implemented Now

- Unbounded cascade depth requires careful batching and limits
- Cycle detection adds complexity for a case that shouldn't occur
- Specialist tool changes needed for re-evaluation mode
- Risk of cascade storms in dense observation graphs
- The v1 implementation (one-hop follow + tombstone preservation)
  handles the common case and provides the schema foundation for
  future work
