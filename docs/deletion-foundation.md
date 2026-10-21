# Inactive deletion foundation and remaining read/cascade inventory

## What this tranche implements

Revision `d8c4f6a2b901` adds nullable `TIMESTAMPTZ deleted_at` to workspaces,
peers, sessions, and messages. There is no default, backfill, new index, or
index removal. The existing document tombstone/index and workspace/session/
collection generations remain unchanged. The normal migration runner supplies
advisory locking, the short lock timeout, and retry policy.

`src/crud/lifecycle.py` provides a small internal, **currently unused** boundary:

- `DedicatedTenantContext(db)` is an already-routed, caller-owned session. The
  dedicated database supplies tenant context; this is not a tenant-ID filter,
  authorization mechanism, or shared-pool implementation.
- `PrimitiveIdentity(kind, workspace_uid, uid)` uses `Workspace.id` for the
  scope and Workspace/Peer/Session.id or **Message.public_id** for the target.
  Names must be resolved under authorization before this boundary. A workspace
  identity requires its two IDs to match.
- `primitive_liveness_clause(identity)` is a self-contained, non-correlated
  EXISTS, not an arbitrary list-row predicate. It checks the supplied stable
  identity even if the outer query aliases or joins the same tables.
- `is_primitive_live(context, identity)` executes one SELECT without caches,
  ORM identity-map reads, autoflush, commit, rollback, or locks. Missing rows or
  parents are not live. Callers explicitly flush changes they want it to see.
  Ordinary sessions can autobegin a transaction; existing transaction isolation
  determines the snapshot. This is **not a final-write barrier**.

Workspace liveness is its own tombstone; peers/sessions additionally require a
live workspace; messages additionally require a live session and author peer.
`is_active` and membership `left_at` remain separate archive/access concepts.
No generic alias-sensitive table predicates or derived-liveness abstraction are
exported. No route, public schema, cache, or existing reader/writer changes.

This tranche alone is **not** the inactive expand release that can become the
activation rollback floor. Old images can read/write live rows on the additive
schema, but ignore tombstones. Do not activate deletion until the remaining
reader, barrier, invalidation, and purge work passes the integrated gate.

## Remaining production inventory

**All rows below are remaining work, not implemented by the nullable-column foundation.** The current schema is dedicated, not shared-pool. Existing public DELETE contracts remain unchanged (workspace/session 202 queueing; conclusion 204 soft delete; no peer/message DELETE).

## Per-primitive matrix

| Primitive | Authoritative liveness | Reads that must adopt lifecycle visibility | Physical cascade / invalidation closure | Lifecycle acceptance cases |
|---|---|---|---|---|
| Workspace | Stable Workspace.id, own tombstone; database selects tenant today | `crud/workspace.py`: `_fetch_workspace`, `get_workspace`, `get_or_create_workspace`, `get_all_workspaces`, stats/active peers; **every descendant route/tool and cache hit**; workspace chat/search, webhook listing | Both collection directions, documents, `document_sources` (derived and source edges), all peers/sessions/messages/chunks/memberships, webhook endpoints and queued webhook payloads, all tenant-bound queue rows and exact active claims. Durable purge/outbox control must be outside workspace cascade. Persist vector namespaces before deleting owners. | Warm all replica/process-local caches; kill API after tombstone commit before eviction; Redis down then restore persisted entries before rollback-floor start; no orphan enqueue or implicit workspace-name recreation; no global reconciler claim deletion; concurrent/repeated deletes preserve approved response policy. |
| Peer (including observer / reserved scope peer) | Stable Peer.id + stable Workspace.id; own and workspace tombstone | `crud/peer.py` fetch/get/list/get-or-create/sessions; `routers/peers.py` representation/chat/card/context/messages; `crud/peer_card.py` **both observer and observed**; named-scope resolution, session-peer listing/config and membership auth | Authored messages/chunks; all collections/documents where observer **or** observed equals selected peer; card blobs on other peers keyed by observed peer; memberships on both sides (session_peers now, peer_memberships when Composable Peers lands); payload observers fan-out, observed/observer/scope_peer; scope backfills/removals; delayed dreams on all workers, exact claim keys. Do not delete an otherwise-live multiparty session or other authors' input. | Delete one observer during multi-observer batch; author deletion hides authored messages but preserves another author/session; stale card setter/get-or-create must not recreate; same names in other workspaces/tenants unaffected; scope copy/restoration cannot clear tombstones. |
| Session | Stable Session.id + stable Workspace.id; own/workspace tombstone; `is_active` remains separate archive state | `crud/session.py` fetch/get/list/clone/membership/config; `routers/sessions.py` context/both summaries/messages; peer-session and scope-session lists; session scopes in message tools; cached session metadata | Messages/chunks, memberships, queue rows through both stable session_id and payload session_name, claims from exact selected keys; explicit memory tied to session; source/copy descendants across observers and sessionless/higher-order memory conservatively invalidated; both summary tiers + affected cards. Completed independent clones excluded, racing clones fenced. | Paused batch with context-only sources; cross-session dream, scope copy, reinforced conclusions; hide derived state during async discovery; clear summaries before rebuild and never seed from old summary; archived live sessions still behave as before; clone race versus completed independent clone. |
| Message | Stable Message.public_id + stable Workspace.id; own/workspace/session/**author** tombstone | `crud/message.py`: get/list/id and seq ranges/seq lookup, `_fetch_messages_by_ids`, lexical/grep/temporal/date tools, pgvector and external hydration, merged snippets; router lists/get/update/upload/create validations; summary/context assembly; conclusion source/evidence context; counts/token pagination | Queue numeric message_id must first resolve under stable workspace/tenant identity (never caller numeric ID alone); `message_embeddings` each chunk and external `{public_id}_{chunk_index}` IDs; conservative session-explicit memory closure, higher-order/cards where lineage incomplete, both summary tiers; document source/copy edges. Keep dependency/vector identifiers until confirmed purge. | Delete while fetch/dedup/model call paused; multi-observer work, context-only messages with no queue row; no stale upsert after external delete; no erased text in surrounding snippets; same public/numeric message IDs in actual shared tenants; source-parent deletion independent of child tombstone. |
| Conclusion (`Document`) | Stable Document.id + stable Workspace.id + both collection peers; optional session live; **representation invalidation state/source closure** beyond direct parents | `crud/document.py` all get/filter/recent/most-derived/semantic/external hydration/children; `crud/representation.py`; `utils/search.py`, agent_tools observation context/reasoning chains; dialectic/context/brief/notes inputs; external vector results rehydrated against DB visibility | `document_sources.derived_id` has cascade FK, **source_id intentionally does not**; reverse source edges + legacy source_ids/internal_metadata sources + COPIED_FROM_KEY links across observers; broad representation hide where evidence incomplete; cards and summary consequences; external vectors; durable suppression is separate record with manual/automatic write policy, not every tombstone. | Reinforcement fallback cannot resurrect; dedup disabled and automatic/manual writes follow policy; five-minute sweep cannot discard dependency edges before closure; empty card replacement clears old card; suppression survives physical conclusion purge; source/copy link discovery not restricted to one representation. |

## Cross-cutting seams / hotspots

1. **Cache authority.** `_fetch_workspace` reconstructs cached dict into ORM without SELECT; peer/session/collection caches have similar paths. `safe_cache_delete` swallows failures and Redis can fall back to process-local memory. Namespaced cache keys are not lifecycle proof. Database checks must precede returning a cached resource/derived artifact for reads starting after acknowledgment. Also inspect cached pagination offset anchors/counts and resolved config/scope paths; preserving deterministic `(created_at,id)` ordering still required. Outbox eviction cleans bytes later, never grants visibility.
2. **Current deletion shortcuts must be replaced, not reused.** `crud/workspace.py:delete_workspace` and `crud/session.py:delete_session` parse `work_unit_key` with `split_part`; session delete calls external vector store while holding DB transaction and commits internally. Workspace cleanup deletes queue/control rows and logs failed external namespace deletion. These are not new production lifecycle orchestration building blocks.
3. **Queue discovery for producer barriers and deletion.** Existing `a5f8da8f6d23` indexes cover pending payload `observers`, `observed`, `observer`, `scope_peer`, `session_name`; numeric message_id index and session_id/workspace indexes already exist. Use payload + relational fields, not parsed work keys. Claim deletion joins exact keys from selected queue rows; global reconcilers survive. A removed queue row does not revoke Python-held input, context-only sources, or another replica's delayed dream.
4. **Producer barriers for all final writers.** Enqueue/get-or-create, message ingestion/clone, deriver `create_documents`/reinforcement/fallback, scope copy/restoration, public/manual conclusions, dreamer conclusions/cards, summarizer both tiers, reconciler `sync_vectors` and low-latency `embed_now`, document-source backfill. Capture stable IDs/generations with input; validate sources and write while serialized against deletion. No external-call lock and no intermediate helper commit. Scope membership lock followed by committing helper is specifically unsafe.
5. **Conservative artifact visibility.** Four tombstones do not cover sessionless derived memory, cross-session synthesis, card strings, working briefs/notes, or copied conclusions. Existing provenance cannot prove exact closure (target-peer batch source stamps, no reinforced-evidence merge, copies in metadata). Hide affected representations synchronously while closure is discovered; clear old artifacts before rebuild; accept empty replacement and never reuse invalid summary as prompt input. Keep retention/invalidation policy decisions explicit.
6. **Purge/suppression.** Reconciler `sync_vectors._cleanup_soft_deleted_documents_pgvector` and `crud/document.cleanup_soft_deleted_documents` sweep after five minutes today. Fence these before enabling erasure. Durable state outside workspace cascade must retain pending vector deletes/namespace/IDs and survive queue cleanup/crash. Suppression content retention/near-duplicate scope policy remains separate approved work, not selected here.
7. **Auth/response compatibility.** Name-based JWTs and member/scope-read allowlists are unchanged. Liveness is an additional check, never a replacement for authorization. Do not enable `allow_member_read` on mutation. Verify own-peer subresource restriction and named-scope active membership. No API schema `deleted_at` field added. Existing 202 acceptance is not retroactively a complete-erasure guarantee. Repeated/concurrent deletion and recreation semantics need the separately approved route tranche.
8. **Shared lineage gate.** The proposed shared-pool layout changes primitive keys to tenant-composite, removes global name/public-id uniqueness and FKs for queue, gives nullable tenant attribution to queue/claims, and adds trigger-managed `queue_item_batches`. Every lookup/join/mutation including queue, claims, suppression and vector namespace needs explicit tenant scoping. No name-only or ID-only bulk operation; nullable service rows need deliberate attribution policy. Do not write aggregate rows directly. Actual shared-schema bootstrap/stamp fixtures and cross-tenant collision tests are required before shared activation, not a fabricated `tenant_id` in dedicated-schema tests.
9. **Rollback boundary.** Additive columns run under old image with NULLs, but the old image ignores tombstones. This foundation is not the complete inactive expand release: readers/writers/outbox/invalidation must all land before establishing that image as floor. Test activation→expand rollback with Redis restored before old-image start; v3.4.0 remains an unsafe negative control after erasure.

## Not settled by this tranche

In-flight streaming cancellation, public repeated-delete/recreation policy changes, durable outbox serialization/control schema, exact invalidation breadth where graph incomplete, suppression retention/threshold policy, and shared nullable-tenant attribution are not silently decided. The established guarantee concerns reads beginning after acknowledgment and completed clones are independent.
