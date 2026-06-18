//! Hybrid-search helpers, ported from `src/utils/search.py`.
//!
//! Covers the deterministic core — Reciprocal Rank Fusion (RRF), ILIKE escaping,
//! the FTS special-char test — plus the two SQL legs that produce the ranked
//! id lists RRF fuses: [`semantic_search_pgvector`] (pgvector cosine distance)
//! and [`fulltext_search`] (Postgres FTS with an ILIKE fallback). The query
//! embedding the semantic leg consumes is produced upstream by
//! [`crate::embedding::embed_openai`]. Per-message metadata filtering is
//! threaded into both legs via a [`crate::filters::FilterClause`]; the
//! peer-perspective temporal filter is applied on top.

use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::hash_map::Entry;
use std::hash::Hash;

use chrono::{DateTime, Utc};
use serde_json::Value;
use sqlx::PgPool;

use crate::filters::FilterClause;

/// RRF constant (`k`) controlling how steeply rank position discounts score.
/// Matches the Python default.
pub const RRF_K: f64 = 60.0;

/// Combine multiple ranked lists with Reciprocal Rank Fusion, porting
/// `reciprocal_rank_fusion`.
///
/// Each item's score is `sum(1 / (k + rank))` over the lists it appears in
/// (rank is 1-indexed). Results are ordered by score descending; ties keep
/// first-seen order (Python relies on dict insertion order + a stable sort), so
/// this reproduces Python's ordering exactly.
pub fn reciprocal_rank_fusion<T>(ranked_lists: &[Vec<T>], k: f64, limit: usize) -> Vec<T>
where
    T: Eq + Hash + Clone,
{
    if ranked_lists.is_empty() {
        return Vec::new();
    }

    let mut scores: HashMap<T, f64> = HashMap::new();
    // First-seen order of unique items, so equal-score ties sort stably the way
    // Python's insertion-ordered dict + stable `sorted` do.
    let mut order: Vec<T> = Vec::new();

    for ranked_list in ranked_lists {
        for (index, item) in ranked_list.iter().enumerate() {
            let contribution = 1.0 / (k + (index + 1) as f64);
            match scores.entry(item.clone()) {
                Entry::Occupied(mut entry) => *entry.get_mut() += contribution,
                Entry::Vacant(entry) => {
                    entry.insert(contribution);
                    order.push(item.clone());
                }
            }
        }
    }

    order.sort_by(|a, b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order.truncate(limit);
    order
}

/// The escape character used with SQL `ILIKE ... ESCAPE`, matching
/// `ILIKE_ESCAPE_CHAR` (backslash).
pub const ILIKE_ESCAPE_CHAR: char = '\\';

/// Escape `%`, `_`, and the backslash escape char in user text so an `ILIKE`
/// pattern matches it literally. Ports `escape_ilike_pattern`; the replacement
/// order (backslash first) matters so freshly-added escapes aren't re-escaped.
pub fn escape_ilike_pattern(text: &str) -> String {
    text.replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

/// The special characters that make Postgres FTS unreliable; their presence
/// switches the full-text leg to a literal `ILIKE` match. Mirrors the Python
/// regex character class `[~`!@#$%^&*()_+=\[\]{};':"\|,.<>/?-]`.
const FTS_SPECIAL_CHARS: &[char] = &[
    '~', '`', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', '=', '[', ']', '{', '}',
    ';', '\'', ':', '"', '\\', '|', ',', '.', '<', '>', '/', '?', '-',
];

/// Whether the query contains characters that Postgres FTS handles poorly, in
/// which case the caller should fall back to a literal `ILIKE` search. Ports the
/// `has_special_chars` regex test in `_fulltext_search`.
pub fn query_has_special_chars(query: &str) -> bool {
    query.chars().any(|c| FTS_SPECIAL_CHARS.contains(&c))
}

/// Render a query embedding as a pgvector text literal (`[v1,v2,...]`) for the
/// `::vector` cast in the cosine-distance ORDER BY.
fn vector_literal(embedding: &[f32]) -> String {
    let mut out = String::from("[");
    for (i, value) in embedding.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&value.to_string());
    }
    out.push(']');
    out
}

/// Semantic message search via pgvector, porting `_semantic_search_pgvector`.
///
/// Joins messages to their embedding chunks, keeps non-null embeddings in the
/// workspace, and orders by cosine distance (`<=>`) to the query vector. The
/// query oversamples (`limit * 2`) because one message can have several chunks;
/// duplicate `public_id`s are removed in insertion order and the first `limit`
/// are returned — matching Python's Python-side dedup (kept out of SQL to
/// preserve the HNSW index scan).
pub async fn semantic_search_pgvector(
    pool: &PgPool,
    workspace_name: &str,
    filter: &FilterClause,
    embedding_query: &[f32],
    limit: usize,
) -> Result<Vec<String>, sqlx::Error> {
    // The embedding columns are isolated in a subquery (`me`) so the outer
    // metadata `filter` can reference bare `messages` columns without ambiguity
    // (both tables carry `workspace_name`/`created_at`). The embeddings-side
    // `workspace_name` filter mirrors Python's explicit
    // `MessageEmbedding.workspace_name == workspace_name`; the message-side
    // workspace scope rides in via `filter` (the route forces `workspace_id`
    // into the filter dict, exactly like `apply_filter`).
    let ws_idx = filter.bindings.len() + 1;
    let vec_idx = filter.bindings.len() + 2;
    let limit_idx = filter.bindings.len() + 3;
    let sql = format!(
        "SELECT messages.public_id \
         FROM messages \
         JOIN (SELECT message_id, embedding FROM message_embeddings \
               WHERE embedding IS NOT NULL AND workspace_name = ${ws_idx}) me \
         ON messages.public_id = me.message_id \
         WHERE TRUE{} \
         ORDER BY me.embedding <=> ${vec_idx}::vector \
         LIMIT ${limit_idx}",
        filter.sql
    );
    let mut query = crate::db::bind_values_as(sqlx::query_as(&sql), &filter.bindings);
    query = query
        .bind(workspace_name)
        .bind(vector_literal(embedding_query))
        .bind((limit * 2) as i64);
    let rows: Vec<(String,)> = query.fetch_all(pool).await?;

    let mut seen: HashSet<String> = HashSet::new();
    let mut deduped: Vec<String> = Vec::new();
    for (public_id,) in rows {
        if seen.insert(public_id.clone()) {
            deduped.push(public_id);
            if deduped.len() == limit {
                break;
            }
        }
    }
    Ok(deduped)
}

/// Full-text message search, porting `_fulltext_search`.
///
/// Queries with FTS-hostile special characters fall back to a literal `ILIKE`
/// (ordered newest-first); natural-language queries combine `plainto_tsquery`
/// FTS with an `ILIKE` fallback and order by `ts_rank` then recency. User text
/// is escaped for the `ILIKE` pattern via [`escape_ilike_pattern`] (`\` escape).
pub async fn fulltext_search(
    pool: &PgPool,
    filter: &FilterClause,
    query: &str,
    limit: usize,
) -> Result<Vec<String>, sqlx::Error> {
    // Single table (no JOIN), so bare `filter` columns are unambiguous. The
    // workspace scope rides in via `filter` (Python's base stmt is already
    // `apply_filter`-ed with workspace + user filters before the FTS leg).
    let ilike_pattern = format!("%{}%", escape_ilike_pattern(query));

    let rows: Vec<(String,)> = if query_has_special_chars(query) {
        let ilike_idx = filter.bindings.len() + 1;
        let limit_idx = filter.bindings.len() + 2;
        let sql = format!(
            "SELECT public_id FROM messages \
             WHERE TRUE{} AND content ILIKE ${ilike_idx} ESCAPE '\\' \
             ORDER BY created_at DESC \
             LIMIT ${limit_idx}",
            filter.sql
        );
        let mut q = crate::db::bind_values_as(sqlx::query_as(&sql), &filter.bindings);
        q = q.bind(&ilike_pattern).bind(limit as i64);
        q.fetch_all(pool).await?
    } else {
        let q_idx = filter.bindings.len() + 1;
        let ilike_idx = filter.bindings.len() + 2;
        let limit_idx = filter.bindings.len() + 3;
        let sql = format!(
            "SELECT public_id FROM messages \
             WHERE TRUE{} AND ( \
                 to_tsvector('english', content) @@ plainto_tsquery('english', ${q_idx}) \
                 OR content ILIKE ${ilike_idx} ESCAPE '\\' \
             ) \
             ORDER BY coalesce( \
                 ts_rank(to_tsvector('english', content), plainto_tsquery('english', ${q_idx})), \
                 0 \
             ) DESC, created_at DESC \
             LIMIT ${limit_idx}",
            filter.sql
        );
        let mut q = crate::db::bind_values_as(sqlx::query_as(&sql), &filter.bindings);
        q = q.bind(query).bind(&ilike_pattern).bind(limit as i64);
        q.fetch_all(pool).await?
    };

    Ok(rows.into_iter().map(|(public_id,)| public_id).collect())
}

/// The fields the peer-perspective filter reads from a (hydrated) message.
#[derive(Debug, Clone, PartialEq)]
pub struct MessageRef {
    pub public_id: String,
    pub session_name: String,
    pub created_at: DateTime<Utc>,
}

/// Keep only messages created while `peer_name` was a member of their session,
/// porting `_filter_by_peer_perspective`. A message survives when its session
/// has at least one membership window `[joined_at, left_at]` (a `NULL` `left_at`
/// being still-open) that contains the message's `created_at`. Order is
/// preserved; messages in sessions the peer never joined are dropped.
pub async fn filter_by_peer_perspective(
    pool: &PgPool,
    workspace_name: &str,
    peer_name: &str,
    messages: &[MessageRef],
) -> Result<Vec<MessageRef>, sqlx::Error> {
    if messages.is_empty() {
        return Ok(Vec::new());
    }

    let memberships: Vec<(String, DateTime<Utc>, Option<DateTime<Utc>>)> = sqlx::query_as(
        "SELECT session_name, joined_at, left_at FROM session_peers \
         WHERE workspace_name = $1 AND peer_name = $2",
    )
    .bind(workspace_name)
    .bind(peer_name)
    .fetch_all(pool)
    .await?;

    // session_name -> [(joined_at, left_at)] membership windows.
    type Windows = HashMap<String, Vec<(DateTime<Utc>, Option<DateTime<Utc>>)>>;
    let mut windows: Windows = HashMap::new();
    for (session_name, joined_at, left_at) in memberships {
        windows
            .entry(session_name)
            .or_default()
            .push((joined_at, left_at));
    }

    let mut filtered = Vec::new();
    for message in messages {
        let Some(session_windows) = windows.get(&message.session_name) else {
            continue;
        };
        let in_window = session_windows.iter().any(|(joined_at, left_at)| {
            message.created_at >= *joined_at
                && left_at.is_none_or(|left| message.created_at <= left)
        });
        if in_window {
            filtered.push(message.clone());
        }
    }
    Ok(filtered)
}

/// Hydrate ids to [`MessageRef`]s (public_id, session_name, created_at),
/// preserving input order and dropping ids that don't resolve.
async fn fetch_message_refs_by_ids(
    pool: &PgPool,
    ids: &[String],
) -> Result<Vec<MessageRef>, sqlx::Error> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let rows: Vec<(String, String, DateTime<Utc>)> = sqlx::query_as(
        "SELECT public_id, session_name, created_at FROM messages WHERE public_id = ANY($1)",
    )
    .bind(ids)
    .fetch_all(pool)
    .await?;
    let mut by_id: HashMap<String, MessageRef> = rows
        .into_iter()
        .map(|(public_id, session_name, created_at)| {
            (
                public_id.clone(),
                MessageRef {
                    public_id,
                    session_name,
                    created_at,
                },
            )
        })
        .collect();
    Ok(ids.iter().filter_map(|id| by_id.remove(id)).collect())
}

/// Drop ids whose message falls outside `peer_name`'s session-membership
/// windows, preserving order. Hydrates the ids then reuses
/// [`filter_by_peer_perspective`].
async fn peer_filter_ids(
    pool: &PgPool,
    workspace_name: &str,
    peer_name: &str,
    ids: &[String],
) -> Result<Vec<String>, sqlx::Error> {
    let refs = fetch_message_refs_by_ids(pool, ids).await?;
    let filtered = filter_by_peer_perspective(pool, workspace_name, peer_name, &refs).await?;
    Ok(filtered.into_iter().map(|r| r.public_id).collect())
}

/// Inputs for [`hybrid_search`].
pub struct HybridSearchParams<'a> {
    pub workspace_name: &'a str,
    pub query: &'a str,
    /// The metadata filter applied to both legs, built from the request's
    /// `filters` with `workspace_id` forced in (mirroring `apply_filter`). The
    /// route is responsible for building this; the legs reference its bare
    /// `messages` columns.
    pub filter: &'a FilterClause,
    /// Pre-computed query embedding, or `None` when semantic search is disabled
    /// (no `EMBED_MESSAGES` / no embedding produced). Computed upstream by the
    /// route via [`crate::embedding::embed_openai`].
    pub query_embedding: Option<&'a [f32]>,
    pub peer_perspective: Option<&'a str>,
    pub limit: usize,
}

/// Hybrid message search, porting `search()._run_search`: run the semantic
/// (pgvector) and full-text legs, peer-filter each leg when a perspective is
/// given (matching Python, which filters semantic results in code and the
/// full-text leg via its SQL join — net effect both legs are filtered), fuse
/// the ranked id lists with RRF (or take the single leg when only one ran), and
/// hydrate the result back to message JSON.
pub async fn hybrid_search(
    pool: &PgPool,
    params: &HybridSearchParams<'_>,
) -> Result<Vec<Value>, sqlx::Error> {
    let limit = params.limit;
    let semantic_limit = if params.peer_perspective.is_some() {
        limit * 4
    } else {
        limit * 2
    };

    let mut ranked_lists: Vec<Vec<String>> = Vec::new();

    if let Some(embedding) = params.query_embedding {
        let mut ids = semantic_search_pgvector(
            pool,
            params.workspace_name,
            params.filter,
            embedding,
            semantic_limit,
        )
        .await?;
        if let Some(peer) = params.peer_perspective {
            ids = peer_filter_ids(pool, params.workspace_name, peer, &ids).await?;
        }
        ranked_lists.push(ids);
    }

    let mut fulltext_ids = fulltext_search(pool, params.filter, params.query, limit * 2).await?;
    if let Some(peer) = params.peer_perspective {
        fulltext_ids = peer_filter_ids(pool, params.workspace_name, peer, &fulltext_ids).await?;
    }
    ranked_lists.push(fulltext_ids);

    let fused: Vec<String> = if ranked_lists.len() > 1 {
        reciprocal_rank_fusion(&ranked_lists, RRF_K, limit)
    } else {
        ranked_lists
            .into_iter()
            .next()
            .unwrap_or_default()
            .into_iter()
            .take(limit)
            .collect()
    };

    crate::db::fetch_messages_by_ids(pool, &fused).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rrf(lists: &[&[&str]], limit: usize) -> Vec<String> {
        let owned: Vec<Vec<String>> = lists
            .iter()
            .map(|list| list.iter().map(|s| s.to_string()).collect())
            .collect();
        reciprocal_rank_fusion(&owned, RRF_K, limit)
    }

    /// Golden values captured from Python `reciprocal_rank_fusion`.
    #[test]
    fn fusion_matches_python() {
        let a: &[&str] = &["m1", "m2", "m3"];
        let b: &[&str] = &["m3", "m4", "m1"];
        // m1 and m3 tie (both appear once at rank 1 and once at rank 3); m1 is
        // seen first, so it leads. m2 (only A, rank 2) before m4 (only B, rank 2).
        assert_eq!(rrf(&[a, b], 10), vec!["m1", "m3", "m2", "m4"]);
        assert_eq!(rrf(&[a, b], 2), vec!["m1", "m3"]);
    }

    #[test]
    fn single_list_passthrough_and_empty() {
        assert_eq!(rrf(&[&["x", "y", "z"]], 2), vec!["x", "y"]);
        assert_eq!(rrf(&[], 5), Vec::<String>::new());
    }

    #[test]
    fn disjoint_lists_interleave_by_first_seen_on_ties() {
        // a,c tie at rank 1; b,d tie at rank 2; first-seen order a,b,c,d.
        assert_eq!(
            rrf(&[&["a", "b"], &["c", "d"]], 10),
            vec!["a", "c", "b", "d"]
        );
    }

    #[test]
    fn escape_ilike_matches_python_examples() {
        assert_eq!(escape_ilike_pattern("100%"), "100\\%");
        assert_eq!(escape_ilike_pattern("file_name"), "file\\_name");
        assert_eq!(escape_ilike_pattern("path\\to\\file"), "path\\\\to\\\\file");
        // Backslash is escaped before % / _, so a literal "\%" becomes "\\\%".
        assert_eq!(escape_ilike_pattern("\\%"), "\\\\\\%");
        assert_eq!(escape_ilike_pattern("plain text"), "plain text");
    }

    #[test]
    fn special_char_detection_matches_python_regex() {
        for query in ["hello!", "100%", "file_name", "a-b", "c:d", "x.y"] {
            assert!(
                query_has_special_chars(query),
                "{query:?} should be special"
            );
        }
        for query in ["hello world", "naïve café", "plain query 123"] {
            assert!(
                !query_has_special_chars(query),
                "{query:?} should not be special"
            );
        }
    }
}
