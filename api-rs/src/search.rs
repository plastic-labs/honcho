//! Hybrid-search helpers, ported from `src/utils/search.py`.
//!
//! So far this covers the deterministic, network-free core: Reciprocal Rank
//! Fusion (RRF), which merges the semantic and full-text result lists. The
//! semantic leg (query embedding + pgvector / external store) and the full-text
//! SQL are layered on top once the embedding client lands.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::hash::Hash;

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
