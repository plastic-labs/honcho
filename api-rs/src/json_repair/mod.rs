//! Port of the `json_repair` library (no-schema path only).
//!
//! Honcho calls `repair_json(json_str)` with no schema, no file descriptor, and
//! all flags default — so only `JSONParser.parse()` (objects/arrays/strings/
//! numbers/literals/comments/parenthesized) is reachable; the ~1000-line schema
//! machinery (`schema_repair.py`, `parser_schema.py`, etc.) is never exercised
//! and is intentionally out of scope.
//!
//! Port status: foundation in place (`JsonContext`, `ObjectComparer`). The
//! recursive parser itself (`JSONParser` + `parse_string`/`parse_object`/
//! `parse_array`/…) is being ported as a dedicated multi-step effort, validated
//! against the real Python library used as a parity oracle. Until that lands,
//! `comprehensive_json_repair` (see [`crate::json_parser`]) remains the only
//! wired repair path.

pub mod boolean_null;
pub mod context;
pub mod object_comparer;
pub mod parser;
pub mod string_helpers;

/// Port of `STRING_DELIMITERS` (`utils/constants.py`).
pub const STRING_DELIMITERS: [char; 4] = ['"', '\'', '\u{201c}', '\u{201d}'];
