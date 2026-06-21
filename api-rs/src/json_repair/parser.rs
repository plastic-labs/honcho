//! Port of the `JSONParser` scaffolding from `json_repair/json_parser.py`:
//! the parser state and its low-level cursor primitives. The recursive
//! `parse_*` methods are ported in subsequent units; this file establishes the
//! buffer + index model they all build on.
//!
//! Faithfulness notes:
//! - The buffer is a mutable `Vec<char>` because the Python parser splices into
//!   `self.json_str` mid-parse (duplicate-key split, empty-object reparse).
//! - `index` is signed and indexing mirrors Python string semantics: a single
//!   negative wrap (`s[-1]` is the last char) before going out of range. The
//!   parser relies on this (e.g. `get_char_at(-1)`).

use serde_json::Value;

use super::STRING_DELIMITERS;
use super::array::parse_array;
use super::comment::parse_comment;
use super::context::{ContextValues, JsonContext};
use super::number::parse_number;
use super::object::parse_object;
use super::object_comparer::is_same_object;
use super::parenthesized::{parenthesized_is_explicit_tuple, top_level_parenthesized_can_start_value};
use super::string::parse_string;

pub struct JsonParser {
    /// The string to parse, as code points (Python `self.json_str`).
    pub json: Vec<char>,
    /// Cursor (Python `self.index`); signed to support negative-offset reads.
    pub index: i64,
    pub context: JsonContext,
    pub deferred_contexts: Vec<ContextValues>,
    pub stream_stable: bool,
    pub strict: bool,
}

impl JsonParser {
    pub fn new(json_str: &str) -> Self {
        JsonParser {
            json: json_str.chars().collect(),
            index: 0,
            context: JsonContext::new(),
            deferred_contexts: Vec::new(),
            stream_stable: false,
            strict: false,
        }
    }

    #[inline]
    pub fn len(&self) -> i64 {
        self.json.len() as i64
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.json.is_empty()
    }

    /// Python string indexing semantics: one negative wrap, then bounds-check.
    fn char_at(&self, mut idx: i64) -> Option<char> {
        let n = self.len();
        if idx < 0 {
            idx += n;
        }
        if idx < 0 || idx >= n {
            None
        } else {
            Some(self.json[idx as usize])
        }
    }

    /// Port of `get_char_at(count=0)`.
    pub fn get_char_at(&self, count: i64) -> Option<char> {
        self.char_at(self.index + count)
    }

    /// Port of `skip_whitespaces`: advance `index` over whitespace.
    pub fn skip_whitespaces(&mut self) {
        while let Some(c) = self.char_at(self.index) {
            if c.is_whitespace() {
                self.index += 1;
            } else {
                break;
            }
        }
    }

    /// Port of `scroll_whitespaces(idx=0)`: advance a local offset over
    /// whitespace without moving `index`; returns the resulting offset.
    pub fn scroll_whitespaces(&self, mut idx: i64) -> i64 {
        while let Some(c) = self.char_at(self.index + idx) {
            if c.is_whitespace() {
                idx += 1;
            } else {
                break;
            }
        }
        idx
    }

    /// Port of `skip_to_character`: advance from `index + idx` until an
    /// *unescaped* target char (even number of preceding backslashes). Returns
    /// the offset from `index` to that position, or the distance to the end.
    pub fn skip_to_character(&self, targets: &[char], idx: i64) -> i64 {
        let mut i = self.index + idx;
        let n = self.len();
        let mut backslashes = 0u32; // consecutive '\' immediately before current char

        while i < n {
            let ch = self.json[i as usize];

            if ch == '\\' {
                backslashes += 1;
                i += 1;
                continue;
            }

            if targets.contains(&ch) && backslashes.is_multiple_of(2) {
                return i - self.index;
            }

            backslashes = 0;
            i += 1;
        }

        n - self.index
    }

    /// Convenience wrapper for the single-character form of `skip_to_character`.
    pub fn skip_to_char(&self, target: char, idx: i64) -> i64 {
        self.skip_to_character(&[target], idx)
    }

    /// Python slice `json_str[start:end]` for non-negative indices: clamped to
    /// bounds, empty when `start >= end`.
    pub fn substr(&self, start: i64, end: i64) -> String {
        let n = self.len();
        let s = start.clamp(0, n);
        let e = end.clamp(0, n);
        if s >= e {
            return String::new();
        }
        self.json[s as usize..e as usize].iter().collect()
    }

    /// Replace `json[start..end]` with `replacement` (the in-place buffer
    /// splicing the Python parser does via `self.json_str = ... + ... + ...`).
    pub fn splice(&mut self, start: i64, end: i64, replacement: &str) {
        let n = self.len();
        let s = start.clamp(0, n) as usize;
        let e = end.clamp(0, n) as usize;
        let e = e.max(s);
        self.json.splice(s..e, replacement.chars());
    }

    /// Port of `JSONParser.parse` + `_parse_top_level` (no schema, non-strict).
    pub fn parse(&mut self) -> Value {
        let mut json = self.parse_json();
        if self.index < self.len() {
            let mut elements = vec![json];
            while self.index < self.len() {
                self.context.clear();
                self.deferred_contexts.clear();
                let j = self.parse_json();
                if super::is_truthy(&j) {
                    let last = elements.last().unwrap();
                    // Treat repeated objects as updates; also drop a falsy tail.
                    if is_same_object(last, &j) || !super::is_truthy(last) {
                        elements.pop();
                    }
                    elements.push(j);
                } else {
                    self.index += 1;
                }
            }
            json = if elements.len() == 1 {
                elements.pop().unwrap()
            } else {
                Value::Array(elements)
            };
        }
        json
    }

    /// Port of `JSONParser.parse_json` (no-schema path).
    pub fn parse_json(&mut self) -> Value {
        if !self.deferred_contexts.is_empty() {
            let deferred = std::mem::take(&mut self.deferred_contexts);
            for &cv in &deferred {
                self.context.set(cv);
            }
            let result = self.parse_json();
            for _ in &deferred {
                self.context.reset();
            }
            return result;
        }

        loop {
            let char = self.get_char_at(0);
            match char {
                None => return Value::String(String::new()),
                Some('{') => {
                    self.index += 1;
                    return parse_object(self);
                }
                Some('[') => {
                    self.index += 1;
                    return parse_array(self, ']');
                }
                Some('(') => {
                    if !self.context.empty || top_level_parenthesized_can_start_value(self) {
                        return self.parse_parenthesized();
                    }
                    self.index += 1;
                }
                Some(c) => {
                    if !self.context.empty && (STRING_DELIMITERS.contains(&c) || c.is_alphabetic()) {
                        return parse_string(self);
                    }
                    if !self.context.empty && (c.is_ascii_digit() || c == '-' || c == '.') {
                        return parse_number(self);
                    }
                    if c == '#' || c == '/' {
                        return parse_comment(self);
                    }
                    self.index += 1;
                }
            }
        }
    }

    /// Port of `JSONParser.parse_parenthesized`.
    fn parse_parenthesized(&mut self) -> Value {
        let explicit_tuple = parenthesized_is_explicit_tuple(self);
        self.index += 1;
        let values = parse_array(self, ')');
        if let Value::Array(arr) = values {
            if explicit_tuple || arr.len() != 1 {
                return Value::Array(arr);
            }
            return arr.into_iter().next().unwrap();
        }
        values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_char_at_negative_wrap() {
        let mut p = JsonParser::new("abc");
        // index 0
        assert_eq!(p.get_char_at(0), Some('a'));
        assert_eq!(p.get_char_at(1), Some('b'));
        assert_eq!(p.get_char_at(3), None);
        assert_eq!(p.get_char_at(-1), Some('c')); // -1 -> wraps to last
        assert_eq!(p.get_char_at(-4), None); // -4 +3 = -1 still < 0
        // index 1
        p.index = 1;
        assert_eq!(p.get_char_at(0), Some('b'));
        assert_eq!(p.get_char_at(-1), Some('a'));
        assert_eq!(p.get_char_at(-2), Some('c')); // index1-2=-1 -> wraps to 2
        assert_eq!(p.get_char_at(5), None);
    }

    #[test]
    fn skip_whitespaces_mutates_index() {
        for (s, idx, expected) in [("   x", 0, 3), ("\t\n a", 0, 3), ("nows", 0, 0), ("ab  ", 2, 4)] {
            let mut p = JsonParser::new(s);
            p.index = idx;
            p.skip_whitespaces();
            assert_eq!(p.index, expected, "skip_whitespaces({s:?}, idx={idx})");
        }
    }

    #[test]
    fn scroll_whitespaces_returns_offset_without_mutating() {
        for (s, idx, arg, expected) in [("  x", 0, 0, 2), ("x  y", 1, 0, 2), ("abc", 0, 0, 0), ("a   ", 1, 0, 3)] {
            let mut p = JsonParser::new(s);
            p.index = idx;
            let r = p.scroll_whitespaces(arg);
            assert_eq!(r, expected, "scroll_whitespaces({s:?}, idx={idx}, arg={arg})");
            assert_eq!(p.index, idx, "scroll must not mutate index");
        }
    }

    #[test]
    fn skip_to_character_escape_aware() {
        // (string, index, targets, arg, expected_offset)
        let single: Vec<(&str, i64, char, i64, i64)> = vec![
            ("say \"hi\"", 0, '"', 0, 4),
            ("say \"hi\"", 0, '"', 6, 7),
            ("a\\\"b\"c", 0, '"', 0, 4),  // escaped quote skipped (odd backslashes)
            ("a\\\\\"b", 0, '"', 0, 3),   // doubled backslash -> quote not escaped
            ("nofind", 0, '"', 0, 6),     // not found -> distance to end
            ("abc", 1, 'c', 0, 1),
        ];
        for (s, idx, ch, arg, expected) in single {
            let mut p = JsonParser::new(s);
            p.index = idx;
            assert_eq!(
                p.skip_to_char(ch, arg),
                expected,
                "skip_to_char({s:?}, idx={idx}, ch={ch:?}, arg={arg})"
            );
        }
        // list-of-targets form
        let mut p = JsonParser::new("a,b]c");
        assert_eq!(p.skip_to_character(&[']', ','], 0), 1);
        p.index = 0;
    }
}
