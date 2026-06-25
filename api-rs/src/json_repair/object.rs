//! Port of `parse_object.py` (no-schema path).

use serde_json::{Map, Value};

use super::STRING_DELIMITERS;
use super::array::parse_array;
use super::context::ContextValues;
use super::parser::JsonParser;
use super::string::parse_string;

/// Insertion-ordered object, mirroring Python `dict` semantics the parser
/// relies on (overwrite-in-place keeps position; `list(obj.keys())[-1]` is the
/// last *inserted* key). Converted to a `serde_json::Map` only at the end.
#[derive(Default)]
struct OrderedObj {
    entries: Vec<(String, Value)>,
}

impl OrderedObj {
    fn new() -> Self {
        OrderedObj { entries: Vec::new() }
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn contains_key(&self, key: &str) -> bool {
        self.entries.iter().any(|(k, _)| k == key)
    }

    fn get(&self, key: &str) -> Option<&Value> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        self.entries.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Python `obj[key] = value` (overwrite in place, else append).
    fn insert(&mut self, key: String, value: Value) {
        if let Some(slot) = self.entries.iter_mut().find(|(k, _)| *k == key) {
            slot.1 = value;
        } else {
            self.entries.push((key, value));
        }
    }

    fn last_key(&self) -> Option<String> {
        self.entries.last().map(|(k, _)| k.clone())
    }

    /// Python `dict.update(other)`.
    fn update(&mut self, other: OrderedObj) {
        for (k, v) in other.entries {
            self.insert(k, v);
        }
    }

    fn into_value(self) -> Value {
        let mut map = Map::new();
        for (k, v) in self.entries {
            map.insert(k, v);
        }
        Value::Object(map)
    }
}

/// Port of `parse_object`.
pub fn parse_object(p: &mut JsonParser) -> Value {
    let mut obj = OrderedObj::new();
    let start_index = p.index;
    let parsing_object_value = p.context.current == Some(ContextValues::ObjectValue);

    while p.get_char_at(0).unwrap_or('}') != '}' {
        p.skip_whitespaces();

        if p.get_char_at(0) == Some(':') {
            // a ':' before a key, ignore
            p.index += 1;
        }

        let (key, rollback_index) = parse_object_key(p, &mut obj);
        if p.context.contains(ContextValues::Array) && obj.contains_key(&key) {
            // strict mode skipped (strict == false)
            if !parsing_object_value && should_split_duplicate_object(p, rollback_index) {
                split_object_on_duplicate_key(p, rollback_index);
                break;
            }
        }

        p.skip_whitespaces();
        if p.get_char_at(0).unwrap_or('}') == '}' {
            continue;
        }

        p.skip_whitespaces();
        // a missing ':' after a key is tolerated (strict skipped)
        p.index += 1;

        let value = parse_object_value(p);

        // strict empty-value check skipped; drop_property is always false.
        obj.insert(key, value);

        if matches!(p.get_char_at(0), Some(',') | Some('\'') | Some('"')) {
            p.index += 1;
        }
        if p.get_char_at(0) == Some(']') && p.context.contains(ContextValues::Array) {
            // closing array bracket: close the object and roll back
            p.index -= 1;
            break;
        }
        p.skip_whitespaces();
    }

    p.index += 1;

    if let Some(repaired) = repair_empty_object_result(p, &obj, start_index) {
        return repaired;
    }

    complete_object_parse(p, obj)
}

fn parse_object_key(p: &mut JsonParser, obj: &mut OrderedObj) -> (String, i64) {
    let mut key = String::new();
    let mut rollback_index = p.index;
    p.context.set(ContextValues::ObjectKey);

    while p.get_char_at(0).is_some() {
        rollback_index = p.index;
        if p.get_char_at(0) == Some('[') && key.is_empty() && merge_object_array_continuation(p, obj) {
            continue;
        }

        let raw_key = parse_string(p);
        key = match raw_key {
            Value::String(s) => s,
            _ => String::new(),
        };
        if key.is_empty() {
            p.skip_whitespaces();
        }
        if !key.is_empty() || (key.is_empty() && matches!(p.get_char_at(0), Some(':') | Some('}'))) {
            // strict empty-key check skipped
            break;
        }
    }

    p.context.reset();
    (key, rollback_index)
}

fn should_split_duplicate_object(p: &JsonParser, rollback_index: i64) -> bool {
    let mut lookback_idx = rollback_index - p.index - 1;
    let mut prev_non_whitespace = p.get_char_at(lookback_idx);
    while matches!(prev_non_whitespace, Some(c) if c.is_whitespace()) {
        lookback_idx -= 1;
        prev_non_whitespace = p.get_char_at(lookback_idx);
    }
    let key_start_char = p.get_char_at(rollback_index - p.index);
    let next_non_whitespace = p.get_char_at(p.scroll_whitespaces(0));
    !(matches!(key_start_char, Some(c) if STRING_DELIMITERS.contains(&c))
        && prev_non_whitespace == Some(',')
        && next_non_whitespace == Some(':'))
}

fn split_object_on_duplicate_key(p: &mut JsonParser, rollback_index: i64) {
    p.index = rollback_index - 1;
    // json_str[:index+1] + "{" + json_str[index+1:]
    p.splice(p.index + 1, p.index + 1, "{");
}

fn parse_object_value(p: &mut JsonParser) -> Value {
    p.context.set(ContextValues::ObjectValue);
    p.skip_whitespaces();
    let char = p.get_char_at(0);
    let result = if matches!(char, Some(',') | Some('}')) {
        // stray ',' or '}' — empty value (schema repairer absent)
        Value::String(String::new())
    } else {
        p.parse_json()
    };
    p.context.reset();
    result
}

fn merge_object_array_continuation(p: &mut JsonParser, obj: &mut OrderedObj) -> bool {
    let prev_key = match obj.last_key() {
        Some(k) => k,
        None => return false,
    };
    if !matches!(obj.get(&prev_key), Some(Value::Array(_))) || p.strict {
        return false;
    }

    p.index += 1;
    let new_array = parse_array(p, ']');

    if let Value::Array(new_items) = &new_array {
        if let Some(Value::Array(prev_value)) = obj.get_mut(&prev_key) {
            let list_lengths: Vec<usize> = prev_value
                .iter()
                .filter_map(|it| if let Value::Array(a) = it { Some(a.len()) } else { None })
                .collect();
            let expected_len = if !list_lengths.is_empty() && list_lengths.iter().all(|&l| l == list_lengths[0]) {
                Some(list_lengths[0])
            } else {
                None
            };

            // Python `if expected_len:` is truthy only for a present, non-zero length.
            if let Some(expected_len) = expected_len.filter(|&l| l != 0) {
                let mut tail: Vec<Value> = Vec::new();
                while let Some(last) = prev_value.last() {
                    if !matches!(last, Value::Array(_)) {
                        tail.push(prev_value.pop().unwrap());
                    } else {
                        break;
                    }
                }
                if !tail.is_empty() {
                    tail.reverse();
                    if tail.len().is_multiple_of(expected_len) {
                        let mut i = 0;
                        while i < tail.len() {
                            let chunk: Vec<Value> = tail[i..i + expected_len].to_vec();
                            prev_value.push(Value::Array(chunk));
                            i += expected_len;
                        }
                    } else {
                        prev_value.extend(tail);
                    }
                }
                if !new_items.is_empty() {
                    if new_items.iter().all(|it| matches!(it, Value::Array(_))) {
                        prev_value.extend(new_items.clone());
                    } else {
                        prev_value.push(Value::Array(new_items.clone()));
                    }
                }
            } else {
                // None or zero expected_len -> flatten a single inner list, else extend.
                if new_items.len() == 1 {
                    if let Value::Array(inner) = &new_items[0] {
                        prev_value.extend(inner.clone());
                    } else {
                        prev_value.extend(new_items.clone());
                    }
                } else {
                    prev_value.extend(new_items.clone());
                }
            }
        }
    }

    p.skip_whitespaces();
    if p.get_char_at(0) == Some(',') {
        p.index += 1;
    }
    p.skip_whitespaces();
    true
}

fn classify_empty_object_repair(p: &JsonParser, start_index: i64) -> (&'static str, Option<String>) {
    let attempted_object = p.substr(start_index - 1, p.index + 1);
    // body = attempted_object[1:].removesuffix("}").lstrip()
    let mut body: String = attempted_object.chars().skip(1).collect();
    if let Some(stripped) = body.strip_suffix('}') {
        body = stripped.to_string();
    }
    let body = body.trim_start().to_string();
    if body.is_empty() {
        return ("keep", None);
    }

    let starts_escaped_dq = body.starts_with("\\\"") && body.contains("\\\":");
    let starts_escaped_sq = body.starts_with("\\'") && body.contains("\\':");
    if starts_escaped_dq || starts_escaped_sq {
        let normalized = attempted_object.replace("\\\"", "\"").replace("\\'", "'");
        return ("object", Some(normalized));
    }

    let mut in_quote: Option<char> = None;
    let mut backslashes = 0u32;
    for ch in body.chars() {
        if ch == '\\' {
            backslashes += 1;
            continue;
        }
        if let Some(q) = in_quote {
            if ch == q && backslashes.is_multiple_of(2) {
                in_quote = None;
            }
        } else if STRING_DELIMITERS.contains(&ch) && backslashes.is_multiple_of(2) {
            in_quote = Some(ch);
        } else if ch == ':' && backslashes.is_multiple_of(2) {
            return ("keep", None);
        }
        backslashes = 0;
    }
    // schema_set_object branch requires a salvage schema (absent here).
    ("array", None)
}

fn repair_empty_object_result(p: &mut JsonParser, obj: &OrderedObj, start_index: i64) -> Option<Value> {
    if !obj.is_empty() || p.index - start_index <= 2 {
        return None;
    }
    // strict mode skipped

    let (repair_kind, normalized) = classify_empty_object_repair(p, start_index);
    match repair_kind {
        "object" => {
            let normalized = normalized.unwrap();
            let end_index = p.index + 1;
            p.splice(start_index - 1, end_index, &normalized);
            p.index = start_index;
            p.context.set(ContextValues::ObjectKey);
            let repaired_value = parse_object(p);
            p.context.reset();
            p.deferred_contexts.push(ContextValues::ObjectKey);
            Some(repaired_value)
        }
        "array" => {
            p.index = start_index;
            p.context.set(ContextValues::ObjectKey);
            let repaired_array = parse_array(p, ']');
            p.context.reset();
            p.deferred_contexts.push(ContextValues::ObjectKey);
            Some(repaired_array)
        }
        _ => None,
    }
}

fn complete_object_parse(p: &mut JsonParser, mut obj: OrderedObj) -> Value {
    if !p.context.empty {
        if p.get_char_at(0) == Some('}')
            && !matches!(p.context.current, Some(ContextValues::ObjectKey) | Some(ContextValues::ObjectValue))
        {
            // extra closing brace, skip it
            p.index += 1;
        }
        return obj.into_value();
    }

    p.skip_whitespaces();
    if p.get_char_at(0) == Some(',') {
        p.index += 1;
        p.skip_whitespaces();
        if matches!(p.get_char_at(0), Some(c) if STRING_DELIMITERS.contains(&c)) {
            // strict skipped: additional key-value pairs after closing brace
            let additional = parse_object(p);
            if let Value::Object(map) = additional {
                let mut extra = OrderedObj::new();
                for (k, v) in map {
                    extra.entries.push((k, v));
                }
                obj.update(extra);
            }
        }
    }

    obj.into_value()
}
