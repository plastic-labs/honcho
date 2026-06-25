//! Port of `parse_string.py` (+ its `parse_string_helpers`). The lenient string
//! scanner is the heart of json_repair's heuristics.

use serde_json::Value;

use super::STRING_DELIMITERS;
use super::boolean_null::parse_boolean_or_null;
use super::context::ContextValues;
use super::llm_block::parse_json_llm_block;
use super::parser::JsonParser;
use super::string_helpers::{CommaClass, classify_object_value_comma, update_inline_container_stack};

const INLINE_CONTAINER_OPENERS: [char; 3] = ['[', '{', '('];

fn inline_container_closer(opener: char) -> Option<char> {
    match opener {
        '[' => Some(']'),
        '{' => Some('}'),
        '(' => Some(')'),
        _ => None,
    }
}

/// Port of `StringParseState`.
pub struct StringParseState {
    pub missing_quotes: bool,
    pub doubled_quotes: bool,
    pub lstring_delimiter: char,
    pub rstring_delimiter: char,
    pub string_acc: String,
    pub unmatched_delimiter: bool,
    pub pending_inline_container: bool,
    pub inline_container_stack: Vec<char>,
}

impl Default for StringParseState {
    fn default() -> Self {
        StringParseState {
            missing_quotes: false,
            doubled_quotes: false,
            lstring_delimiter: '"',
            rstring_delimiter: '"',
            string_acc: String::new(),
            unmatched_delimiter: false,
            pending_inline_container: false,
            inline_container_stack: Vec::new(),
        }
    }
}

fn acc_last(s: &str) -> Option<char> {
    s.chars().last()
}

fn is_hex(c: char) -> bool {
    c.is_ascii_hexdigit()
}

enum PrepareResult {
    Direct(Value),
    Continue(StringParseState),
}

/// Port of `parse_string`.
pub fn parse_string(p: &mut JsonParser) -> Value {
    let mut state = match prepare_string_entry(p) {
        PrepareResult::Direct(value) => return value,
        PrepareResult::Continue(state) => state,
    };
    let char = scan_string_body(p, &mut state);
    Value::String(finalize_string_result(p, &mut state, char))
}

fn try_parse_simple_quoted_string(p: &JsonParser) -> Option<String> {
    if p.get_char_at(0) != Some('"') {
        return None;
    }
    let start = p.index + 1;
    let limit = p.len();

    // Find the closing quote, rejecting escapes/newlines (handled by slow path).
    let mut end = start;
    while end < limit {
        let ch = p.json[end as usize];
        if ch == '"' {
            break;
        }
        if ch == '\\' || ch == '\n' || ch == '\r' {
            return None;
        }
        end += 1;
    }
    if end >= limit {
        return None;
    }
    let value: String = p.json[start as usize..end as usize].iter().collect();

    let mut next_index = end + 1;
    while next_index < limit && p.json[next_index as usize].is_whitespace() {
        next_index += 1;
    }
    let next_char = if next_index < limit {
        Some(p.json[next_index as usize])
    } else {
        None
    };

    match p.context.current {
        Some(ContextValues::ObjectKey) => {
            if next_char != Some(':') {
                return None;
            }
        }
        Some(ContextValues::ObjectValue) => {
            if !matches!(next_char, Some(',') | Some('}') | None) {
                return None;
            }
        }
        Some(ContextValues::Array) => {
            if !matches!(next_char, Some(',') | Some(']') | None) {
                return None;
            }
        }
        None => {
            if next_char.is_some() {
                return None;
            }
        }
    }

    // SAFETY: caller treats this as &mut via parse_string; index advance done by caller path.
    // We can't mutate here (p is &), so the caller re-runs with &mut — instead this is called
    // from prepare_string_entry which holds &mut. We return the value and the new index.
    Some(value)
}

fn prepare_string_entry(p: &mut JsonParser) -> PrepareResult {
    let mut char = p.get_char_at(0);
    if matches!(char, Some('#') | Some('/')) {
        return PrepareResult::Direct(super::comment::parse_comment(p));
    }

    while let Some(c) = char {
        if !STRING_DELIMITERS.contains(&c) && !c.is_alphanumeric() {
            p.index += 1;
            char = p.get_char_at(0);
        } else {
            break;
        }
    }

    let char = match char {
        None => return PrepareResult::Direct(Value::String(String::new())),
        Some(c) => c,
    };

    // Fast path: a plain quoted string. (mutate index on success)
    if let Some(value) = try_parse_simple_quoted_string(p) {
        // Reproduce the Python `self.index = end + 1`: recompute end.
        let start = p.index + 1;
        let mut end = start;
        let limit = p.len();
        while end < limit && p.json[end as usize] != '"' {
            end += 1;
        }
        p.index = end + 1;
        return PrepareResult::Direct(Value::String(value));
    }

    let mut state = StringParseState::default();

    if char == '\'' {
        state.lstring_delimiter = '\'';
        state.rstring_delimiter = '\'';
    } else if char == '\u{201c}' {
        state.lstring_delimiter = '\u{201c}';
        state.rstring_delimiter = '\u{201d}';
    } else if char.is_alphanumeric() {
        let lower = char.to_ascii_lowercase();
        if matches!(lower, 't' | 'f' | 'n') && p.context.current != Some(ContextValues::ObjectKey) {
            if let Some(value) = parse_boolean_or_null(p) {
                return PrepareResult::Direct(value);
            }
        }
        state.missing_quotes = true;
    }

    if !state.missing_quotes {
        p.index += 1;
    }

    if p.get_char_at(0) == Some('`') {
        if let Some(ret_val) = parse_json_llm_block(p) {
            return PrepareResult::Direct(ret_val);
        }
    }

    if p.get_char_at(0) == Some(state.lstring_delimiter) {
        let next1 = p.get_char_at(1);
        if (p.context.current == Some(ContextValues::ObjectKey) && next1 == Some(':'))
            || (p.context.current == Some(ContextValues::ObjectValue)
                && matches!(next1, Some(',') | Some('}')))
            || (p.context.current == Some(ContextValues::Array)
                && matches!(next1, Some(',') | Some(']')))
        {
            p.index += 1;
            return PrepareResult::Direct(Value::String(String::new()));
        }
        if next1 == Some(state.lstring_delimiter) {
            // doubled quote then a quote again, ignore it
            return PrepareResult::Direct(Value::String(String::new()));
        }
        let i = p.skip_to_char(state.rstring_delimiter, 1);
        if p.get_char_at(i + 1) == Some(state.rstring_delimiter) {
            state.doubled_quotes = true;
            p.index += 1;
        } else {
            let i = p.scroll_whitespaces(1);
            let next_c = p.get_char_at(i);
            if matches!(next_c, Some(c) if STRING_DELIMITERS.contains(&c) || c == '{' || c == '[') {
                p.index += 1;
                return PrepareResult::Direct(Value::String(String::new()));
            }
            if !matches!(next_c, Some(',') | Some(']') | Some('}')) {
                p.index += 1;
            }
        }
    }

    PrepareResult::Continue(state)
}

fn append_literal_char(p: &mut JsonParser, state: &mut StringParseState, current_char: char) -> Option<char> {
    state.string_acc.push(current_char);
    p.index += 1;
    p.get_char_at(0)
}

fn normalize_escape_sequence(
    p: &mut JsonParser,
    state: &mut StringParseState,
    char: char,
) -> (bool, Option<char>) {
    if char == state.rstring_delimiter || matches!(char, 't' | 'n' | 'r' | 'b' | '\\') {
        state.string_acc.pop();
        let replacement = match char {
            't' => '\t',
            'n' => '\n',
            'r' => '\r',
            'b' => '\u{0008}',
            other => other,
        };
        state.string_acc.push(replacement);
        p.index += 1;
        let mut next_char = p.get_char_at(0);
        while let Some(nc) = next_char {
            if acc_last(&state.string_acc) == Some('\\') && (nc == state.rstring_delimiter || nc == '\\') {
                state.string_acc.pop();
                state.string_acc.push(nc);
                p.index += 1;
                next_char = p.get_char_at(0);
            } else {
                break;
            }
        }
        return (true, next_char);
    }
    if char == 'u' || char == 'x' {
        let num_chars = if char == 'u' { 4 } else { 2 };
        let next_chars = p.substr(p.index + 1, p.index + 1 + num_chars);
        if next_chars.chars().count() == num_chars as usize && next_chars.chars().all(is_hex) {
            if let Ok(code) = u32::from_str_radix(&next_chars, 16) {
                if let Some(decoded) = char::from_u32(code) {
                    state.string_acc.pop();
                    state.string_acc.push(decoded);
                    p.index += 1 + num_chars;
                    return (true, p.get_char_at(0));
                }
            }
        }
    } else if STRING_DELIMITERS.contains(&char) && char != state.rstring_delimiter {
        state.string_acc.pop();
        state.string_acc.push(char);
        p.index += 1;
        return (true, p.get_char_at(0));
    }
    (false, Some(char))
}

fn matching_string_delimiter(delimiter: char) -> char {
    if delimiter == '\u{201c}' {
        '\u{201d}'
    } else {
        delimiter
    }
}

fn bare_key_is_followed_by_colon(p: &JsonParser, mut key_idx: i64) -> bool {
    let key_char = p.get_char_at(key_idx);
    if !matches!(key_char, Some(c) if c.is_alphanumeric() || c == '_') {
        return false;
    }
    loop {
        match p.get_char_at(key_idx) {
            Some(c) if c.is_alphanumeric() || c == '_' || c == '-' => key_idx += 1,
            _ => break,
        }
    }
    key_idx = p.scroll_whitespaces(key_idx);
    p.get_char_at(key_idx) == Some(':')
}

fn object_member_starts_at(p: &JsonParser, next_member_idx: i64) -> bool {
    let next_member = p.get_char_at(next_member_idx);
    if matches!(next_member, Some('}') | None) {
        return false;
    }
    let next_member = next_member.unwrap();
    if STRING_DELIMITERS.contains(&next_member) {
        let key_end_delimiter = matching_string_delimiter(next_member);
        let key_end_idx = p.skip_to_char(key_end_delimiter, next_member_idx + 1);
        if p.get_char_at(key_end_idx) != Some(key_end_delimiter) {
            return false;
        }
        let after_key_idx = p.scroll_whitespaces(key_end_idx + 1);
        return p.get_char_at(after_key_idx) == Some(':');
    }
    if next_member.is_alphanumeric() || next_member == '_' {
        return bare_key_is_followed_by_colon(p, next_member_idx);
    }
    false
}

fn scroll_comment_prefixed_member_start(p: &JsonParser, mut idx: i64) -> i64 {
    idx = p.scroll_whitespaces(idx);
    loop {
        let char = p.get_char_at(idx);
        if char == Some('#') {
            let mut c = char;
            while let Some(ch) = c {
                if ch == '\n' || ch == '\r' {
                    break;
                }
                idx += 1;
                c = p.get_char_at(idx);
            }
            idx = p.scroll_whitespaces(idx);
            continue;
        }
        if char == Some('/') {
            let next_char = p.get_char_at(idx + 1);
            if next_char == Some('/') {
                idx += 2;
                let mut c = p.get_char_at(idx);
                while let Some(ch) = c {
                    if ch == '\n' || ch == '\r' {
                        break;
                    }
                    idx += 1;
                    c = p.get_char_at(idx);
                }
                idx = p.scroll_whitespaces(idx);
                continue;
            }
            if next_char == Some('*') {
                idx += 2;
                loop {
                    let c = p.get_char_at(idx);
                    if c.is_none() {
                        return idx;
                    }
                    if c == Some('*') && p.get_char_at(idx + 1) == Some('/') {
                        idx += 2;
                        break;
                    }
                    idx += 1;
                }
                idx = p.scroll_whitespaces(idx);
                continue;
            }
        }
        return idx;
    }
}

fn quoted_object_member_follows(p: &JsonParser, quote_idx: i64) -> bool {
    let comma_idx = p.scroll_whitespaces(quote_idx + 1);
    if p.get_char_at(comma_idx) != Some(',') {
        return false;
    }
    let next_member_idx = scroll_comment_prefixed_member_start(p, comma_idx + 1);
    object_member_starts_at(p, next_member_idx)
}

fn post_fence_container_starts_next_member(p: &JsonParser, container_end_idx: i64) -> bool {
    let after_container_idx = p.scroll_whitespaces(container_end_idx);
    let after_container = p.get_char_at(after_container_idx);
    if matches!(after_container, Some('}') | None) {
        return true;
    }
    if after_container != Some(',') {
        return false;
    }
    let next_member_idx = scroll_comment_prefixed_member_start(p, after_container_idx + 1);
    matches!(p.get_char_at(next_member_idx), Some('}') | None)
        || object_member_starts_at(p, next_member_idx)
}

fn starts_nested_inline_container(p: &JsonParser, idx: i64) -> bool {
    let opening_delimiter = p.get_char_at(idx);
    let mut prev_idx = idx - 1;
    while prev_idx >= 0 {
        let prev_char = p.get_char_at(prev_idx);
        let prev_char = match prev_char {
            None => return true,
            Some(c) => c,
        };
        if !prev_char.is_whitespace() {
            if INLINE_CONTAINER_OPENERS.contains(&prev_char) {
                return true;
            }
            if prev_char != ',' && prev_char != ':' {
                return false;
            }
            let next_idx = p.scroll_whitespaces(idx + 1);
            let next_char = p.get_char_at(next_idx);
            if opening_delimiter == Some('[') || opening_delimiter == Some('(') {
                return matches!(next_char, Some(c) if c == ']'
                    || c == ')'
                    || STRING_DELIMITERS.contains(&c)
                    || c == '-'
                    || INLINE_CONTAINER_OPENERS.contains(&c)
                    || c == 't'
                    || c == 'f'
                    || c == 'n'
                    || c.is_ascii_digit());
            }
            if opening_delimiter != Some('{') {
                return false;
            }
            if matches!(next_char, Some('}')) || matches!(next_char, Some(c) if STRING_DELIMITERS.contains(&c)) {
                return true;
            }
            return prev_char == ':' && bare_key_is_followed_by_colon(p, next_idx);
        }
        prev_idx -= 1;
    }
    true
}

fn skip_inline_container(p: &JsonParser, idx: i64) -> Option<i64> {
    let opening_delimiter = match p.get_char_at(idx) {
        Some(c) if inline_container_closer(c).is_some() => c,
        _ => return Some(idx),
    };
    let mut stack = vec![inline_container_closer(opening_delimiter).unwrap()];
    let mut i = idx + 1;
    while !stack.is_empty() {
        let char = p.get_char_at(i);
        let char = char?;
        if STRING_DELIMITERS.contains(&char) {
            let end_delimiter = matching_string_delimiter(char);
            i = p.skip_to_char(end_delimiter, i + 1);
            if p.get_char_at(i) != Some(end_delimiter) {
                return None;
            }
        } else if inline_container_closer(char).is_some() && starts_nested_inline_container(p, i) {
            stack.push(inline_container_closer(char).unwrap());
        } else if Some(char) == stack.last().copied() {
            stack.pop();
            if stack.is_empty() {
                return Some(i + 1);
            }
        }
        i += 1;
    }
    None
}

fn only_whitespace_until(p: &JsonParser, end: i64) -> bool {
    let mut j = 1;
    while j < end {
        let c = p.get_char_at(j);
        if matches!(c, Some(ch) if !ch.is_whitespace()) {
            return false;
        }
        j += 1;
    }
    true
}

fn brace_before_code_fence_belongs_to_string(p: &mut JsonParser, state: &mut StringParseState, fence_idx: i64) -> bool {
    let mut quote_search_idx = fence_idx + 3;
    let next_content_idx = scroll_comment_prefixed_member_start(p, quote_search_idx);
    let mut keep_post_fence_container = false;
    if matches!(p.get_char_at(next_content_idx), Some(c) if INLINE_CONTAINER_OPENERS.contains(&c)) {
        if let Some(container_end_idx) = skip_inline_container(p, next_content_idx) {
            if post_fence_container_starts_next_member(p, container_end_idx) {
                return false;
            }
            keep_post_fence_container = true;
            quote_search_idx = container_end_idx;
        }
    }

    let mut quote_idx = p.skip_to_char(state.rstring_delimiter, quote_search_idx);
    while p.get_char_at(quote_idx) == Some(state.rstring_delimiter) {
        let after_quote_idx = p.scroll_whitespaces(quote_idx + 1);
        let after_quote = p.get_char_at(after_quote_idx);
        if matches!(after_quote, Some(',') | Some('}') | Some(']') | None) {
            if keep_post_fence_container {
                state.pending_inline_container = true;
            }
            return true;
        }
        quote_idx = p.skip_to_char(state.rstring_delimiter, quote_idx + 1);
    }
    false
}

fn handle_right_delimiter_candidate(
    p: &mut JsonParser,
    state: &mut StringParseState,
    char: char,
) -> (bool, Option<char>, bool) {
    if state.doubled_quotes && p.get_char_at(1) == Some(state.rstring_delimiter) {
        p.index += 1;
        return (true, Some(char), false);
    }

    if state.missing_quotes && p.context.current == Some(ContextValues::ObjectValue) {
        let mut i = 1;
        let mut next_c = p.get_char_at(i);
        while let Some(nc) = next_c {
            if nc == state.rstring_delimiter || nc == state.lstring_delimiter {
                break;
            }
            i += 1;
            next_c = p.get_char_at(i);
        }
        if next_c.is_some() {
            i += 1;
            i = p.scroll_whitespaces(i);
            if p.get_char_at(i) == Some(':') {
                p.index -= 1;
                let next_char = p.get_char_at(0);
                return (false, next_char, true);
            }
        }
        return (false, Some(char), false);
    }

    if state.unmatched_delimiter {
        state.unmatched_delimiter = false;
        let next_char = append_literal_char(p, state, char);
        return (true, next_char, false);
    }

    let mut i = 1;
    let mut next_c = p.get_char_at(i);
    let mut check_comma_in_object_value = true;
    while let Some(nc) = next_c {
        if nc == state.rstring_delimiter || nc == state.lstring_delimiter {
            break;
        }
        if check_comma_in_object_value && nc.is_alphabetic() {
            check_comma_in_object_value = false;
        }
        if (p.context.contains(ContextValues::ObjectKey) && (nc == ':' || nc == '}'))
            || (p.context.contains(ContextValues::ObjectValue) && nc == '}')
            || (p.context.contains(ContextValues::Array) && (nc == ']' || nc == ','))
            || (check_comma_in_object_value
                && p.context.current == Some(ContextValues::ObjectValue)
                && nc == ',')
        {
            break;
        }
        i += 1;
        next_c = p.get_char_at(i);
    }

    if next_c == Some(',') && p.context.current == Some(ContextValues::ObjectValue) {
        i += 1;
        i = p.skip_to_char(state.rstring_delimiter, i);
        i += 1;
        i = p.scroll_whitespaces(i);
        next_c = p.get_char_at(i);
        if matches!(next_c, Some('}') | Some(',')) {
            let next_char = append_literal_char(p, state, char);
            return (true, next_char, false);
        }
    } else if next_c == Some(state.rstring_delimiter) && p.get_char_at(i - 1) != Some('\\') {
        if only_whitespace_until(p, i)
            && !(p.context.current == Some(ContextValues::ObjectValue)
                && quoted_object_member_follows(p, i))
        {
            return (false, Some(char), true);
        }
        if p.context.current == Some(ContextValues::ObjectValue) {
            if quoted_object_member_follows(p, i) {
                let next_char = append_literal_char(p, state, char);
                return (true, next_char, false);
            }
            i = p.skip_to_char(state.rstring_delimiter, i + 1);
            i += 1;
            next_c = p.get_char_at(i);
            while let Some(nc) = next_c {
                if nc == ':' {
                    break;
                }
                if matches!(nc, ',' | ']' | '}')
                    || (nc == state.rstring_delimiter && p.get_char_at(i - 1) != Some('\\'))
                {
                    break;
                }
                i += 1;
                next_c = p.get_char_at(i);
            }
            if next_c != Some(':') {
                state.unmatched_delimiter = !state.unmatched_delimiter;
                let next_char = append_literal_char(p, state, char);
                return (true, next_char, false);
            }
        } else if p.context.current == Some(ContextValues::Array) {
            let mut even_delimiters = next_c == Some(state.rstring_delimiter);
            while next_c == Some(state.rstring_delimiter) {
                i = p.skip_to_character(&[state.rstring_delimiter, ']'], i + 1);
                next_c = p.get_char_at(i);
                if next_c != Some(state.rstring_delimiter) {
                    even_delimiters = false;
                    break;
                }
                i = p.skip_to_character(&[state.rstring_delimiter, ']'], i + 1);
                next_c = p.get_char_at(i);
            }
            if even_delimiters {
                state.unmatched_delimiter = !state.unmatched_delimiter;
                let next_char = append_literal_char(p, state, char);
                return (true, next_char, false);
            }
            return (false, Some(char), true);
        } else if p.context.current == Some(ContextValues::ObjectKey) {
            let next_char = append_literal_char(p, state, char);
            return (true, next_char, false);
        }
    }

    (false, Some(char), false)
}

fn scan_string_body(p: &mut JsonParser, state: &mut StringParseState) -> Option<char> {
    let mut char = p.get_char_at(0);
    while let Some(c) = char {
        if c == state.rstring_delimiter {
            break;
        }

        if state.missing_quotes {
            if p.context.current == Some(ContextValues::ObjectKey) && (c == ':' || c.is_whitespace()) {
                break;
            }
            if p.context.current == Some(ContextValues::Array) && (c == ']' || c == ',') {
                break;
            }
        }

        if state.pending_inline_container && INLINE_CONTAINER_OPENERS.contains(&c) {
            if let Some(container_end_idx) = skip_inline_container(p, 0) {
                state.pending_inline_container = false;
                state.inline_container_stack.clear();
                let chunk = p.substr(p.index, p.index + container_end_idx);
                state.string_acc.push_str(&chunk);
                p.index += container_end_idx;
                char = p.get_char_at(0);
                continue;
            }
        }

        if !p.stream_stable
            && p.context.current == Some(ContextValues::ObjectValue)
            && c == ','
            && !state.pending_inline_container
            && state.inline_container_stack.is_empty()
        {
            let comma_classification = classify_object_value_comma(p);
            if comma_classification == CommaClass::Member {
                break;
            }
            state.pending_inline_container = comma_classification == CommaClass::Container;
            char = append_literal_char(p, state, c);
            continue;
        }

        let (pending, keep_inline_container_char) =
            update_inline_container_stack(c, state.pending_inline_container, &mut state.inline_container_stack);
        state.pending_inline_container = pending;
        if keep_inline_container_char {
            char = append_literal_char(p, state, c);
            continue;
        }

        if !p.stream_stable
            && p.context.current == Some(ContextValues::ObjectValue)
            && c == '}'
            && (state.string_acc.is_empty() || acc_last(&state.string_acc) != Some(state.rstring_delimiter))
        {
            let mut rstring_delimiter_missing = true;
            p.skip_whitespaces();
            if p.get_char_at(1) == Some('\\') {
                rstring_delimiter_missing = false;
            }
            let mut i = p.skip_to_char(state.rstring_delimiter, 1);
            let mut next_c = p.get_char_at(i);
            if next_c.is_some() {
                i += 1;
                i = p.scroll_whitespaces(i);
                next_c = p.get_char_at(i);
                if matches!(next_c, None | Some(',') | Some('}')) {
                    rstring_delimiter_missing = false;
                } else {
                    i = p.skip_to_char(state.lstring_delimiter, i);
                    next_c = p.get_char_at(i);
                    if next_c.is_none() {
                        rstring_delimiter_missing = false;
                    } else {
                        i = p.scroll_whitespaces(i + 1);
                        next_c = p.get_char_at(i);
                        if matches!(next_c, Some(ch) if ch != ':') {
                            rstring_delimiter_missing = false;
                        }
                    }
                }
            } else {
                i = p.skip_to_char(':', 1);
                next_c = p.get_char_at(i);
                if next_c.is_some() {
                    break;
                }
                i = p.scroll_whitespaces(1);
                let j = p.skip_to_char('}', i);
                if j - i > 1 {
                    rstring_delimiter_missing = false;
                } else if p.get_char_at(j).is_some() {
                    for ch in state.string_acc.chars().rev() {
                        if ch == '{' {
                            rstring_delimiter_missing = false;
                            break;
                        }
                    }
                }
            }
            if rstring_delimiter_missing {
                break;
            }
        }

        if !p.stream_stable
            && c == ']'
            && p.context.contains(ContextValues::Array)
            && (state.string_acc.is_empty() || acc_last(&state.string_acc) != Some(state.rstring_delimiter))
        {
            let i = p.skip_to_char(state.rstring_delimiter, 0);
            if p.get_char_at(i).is_none() {
                break;
            }
        }

        if p.context.current == Some(ContextValues::ObjectValue) && c == '}' {
            let i = p.scroll_whitespaces(1);
            let next_c = p.get_char_at(i);
            if next_c == Some('`') && p.get_char_at(i + 1) == Some('`') && p.get_char_at(i + 2) == Some('`') {
                if brace_before_code_fence_belongs_to_string(p, state, i) {
                    char = append_literal_char(p, state, c);
                    continue;
                }
                break;
            }
            if next_c.is_none() {
                break;
            }
        }

        state.string_acc.push(c);
        p.index += 1;
        char = p.get_char_at(0);
        if char.is_none() {
            if p.stream_stable && acc_last(&state.string_acc) == Some('\\') {
                state.string_acc.pop();
            }
            break;
        }
        if acc_last(&state.string_acc) == Some('\\') {
            let (handled_escape, new_char) = normalize_escape_sequence(p, state, char.unwrap());
            char = new_char;
            if handled_escape {
                continue;
            }
        }
        if char == Some(':')
            && !state.missing_quotes
            && p.context.current == Some(ContextValues::ObjectKey)
        {
            let mut i = p.skip_to_char(state.lstring_delimiter, 1);
            let mut next_c = p.get_char_at(i);
            if next_c.is_some() {
                i += 1;
                i = p.skip_to_char(state.rstring_delimiter, i);
                next_c = p.get_char_at(i);
                if next_c.is_some() {
                    i += 1;
                    i = p.scroll_whitespaces(i);
                    let ch = p.get_char_at(i);
                    if matches!(ch, Some(',') | Some('}')) {
                        break;
                    }
                }
            } else {
                break;
            }
        }
        if char == Some(state.rstring_delimiter)
            && !state.string_acc.is_empty()
            && acc_last(&state.string_acc) != Some('\\')
        {
            let cur = char.unwrap();
            let (handled_delimiter, new_char, should_break) = handle_right_delimiter_candidate(p, state, cur);
            char = new_char;
            if should_break {
                break;
            }
            if handled_delimiter {
                continue;
            }
        }
    }
    char
}

fn finalize_string_result(p: &mut JsonParser, state: &mut StringParseState, char: Option<char>) -> String {
    if let Some(c) = char {
        if state.missing_quotes
            && p.context.current == Some(ContextValues::ObjectKey)
            && c.is_whitespace()
        {
            p.skip_whitespaces();
            if !matches!(p.get_char_at(0), Some(':') | Some(',')) {
                return String::new();
            }
        }
    }

    if char != Some(state.rstring_delimiter) {
        if !p.stream_stable {
            let trimmed = state.string_acc.trim_end().to_string();
            state.string_acc = trimmed;
        }
    } else {
        p.index += 1;
    }

    if !p.stream_stable
        && (state.missing_quotes || acc_last(&state.string_acc) == Some('\n'))
    {
        let trimmed = state.string_acc.trim_end().to_string();
        state.string_acc = trimmed;
    }

    std::mem::take(&mut state.string_acc)
}
