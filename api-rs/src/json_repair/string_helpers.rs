//! Port of `parse_string_helpers/object_value_context.py`.

use super::STRING_DELIMITERS;
use super::parser::JsonParser;

/// Port of `ObjectValueCommaClassification`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommaClass {
    Container,
    Member,
    StringValue,
}

/// Port of `classify_object_value_comma`. The parser cursor points at the comma
/// being classified; reads only via cursor primitives (no mutation).
pub fn classify_object_value_comma(parser: &JsonParser) -> CommaClass {
    let next_idx = parser.scroll_whitespaces(1);
    let next_c = parser.get_char_at(next_idx);

    // next_c in ["}", None]
    if next_c == Some('}') || next_c.is_none() {
        return CommaClass::Member;
    }
    let next_c = next_c.unwrap();

    if STRING_DELIMITERS.contains(&next_c) {
        let key_end_idx = parser.skip_to_char(next_c, next_idx + 1);
        if parser.get_char_at(key_end_idx).is_none() {
            return CommaClass::StringValue;
        }
        let key_end_idx = parser.scroll_whitespaces(key_end_idx + 1);
        return if parser.get_char_at(key_end_idx) == Some(':') {
            CommaClass::Member
        } else {
            CommaClass::StringValue
        };
    }

    if next_c == '`' {
        let mut bare_key_idx = next_idx + 1;
        loop {
            match parser.get_char_at(bare_key_idx) {
                Some(c) if c.is_alphanumeric() || c == '_' || c == '-' => bare_key_idx += 1,
                _ => break,
            }
        }
        let bare_key_idx = parser.scroll_whitespaces(bare_key_idx);
        return if parser.get_char_at(bare_key_idx) == Some(':') {
            CommaClass::Member
        } else {
            CommaClass::StringValue
        };
    }

    if next_c.is_alphanumeric() || next_c == '_' {
        let mut bare_key_idx = next_idx;
        loop {
            match parser.get_char_at(bare_key_idx) {
                Some(c) if c.is_alphanumeric() || c == '_' || c == '-' => bare_key_idx += 1,
                _ => break,
            }
        }
        let bare_key_idx = parser.scroll_whitespaces(bare_key_idx);
        if parser.get_char_at(bare_key_idx) == Some(':') {
            return CommaClass::Member;
        }
    }

    let next_quote_idx = parser.skip_to_character(&STRING_DELIMITERS, next_idx);
    let next_quote = parser.get_char_at(next_quote_idx);
    if next_quote.is_none() {
        return CommaClass::StringValue;
    }
    let next_quote = next_quote.unwrap();

    let container_idx = parser.skip_to_character(&['{', '['], next_idx);
    let container_c = parser.get_char_at(container_idx);
    if (container_c == Some('{') || container_c == Some('[')) && container_idx < next_quote_idx {
        return CommaClass::Container;
    }

    let key_end_idx = parser.skip_to_char(next_quote, next_quote_idx + 1);
    if parser.get_char_at(key_end_idx).is_none() {
        return CommaClass::StringValue;
    }
    let key_end_idx = parser.scroll_whitespaces(key_end_idx + 1);
    if parser.get_char_at(key_end_idx) == Some(':') {
        CommaClass::Member
    } else {
        CommaClass::StringValue
    }
}

/// Port of `update_inline_container_stack`. Returns `(pending_inline_container,
/// keep_inline_container_char)` and mutates the stack in place.
pub fn update_inline_container_stack(
    char: char,
    pending_inline_container: bool,
    inline_container_stack: &mut Vec<char>,
) -> (bool, bool) {
    if char == '{' || char == '[' {
        if pending_inline_container {
            inline_container_stack.push(char);
            return (false, false);
        }
        if !inline_container_stack.is_empty() {
            inline_container_stack.push(char);
        }
    }

    if let Some(&top) = inline_container_stack.last() {
        if (char == '}' && top == '{') || (char == ']' && top == '[') {
            inline_container_stack.pop();
            return (pending_inline_container, true);
        }
    }

    (pending_inline_container, false)
}

#[cfg(test)]
mod tests {
    use super::super::context::ContextValues;
    use super::*;

    fn parser_at_comma(s: &str) -> JsonParser {
        let comma = s.find(',').unwrap() as i64;
        let mut p = JsonParser::new(s);
        p.index = comma;
        // classify reads only cursor primitives, but set realistic context anyway.
        p.context.set(ContextValues::ObjectKey);
        p.context.set(ContextValues::ObjectValue);
        p
    }

    #[test]
    fn classify_object_value_comma_golden() {
        let cases: Vec<(&str, CommaClass)> = vec![
            ("x, \"key\": 1}", CommaClass::Member),
            ("x, \"more text\"}", CommaClass::StringValue),
            ("x, [1,2]}", CommaClass::StringValue),
            ("x, {\"a\":1}}", CommaClass::Container),
            ("x, bare: 1}", CommaClass::Member),
            ("x, more, words}", CommaClass::StringValue),
            ("x,}", CommaClass::Member),
            ("x, \"noclose", CommaClass::StringValue),
        ];
        for (input, expected) in cases {
            let p = parser_at_comma(input);
            assert_eq!(classify_object_value_comma(&p), expected, "classify({input:?})");
        }
    }

    #[test]
    fn update_inline_container_stack_golden() {
        // (char, pending, stack_before) -> ((pending, keep), stack_after)
        let cases: Vec<(char, bool, Vec<char>, (bool, bool), Vec<char>)> = vec![
            ('{', true, vec![], (false, false), vec!['{']),
            ('[', false, vec![], (false, false), vec![]),
            ('{', false, vec!['['], (false, false), vec!['[', '{']),
            ('}', false, vec!['{'], (false, true), vec![]),
            (']', false, vec!['['], (false, true), vec![]),
            ('}', false, vec!['['], (false, false), vec!['[']),
            ('x', false, vec!['{'], (false, false), vec!['{']),
            ('[', true, vec!['{'], (false, false), vec!['{', '[']),
        ];
        for (ch, pending, before, expected_ret, expected_after) in cases {
            let mut stack = before.clone();
            let ret = update_inline_container_stack(ch, pending, &mut stack);
            assert_eq!(ret, expected_ret, "ret for ({ch:?}, {pending}, {before:?})");
            assert_eq!(stack, expected_after, "stack for ({ch:?}, {pending}, {before:?})");
        }
    }
}
