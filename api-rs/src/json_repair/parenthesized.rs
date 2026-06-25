//! Port of `parser_parenthesized.py` (the no-schema-relevant helpers).

use super::STRING_DELIMITERS;
use super::parser::JsonParser;

/// Port of `parenthesized_is_explicit_tuple`.
pub fn parenthesized_is_explicit_tuple(p: &JsonParser) -> bool {
    let mut i = p.index + 1;
    let n = p.len();
    let mut nested_parentheses = 0i64;
    let mut square_brackets = 0i64;
    let mut braces = 0i64;
    let mut in_quote: Option<char> = None;
    let mut backslashes = 0u32;
    let mut saw_top_level_content = false;

    while i < n {
        let ch = p.json[i as usize];

        if ch == '\\' {
            backslashes += 1;
            i += 1;
            continue;
        }

        if let Some(q) = in_quote {
            if ch == q && backslashes.is_multiple_of(2) {
                in_quote = None;
            }
            backslashes = 0;
            i += 1;
            continue;
        }

        if STRING_DELIMITERS.contains(&ch) && backslashes.is_multiple_of(2) {
            in_quote = Some(ch);
            saw_top_level_content = saw_top_level_content
                || (nested_parentheses == 0 && square_brackets == 0 && braces == 0);
            backslashes = 0;
            i += 1;
            continue;
        }

        backslashes = 0;

        if !ch.is_whitespace()
            && ch != ','
            && ch != ')'
            && nested_parentheses == 0
            && square_brackets == 0
            && braces == 0
        {
            saw_top_level_content = true;
        }

        if ch == '(' {
            nested_parentheses += 1;
        } else if ch == ')' {
            if nested_parentheses == 0 && square_brackets == 0 && braces == 0 {
                return !saw_top_level_content;
            }
            if nested_parentheses > 0 {
                nested_parentheses -= 1;
            }
        } else if ch == '[' {
            square_brackets += 1;
        } else if ch == ']' && square_brackets > 0 {
            square_brackets -= 1;
        } else if ch == '{' {
            braces += 1;
        } else if ch == '}' && braces > 0 {
            braces -= 1;
        } else if ch == ','
            && nested_parentheses == 0
            && square_brackets == 0
            && braces == 0
        {
            return true;
        }

        i += 1;
    }

    !saw_top_level_content
}

/// Port of `top_level_parenthesized_can_start_value`.
pub fn top_level_parenthesized_can_start_value(p: &JsonParser) -> bool {
    let mut i = p.index - 1;
    while i >= 0 {
        let ch = p.json[i as usize];
        if ch == '\n' || ch == '\r' {
            break;
        }
        if !ch.is_whitespace() {
            return false;
        }
        i -= 1;
    }

    let idx = p.scroll_whitespaces(1);
    let first_inner_char = match p.get_char_at(idx) {
        None => return false,
        Some(c) => c,
    };

    let s4 = p.substr(p.index + idx, p.index + idx + 4);
    let s5 = p.substr(p.index + idx, p.index + idx + 5);
    let is_starter = first_inner_char == ')'
        || first_inner_char == '{'
        || first_inner_char == '['
        || first_inner_char == '('
        || STRING_DELIMITERS.contains(&first_inner_char)
        || first_inner_char.is_ascii_digit()
        || first_inner_char == '-'
        || first_inner_char == '.'
        || s4 == "true"
        || s4 == "null"
        || s5 == "false";
    if !is_starter {
        return false;
    }

    let mut i = p.index + 1;
    let n = p.len();
    let mut nested_parentheses = 0i64;
    let mut square_brackets = 0i64;
    let mut braces = 0i64;
    let mut in_quote: Option<char> = None;
    let mut backslashes = 0u32;

    while i < n {
        let ch = p.json[i as usize];

        if ch == '\\' {
            backslashes += 1;
            i += 1;
            continue;
        }

        if let Some(q) = in_quote {
            if ch == q && backslashes.is_multiple_of(2) {
                in_quote = None;
            }
            backslashes = 0;
            i += 1;
            continue;
        }

        if STRING_DELIMITERS.contains(&ch) && backslashes.is_multiple_of(2) {
            in_quote = Some(ch);
            backslashes = 0;
            i += 1;
            continue;
        }

        backslashes = 0;

        if ch == '(' {
            nested_parentheses += 1;
        } else if ch == ')' {
            if nested_parentheses == 0 && square_brackets == 0 && braces == 0 {
                i += 1;
                while i < n {
                    let trailer = p.json[i as usize];
                    if trailer == '\n' || trailer == '\r' {
                        return true;
                    }
                    if !trailer.is_whitespace() {
                        return false;
                    }
                    i += 1;
                }
                return true;
            }
            nested_parentheses -= 1;
        } else if ch == '[' {
            square_brackets += 1;
        } else if ch == ']' && square_brackets > 0 {
            square_brackets -= 1;
        } else if ch == '{' {
            braces += 1;
        } else if ch == '}' && braces > 0 {
            braces -= 1;
        }

        i += 1;
    }

    true
}
