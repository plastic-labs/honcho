//! Port of `parse_comment.py`.

use serde_json::Value;

use super::context::ContextValues;
use super::parser::JsonParser;

/// Port of `parse_comment`. Skips `#`/`//` line comments and `/* */` block
/// comments, returning an empty string so they don't interfere with JSON.
pub fn parse_comment(p: &mut JsonParser) -> Value {
    loop {
        let mut char = p.get_char_at(0);

        let mut termination: Vec<char> = vec!['\n', '\r'];
        if p.context.contains(ContextValues::Array) {
            termination.push(']');
        }
        if p.context.contains(ContextValues::ObjectValue) {
            termination.push('}');
        }
        if p.context.contains(ContextValues::ObjectKey) {
            termination.push(':');
        }

        // Line comment starting with #
        if char == Some('#') {
            while let Some(c) = char {
                if termination.contains(&c) {
                    break;
                }
                p.index += 1;
                char = p.get_char_at(0);
            }
        } else if char == Some('/') {
            let next_char = p.get_char_at(1);
            if next_char == Some('/') {
                // Line comment starting with //
                p.index += 2;
                char = p.get_char_at(0);
                while let Some(c) = char {
                    if termination.contains(&c) {
                        break;
                    }
                    p.index += 1;
                    char = p.get_char_at(0);
                }
            } else if next_char == Some('*') {
                // Block comment starting with /*
                p.index += 2;
                let mut comment = String::from("/*");
                loop {
                    let c = p.get_char_at(0);
                    match c {
                        None => break, // unclosed block comment
                        Some(ch) => {
                            comment.push(ch);
                            p.index += 1;
                            if comment.ends_with("*/") {
                                break;
                            }
                        }
                    }
                }
            } else {
                // Skip standalone '/'
                p.index += 1;
            }
        }

        if p.context.empty {
            // Re-enter only once after consuming a run of top-level comments.
            p.skip_whitespaces();
            if matches!(p.get_char_at(0), Some('#') | Some('/')) {
                continue;
            }
            return p.parse_json();
        }
        break;
    }
    Value::String(String::new())
}
