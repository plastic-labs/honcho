//! Port of `parse_string_helpers/parse_json_llm_block.py`.

use serde_json::Value;

use super::parser::JsonParser;

/// Port of `parse_json_llm_block`. Extracts JSON enclosed in a ```` ```json ... ``` ````
/// block. Returns `None` for the Python `False` (not a block) sentinel.
pub fn parse_json_llm_block(p: &mut JsonParser) -> Option<Value> {
    if p.substr(p.index, p.index + 7) == "```json" {
        let i = p.skip_to_char('`', 7);
        if p.substr(p.index + i, p.index + i + 3) == "```" {
            p.index += 7; // Move past ```json
            return Some(p.parse_json());
        }
    }
    None
}
