//! Faithful ports of the Python text helpers used when building prompts:
//! `str.expandtabs` and `inspect.cleandoc` (Python's `c()` import alias in the
//! prompt modules). Operating on Unicode code points (Python `str` semantics)
//! so dedent margins and slices match CPython exactly.

/// Port of `str.expandtabs(tabsize)`: replace tabs with spaces to the next tab
/// stop, resetting the column on `\n`/`\r`. With `tabsize == 0`, tabs are
/// dropped (matching CPython).
pub fn expandtabs(text: &str, tabsize: usize) -> String {
    let mut result = String::with_capacity(text.len());
    let mut column = 0usize;
    for ch in text.chars() {
        match ch {
            '\t' => {
                if tabsize > 0 {
                    let spaces = tabsize - (column % tabsize);
                    for _ in 0..spaces {
                        result.push(' ');
                    }
                    column += spaces;
                }
            }
            '\n' | '\r' => {
                result.push(ch);
                column = 0;
            }
            other => {
                result.push(other);
                column += 1;
            }
        }
    }
    result
}

/// Port of `inspect.cleandoc`: expand tabs, strip leading whitespace from the
/// first line, remove the common leading-whitespace margin from the remaining
/// non-blank lines, then drop leading and trailing fully-empty lines. Note that
/// only *empty* (length-0) lines are dropped at the ends — a whitespace-only
/// line shorter or longer than the margin can survive, exactly as in CPython.
pub fn cleandoc(doc: &str) -> String {
    let expanded = expandtabs(doc, 8);
    let mut lines: Vec<Vec<char>> = expanded
        .split('\n')
        .map(|line| line.chars().collect())
        .collect();

    // Minimum indentation of non-blank lines after the first. Python:
    // `content = len(line.lstrip()); if content: indent = len(line) - content`.
    // An all-whitespace line has content 0 and is skipped.
    let mut margin: Option<usize> = None;
    for line in lines.iter().skip(1) {
        let stripped_len = lstrip_len(line);
        if stripped_len > 0 {
            let indent = line.len() - stripped_len;
            margin = Some(margin.map_or(indent, |m| m.min(indent)));
        }
    }

    if let Some(first) = lines.first_mut() {
        let start = first.len() - lstrip_len(first);
        first.drain(0..start);
    }

    if let Some(margin) = margin {
        for line in lines.iter_mut().skip(1) {
            let drop = margin.min(line.len());
            line.drain(0..drop);
        }
    }

    while lines.last().is_some_and(|l| l.is_empty()) {
        lines.pop();
    }
    while lines.first().is_some_and(|l| l.is_empty()) {
        lines.remove(0);
    }

    lines
        .into_iter()
        .map(|line| line.into_iter().collect::<String>())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Length (in chars) of `line` after stripping leading whitespace — the Python
/// `len(line.lstrip())`.
fn lstrip_len(line: &[char]) -> usize {
    let leading = line.iter().take_while(|c| c.is_whitespace()).count();
    line.len() - leading
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expandtabs_basic() {
        assert_eq!(expandtabs("a\tb", 8), "a       b");
        assert_eq!(expandtabs("\t", 8), "        ");
        assert_eq!(expandtabs("ab\tc", 8), "ab      c");
        assert_eq!(expandtabs("a\tb", 0), "ab");
        assert_eq!(expandtabs("a\nb\tc", 4), "a\nb   c");
    }

    #[test]
    fn cleandoc_strips_uniform_margin() {
        let input = "\n        CUSTOM INSTRUCTIONS:\n        be terse\n        ";
        assert_eq!(cleandoc(input), "CUSTOM INSTRUCTIONS:\nbe terse");
    }

    #[test]
    fn cleandoc_zero_margin_keeps_whitespace_line() {
        // A later line at indent 0 forces margin 0; the trailing whitespace-only
        // line is non-empty so it survives, matching CPython.
        let input = "\n        CUSTOM INSTRUCTIONS:\n        line1\nline2\n        ";
        assert_eq!(
            cleandoc(input),
            "        CUSTOM INSTRUCTIONS:\n        line1\nline2\n        "
        );
    }

    #[test]
    fn cleandoc_strips_leading_and_trailing_blank_lines_only() {
        let input = "\n\n  first\n    second\n\n";
        // margin = 2 (min over the two indented lines); blank ends dropped.
        assert_eq!(cleandoc(input), "first\n  second");
    }

    #[test]
    fn cleandoc_first_line_lstripped() {
        let input = "   hello\n   world";
        // margin from line[1:] = 3; first line lstripped to "hello".
        assert_eq!(cleandoc(input), "hello\nworld");
    }
}
