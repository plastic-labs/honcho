//! Port of `json_repair/utils/object_comparer.py`.
//!
//! Used by the parser's top-level multi-value handling to decide whether two
//! parsed values share the same *type and structure* (atomic values are not
//! compared by content).

use serde_json::Value;

/// Whether a JSON number is integer-typed (Python `int`) vs float-typed
/// (Python `float`). `repair_json`'s number parser emits `int(...)` for plain
/// integers and `float(...)` for anything with `.`/`e`/`E`, which maps to
/// serde's i64/u64 vs f64 number representation.
fn is_integer_number(n: &serde_json::Number) -> bool {
    n.is_i64() || n.is_u64()
}

/// Port of `ObjectComparer.is_same_object`.
///
/// Mirrors Python's `type(obj1) is not type(obj2)` strictness: `bool` != number,
/// `int` != `float`, `dict` != `list`. Atomic values of matching type compare
/// equal regardless of content.
pub fn is_same_object(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Object(m1), Value::Object(m2)) => {
            if m1.len() != m2.len() {
                return false;
            }
            for (key, v1) in m1 {
                match m2.get(key) {
                    Some(v2) => {
                        if !is_same_object(v1, v2) {
                            return false;
                        }
                    }
                    None => return false,
                }
            }
            true
        }
        (Value::Array(a1), Value::Array(a2)) => {
            if a1.len() != a2.len() {
                return false;
            }
            a1.iter().zip(a2.iter()).all(|(x, y)| is_same_object(x, y))
        }
        (Value::String(_), Value::String(_)) => true,
        (Value::Bool(_), Value::Bool(_)) => true,
        (Value::Null, Value::Null) => true,
        // Same variant (Number) but Python distinguishes int from float.
        (Value::Number(n1), Value::Number(n2)) => is_integer_number(n1) == is_integer_number(n2),
        // Any other combination is a type mismatch.
        _ => false,
    }
}

/// Port of `ObjectComparer.is_strictly_empty`: True only for empty containers
/// (str/list/dict). `None`, `0`, `False` etc. are not strictly empty.
pub fn is_strictly_empty(value: &Value) -> bool {
    match value {
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(m) => m.is_empty(),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn is_same_object_golden() {
        // (a, b, expected) — captured from the Python ObjectComparer.
        let cases: Vec<(Value, Value, bool)> = vec![
            (json!({"a": 1}), json!({"b": 2}), false),
            (json!({"a": 1}), json!({"a": 2}), true),
            (json!({"a": 1, "b": 2}), json!({"a": 1}), false),
            (json!([1, 2, 3]), json!([4, 5, 6]), true),
            (json!([1, 2]), json!([1, 2, 3]), false),
            (json!("hello"), json!("world"), true),
            (json!(1), json!(1.0), false),
            (json!(true), json!(1), false),
            (json!(1), json!(2), true),
            (json!(1.5), json!(2.5), true),
            (json!(null), json!(null), true),
            (json!(null), json!(1), false),
            (json!({"a": [1, 2]}), json!({"a": [3, 4]}), true),
            (json!({"a": [1, 2]}), json!({"a": [3]}), false),
            (json!([{"x": 1}]), json!([{"y": 2}]), false),
            (json!([]), json!([]), true),
            (json!({}), json!({}), true),
            (json!({}), json!([]), false),
        ];
        for (a, b, expected) in cases {
            assert_eq!(is_same_object(&a, &b), expected, "is_same_object({a}, {b})");
        }
    }

    #[test]
    fn is_strictly_empty_golden() {
        assert!(is_strictly_empty(&json!("")));
        assert!(!is_strictly_empty(&json!("x")));
        assert!(is_strictly_empty(&json!([])));
        assert!(!is_strictly_empty(&json!([1])));
        assert!(is_strictly_empty(&json!({})));
        assert!(!is_strictly_empty(&json!({"a": 1})));
        assert!(!is_strictly_empty(&json!(null)));
        assert!(!is_strictly_empty(&json!(0)));
        assert!(!is_strictly_empty(&json!(false)));
        assert!(!is_strictly_empty(&json!(1.5)));
    }
}
