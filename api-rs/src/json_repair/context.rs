//! Port of `json_repair/utils/json_context.py`: the parser's context stack.

/// Port of `ContextValues`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContextValues {
    ObjectKey,
    ObjectValue,
    Array,
}

/// Port of `JsonContext`.
///
/// The Python `enter()` context manager (push on `__enter__`, pop on `__exit__`)
/// is modeled at call sites with explicit `set()`/`reset()` pairs; the data and
/// its `set`/`reset`/`clear` semantics are reproduced exactly here.
#[derive(Debug, Default)]
pub struct JsonContext {
    pub context: Vec<ContextValues>,
    pub current: Option<ContextValues>,
    pub empty: bool,
}

impl JsonContext {
    pub fn new() -> Self {
        JsonContext {
            context: Vec::new(),
            current: None,
            empty: true,
        }
    }

    /// Set a new context value (push).
    pub fn set(&mut self, value: ContextValues) {
        self.context.push(value);
        self.current = Some(value);
        self.empty = false;
    }

    /// Remove the most recent context value (pop), updating `current`/`empty`.
    pub fn reset(&mut self) {
        self.context.pop();
        match self.context.last() {
            Some(&v) => self.current = Some(v),
            None => {
                self.current = None;
                self.empty = true;
            }
        }
    }

    /// Remove all context values.
    pub fn clear(&mut self) {
        self.context.clear();
        self.current = None;
        self.empty = true;
    }

    /// `value in self.context` — whether the context stack contains `value`.
    pub fn contains(&self, value: ContextValues) -> bool {
        self.context.contains(&value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_context_is_empty() {
        let ctx = JsonContext::new();
        assert!(ctx.empty);
        assert_eq!(ctx.current, None);
        assert!(ctx.context.is_empty());
    }

    #[test]
    fn set_reset_tracks_current_and_empty() {
        let mut ctx = JsonContext::new();
        ctx.set(ContextValues::Array);
        assert!(!ctx.empty);
        assert_eq!(ctx.current, Some(ContextValues::Array));

        ctx.set(ContextValues::ObjectKey);
        assert_eq!(ctx.current, Some(ContextValues::ObjectKey));
        assert!(ctx.contains(ContextValues::Array));
        assert!(ctx.contains(ContextValues::ObjectKey));

        ctx.reset();
        // Back to Array as current, still non-empty.
        assert_eq!(ctx.current, Some(ContextValues::Array));
        assert!(!ctx.empty);

        ctx.reset();
        // Stack drained -> current None, empty True (mirrors the IndexError branch).
        assert_eq!(ctx.current, None);
        assert!(ctx.empty);

        // Extra reset on an already-empty stack stays empty (pop is a no-op).
        ctx.reset();
        assert_eq!(ctx.current, None);
        assert!(ctx.empty);
    }

    #[test]
    fn clear_drains_everything() {
        let mut ctx = JsonContext::new();
        ctx.set(ContextValues::Array);
        ctx.set(ContextValues::ObjectValue);
        ctx.clear();
        assert!(ctx.empty);
        assert_eq!(ctx.current, None);
        assert!(!ctx.contains(ContextValues::Array));
    }
}
