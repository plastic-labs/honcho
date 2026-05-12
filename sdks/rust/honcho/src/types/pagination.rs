//! Generic pagination types.

/// A page of results from a paginated list endpoint.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Page<T> {
    /// The items in this page.
    pub items: Vec<T>,
    /// Total number of items across all pages.
    pub total: u64,
    /// Current page number (1-based).
    pub page: u64,
    /// Number of items per page.
    pub size: u64,
    /// Total number of pages.
    pub pages: u64,
}

impl<T> Page<T> {
    /// Create a new `Page`.
    #[must_use]
    pub fn new(items: Vec<T>, total: u64, page: u64, size: u64, pages: u64) -> Self {
        Self {
            items,
            total,
            page,
            size,
            pages,
        }
    }
}

impl<T: Default> Default for Page<T> {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            total: 0,
            page: 1,
            size: 0,
            pages: 0,
        }
    }
}
