use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const DEFAULT_PAGE: u64 = 1;
const DEFAULT_SIZE: u64 = 50;

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, Serialize)]
pub struct Pagination {
    pub page: u64,
    pub size: u64,
}

impl Default for Pagination {
    fn default() -> Self {
        Self {
            page: DEFAULT_PAGE,
            size: DEFAULT_SIZE,
        }
    }
}

impl Pagination {
    pub fn new(page: Option<u64>, size: Option<u64>) -> Self {
        Self {
            page: page.unwrap_or(DEFAULT_PAGE).max(1),
            size: size.unwrap_or(DEFAULT_SIZE).max(1),
        }
    }

    pub fn limit(&self) -> i64 {
        self.size.min(i64::MAX as u64) as i64
    }

    pub fn offset(&self) -> i64 {
        self.page
            .saturating_sub(1)
            .saturating_mul(self.size)
            .min(i64::MAX as u64) as i64
    }
}

pub fn page_response(items: Vec<Value>, total: u64, pagination: Pagination) -> Value {
    let pages = if total == 0 {
        0
    } else {
        total.div_ceil(pagination.size)
    };

    json!({
        "items": items,
        "total": total,
        "page": pagination.page,
        "size": pagination.size,
        "pages": pages
    })
}
