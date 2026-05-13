use std::fmt::Display;

use honcho_ai::http::client::HttpClient;
use honcho_ai::{Conclusion, ConclusionScope};
use static_assertions::assert_impl_all;

assert_impl_all!(HttpClient: Send, Sync, Clone);
assert_impl_all!(Conclusion: Send, Sync, Clone, std::fmt::Debug, Display);
assert_impl_all!(ConclusionScope: Send, Sync, Clone);
