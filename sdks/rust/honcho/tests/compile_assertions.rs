use honcho_ai::http::client::HttpClient;
use static_assertions::assert_impl_all;

assert_impl_all!(HttpClient: Send, Sync, Clone);
