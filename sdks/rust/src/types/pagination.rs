//! Generic pagination types.

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use futures_util::Stream;
use serde::de::DeserializeOwned;

use crate::error::Result;
use crate::http::client::HttpClient;

type PageFetcher<TRaw> = Arc<
    dyn Fn(u64) -> Pin<Box<dyn Future<Output = Result<PageResponse<TRaw>>> + Send>> + Send + Sync,
>;

/// Serde-friendly raw page response from the API.
///
/// Deserializes directly from paginated JSON responses. Convert to
/// [`Page`] for lazy-fetch and transform support via
/// [`Page::from_page_response`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PageResponse<T> {
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

impl<T> PageResponse<T> {
    /// Create a new `PageResponse`.
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

impl<T> Default for PageResponse<T> {
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

/// A page of results with lazy next-page fetching and item transform support.
///
/// `Page<TRaw, TOut>` holds raw items of type `TRaw` and lazily transforms
/// them to `TOut` via a configurable closure. The default `TOut = TRaw` uses
/// the identity transform.
///
/// Construct with [`Page::from_page_response`] or [`Page::new`], then
/// optionally chain [`Page::with_fetcher`] to enable [`Page::next_page`].
///
/// `Page` is cheaply [`Clone`] (Arc bump) and implements [`serde::Serialize`]
/// + [`serde::Deserialize`] when `TOut = TRaw`.
pub struct Page<TRaw, TOut = TRaw> {
    inner: Arc<PageInner<TRaw, TOut>>,
}

struct PageInner<TRaw, TOut> {
    items: Vec<TRaw>,
    total: u64,
    page: u64,
    size: u64,
    pages: u64,
    next_fetcher: Option<PageFetcher<TRaw>>,
    transform: Arc<dyn Fn(TRaw) -> TOut + Send + Sync>,
}

impl<TRaw: 'static, TOut: 'static> Page<TRaw, TOut> {
    /// Returns a reference to the raw (untransformed) items.
    #[must_use]
    pub fn raw_items(&self) -> &[TRaw] {
        &self.inner.items
    }

    /// Returns transformed items as an owned `Vec<TOut>`.
    ///
    /// Each raw item is cloned and passed through the transform closure.
    /// Use [`raw_items`](Self::raw_items) to avoid the clone when no
    /// transform is needed.
    #[must_use]
    pub fn items(&self) -> Vec<TOut>
    where
        TRaw: Clone,
    {
        self.inner
            .items
            .iter()
            .cloned()
            .map(|v| (self.inner.transform)(v))
            .collect()
    }

    /// Total number of items across all pages.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.inner.total
    }

    /// Current page number (1-based).
    #[must_use]
    pub fn page(&self) -> u64 {
        self.inner.page
    }

    /// Number of items per page.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.inner.size
    }

    /// Total number of pages.
    #[must_use]
    pub fn pages(&self) -> u64 {
        self.inner.pages
    }

    /// Whether there are more pages after this one.
    #[must_use]
    pub fn has_next(&self) -> bool {
        self.inner.page < self.inner.pages
    }

    /// Fetch the next page, if a fetcher is configured and more pages exist.
    ///
    /// Returns `Some(..)` on success, `None` when no fetcher is set,
    /// no more pages remain, or the fetch fails.
    pub async fn next_page(&self) -> Option<Self> {
        if !self.has_next() {
            return None;
        }
        let fetcher = Arc::clone(self.inner.next_fetcher.as_ref()?);
        let next_num = self.inner.page + 1;
        let transform = Arc::clone(&self.inner.transform);
        let next_fetcher = self.inner.next_fetcher.clone();

        let resp = fetcher(next_num).await.ok()?;
        Some(Self {
            inner: Arc::new(PageInner {
                items: resp.items,
                total: resp.total,
                page: resp.page,
                size: resp.size,
                pages: resp.pages,
                next_fetcher,
                transform,
            }),
        })
    }

    /// Attach a next-page fetcher, consuming `self` and returning a new `Page`.
    ///
    /// The fetcher receives a page number and returns a future that resolves to
    /// a [`PageResponse<TRaw>`]. The resulting page (and every page fetched
    /// through it) carries the same fetcher.
    #[must_use]
    pub fn with_fetcher<F, Fut>(self, fetcher: F) -> Self
    where
        F: Fn(u64) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<PageResponse<TRaw>>> + Send + 'static,
        TRaw: Clone,
    {
        Self {
            inner: Arc::new(PageInner {
                items: self.inner.items.clone(),
                total: self.inner.total,
                page: self.inner.page,
                size: self.inner.size,
                pages: self.inner.pages,
                next_fetcher: Some(Arc::new(move |pn| Box::pin(fetcher(pn)))),
                transform: self.inner.transform.clone(),
            }),
        }
    }

    /// Apply a secondary transform, producing a `Page<TRaw, TNewOut>`.
    ///
    /// The new transform composes `f` after the existing one.
    pub fn map<TNewOut>(
        self,
        f: impl Fn(TOut) -> TNewOut + Send + Sync + 'static,
    ) -> Page<TRaw, TNewOut>
    where
        TRaw: Clone,
    {
        let prev = self.inner.transform.clone();
        Page {
            inner: Arc::new(PageInner {
                items: self.inner.items.clone(),
                total: self.inner.total,
                page: self.inner.page,
                size: self.inner.size,
                pages: self.inner.pages,
                next_fetcher: self.inner.next_fetcher.clone(),
                transform: Arc::new(move |raw| f(prev(raw))),
            }),
        }
    }

    /// Convert this page into a stream that auto-fetches subsequent pages.
    ///
    /// Yields transformed items from the current page, then lazily fetches
    /// and yields items from each subsequent page until all pages are exhausted.
    ///
    /// If no fetcher is attached, only the current page's items are yielded.
    pub fn into_stream(self) -> impl Stream<Item = Result<TOut>> + Send + 'static
    where
        TRaw: Clone + Send + 'static,
        TOut: Send + 'static,
    {
        let items = self.inner.items.clone();
        let has_next = self.has_next();
        let next_page_num = self.inner.page + 1;
        let fetcher = self.inner.next_fetcher.clone();
        let total_pages = self.inner.pages;
        let transform = self.inner.transform.clone();

        async_stream::try_stream! {
            for item in items {
                yield transform(item);
            }

            if let Some(fetcher) = fetcher
                && has_next
            {
                let mut current_page = next_page_num;
                let mut pages = total_pages;
                while current_page <= pages {
                    let resp = (fetcher)(current_page).await?;
                    pages = resp.pages;
                    let is_last = resp.page >= pages;
                    for item in resp.items {
                        yield transform(item);
                    }
                    if is_last {
                        break;
                    }
                    current_page = resp.page + 1;
                }
            }
        }
    }
}

#[allow(clippy::mismatching_type_param_order)]
impl<TRaw: 'static> Page<TRaw, TRaw> {
    /// Create a new `Page` from raw data with no fetcher (identity transform).
    #[must_use]
    pub fn new(items: Vec<TRaw>, total: u64, page: u64, size: u64, pages: u64) -> Self {
        Self {
            inner: Arc::new(PageInner {
                items,
                total,
                page,
                size,
                pages,
                next_fetcher: None,
                transform: Arc::new(std::convert::identity),
            }),
        }
    }

    /// Create a `Page` from a deserialized [`PageResponse`].
    #[must_use]
    pub fn from_page_response(resp: PageResponse<TRaw>) -> Self {
        Self::new(resp.items, resp.total, resp.page, resp.size, resp.pages)
    }
}

#[allow(clippy::mismatching_type_param_order)]
impl<TRaw: 'static> Default for Page<TRaw, TRaw> {
    fn default() -> Self {
        Self {
            inner: Arc::new(PageInner {
                items: Vec::new(),
                total: 0,
                page: 1,
                size: 0,
                pages: 0,
                next_fetcher: None,
                transform: Arc::new(std::convert::identity),
            }),
        }
    }
}

impl<TRaw, TOut> Clone for Page<TRaw, TOut> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<TRaw: fmt::Debug, TOut> fmt::Debug for Page<TRaw, TOut> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Page")
            .field("items", &self.inner.items)
            .field("total", &self.inner.total)
            .field("page", &self.inner.page)
            .field("size", &self.inner.size)
            .field("pages", &self.inner.pages)
            .finish_non_exhaustive()
    }
}

#[allow(clippy::mismatching_type_param_order)]
impl<TRaw: PartialEq> PartialEq for Page<TRaw, TRaw> {
    fn eq(&self, other: &Self) -> bool {
        self.inner.items == other.inner.items
            && self.inner.total == other.inner.total
            && self.inner.page == other.inner.page
            && self.inner.size == other.inner.size
            && self.inner.pages == other.inner.pages
    }
}

#[allow(clippy::mismatching_type_param_order)]
impl<TRaw: Eq> Eq for Page<TRaw, TRaw> {}

#[allow(clippy::mismatching_type_param_order)]
impl<TRaw: serde::Serialize> serde::Serialize for Page<TRaw, TRaw> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("Page", 5)?;
        s.serialize_field("items", &self.inner.items)?;
        s.serialize_field("total", &self.inner.total)?;
        s.serialize_field("page", &self.inner.page)?;
        s.serialize_field("size", &self.inner.size)?;
        s.serialize_field("pages", &self.inner.pages)?;
        s.end()
    }
}

#[allow(clippy::mismatching_type_param_order)]
impl<'de, TRaw: serde::Deserialize<'de> + 'static> serde::Deserialize<'de> for Page<TRaw, TRaw> {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let resp = PageResponse::<TRaw>::deserialize(deserializer)?;
        Ok(Self::from_page_response(resp))
    }
}

/// Paginate a POST endpoint that accepts `page` / `size` / `reverse` query
/// parameters and returns a [`PageResponse`].
///
/// The returned [`Page`] carries an attached fetcher so that
/// [`Page::next_page`] works automatically.
///
/// TODO(F4): reduce visibility to `pub(crate)` once the high-level
/// Honcho client exposes public paginated methods.
#[doc(hidden)]
pub async fn paginate_post<T>(
    http: &HttpClient,
    route: &str,
    body: Option<&serde_json::Value>,
    page: u64,
    size: u64,
    reverse: bool,
) -> Result<Page<T>>
where
    T: DeserializeOwned + Clone + Send + 'static,
{
    let page_str = page.to_string();
    let size_str = size.to_string();
    let rev_str;
    let mut query: Vec<(&str, &str)> = vec![("page", &page_str), ("size", &size_str)];
    if reverse {
        rev_str = "true".to_owned();
        query.push(("reverse", &rev_str));
    }

    let resp: PageResponse<T> = http.post(route, body, &query).await?;
    let result = Page::from_page_response(resp);

    let http_clone = http.clone();
    let route_owned = route.to_owned();
    let body_clone = body.cloned();

    Ok(result.with_fetcher(move |page_num| {
        let http = http_clone.clone();
        let route = route_owned.clone();
        let body = body_clone.clone();
        Box::pin(async move {
            let page_str = page_num.to_string();
            let size_str = size.to_string();
            let rev_str;
            let mut query: Vec<(&str, &str)> = vec![("page", &page_str), ("size", &size_str)];
            if reverse {
                rev_str = "true".to_owned();
                query.push(("reverse", &rev_str));
            }
            let resp: PageResponse<T> = http.post(&route, body.as_ref(), &query).await?;
            Ok(resp)
        })
    }))
}
