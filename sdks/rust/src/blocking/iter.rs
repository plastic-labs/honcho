use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures_util::Stream;

use super::runtime::block_on;

const MAX_COLLECT_PAGES: u32 = 1_000;

pub(crate) struct BlockingIter<S> {
    stream: Pin<Box<S>>,
}

impl<S> BlockingIter<S> {
    pub(crate) fn new(stream: S) -> Self {
        Self {
            stream: Box::pin(stream),
        }
    }

    pub(crate) fn stream(&self) -> &S {
        self.stream.as_ref().get_ref()
    }
}

struct StreamNext<'a, S> {
    stream: &'a mut Pin<Box<S>>,
}

impl<S: Stream> Future for StreamNext<'_, S> {
    type Output = Option<S::Item>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.stream.as_mut().poll_next(cx)
    }
}

impl<S> Iterator for BlockingIter<S>
where
    S: Stream,
{
    type Item = S::Item;

    fn next(&mut self) -> Option<Self::Item> {
        block_on(StreamNext {
            stream: &mut self.stream,
        })
    }
}

pub(crate) async fn collect_all_pages<
    TRaw: Clone + Send + 'static,
    TOut: Clone + Send + 'static,
>(
    first_page: crate::types::pagination::Page<TRaw, TOut>,
) -> crate::error::Result<Vec<TOut>> {
    let cap = usize::try_from(first_page.total()).unwrap_or(usize::MAX);
    let mut all = Vec::with_capacity(cap.min(10_000));
    let mut first_items = first_page.items();
    all.append(&mut first_items);
    let mut current = first_page;
    let mut pages: u32 = 1;
    while let Some(next) = current.next_page().await? {
        pages += 1;
        if pages > MAX_COLLECT_PAGES {
            return Err(crate::error::HonchoError::Validation(format!(
                "pagination exceeded {MAX_COLLECT_PAGES} pages (attempted {pages}), aborting to prevent infinite-loop safety cap"
            )));
        }
        let mut next_items = next.items();
        all.append(&mut next_items);
        current = next;
    }
    Ok(all)
}
