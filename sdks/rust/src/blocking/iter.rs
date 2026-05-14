use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures_util::Stream;

use super::runtime::block_on;

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

#[allow(clippy::cast_possible_truncation)]
pub(crate) async fn collect_all_pages<T: Clone + Send + 'static>(
    first_page: crate::types::pagination::Page<T>,
) -> crate::error::Result<Vec<T>> {
    let mut all = Vec::with_capacity(first_page.total() as usize);
    let mut first_items = first_page.items();
    all.append(&mut first_items);
    let mut current = first_page;
    while let Some(next) = current.next_page().await? {
        let mut next_items = next.items();
        all.append(&mut next_items);
        current = next;
    }
    Ok(all)
}
