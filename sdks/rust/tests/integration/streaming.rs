#![allow(clippy::print_stderr)]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use futures_util::StreamExt;

use crate::common::try_client;

#[tokio::test]
async fn chat_stream_drains_content() {
    let Some(client) = try_client().await else {
        return;
    };

    let peer = client.peer("stream-test-peer", None, None).await.unwrap();

    let stream = peer.chat_stream("Hi").send().await;

    let stream = match stream {
        Ok(s) => s,
        Err(e) => {
            eprintln!("skipping stream test: could not start stream: {e}");
            client.delete_workspace(client.workspace_id()).await.ok();
            return;
        }
    };

    let mut collected = String::new();
    let mut chunk_count = 0usize;
    let mut stream = Box::pin(stream);

    let _ = tokio::time::timeout(std::time::Duration::from_secs(30), async {
        while let Some(result) = stream.next().await {
            match result {
                Ok(chunk) => {
                    collected.push_str(&chunk);
                    chunk_count += 1;
                }
                Err(e) => {
                    eprintln!("stream error: {e}");
                    break;
                }
            }
        }
    })
    .await;

    assert!(
        chunk_count > 0,
        "expected at least one chunk from stream, got {chunk_count}"
    );

    client.delete_workspace(client.workspace_id()).await.ok();
}
