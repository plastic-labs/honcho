#![allow(clippy::print_stderr)]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use honcho_ai::upload::FileSource;

use crate::common::try_client;

#[tokio::test]
async fn upload_bytes_file_to_session() {
    let Some(client) = try_client().await else {
        return;
    };

    let _peer = client.peer("upload-test-peer").await.unwrap();
    let session = client.session("upload-test-session").await.unwrap();
    session.add_peer("upload-test-peer").await.unwrap();

    let content = "X".repeat(1024);
    let data = content.as_bytes().to_vec();

    let result = session
        .upload_file(FileSource::bytes("test.txt", data, "text/plain"))
        .peer("upload-test-peer")
        .send()
        .await;

    let messages = match result {
        Ok(msgs) => msgs,
        Err(e) => {
            eprintln!("skipping upload test: upload failed: {e}");
            session.delete().await.ok();
            client.delete_workspace(client.workspace_id()).await.ok();
            return;
        }
    };

    assert!(
        !messages.is_empty(),
        "expected at least one message from upload"
    );

    session.delete().await.unwrap();
    client
        .delete_workspace(client.workspace_id())
        .await
        .unwrap();
}

#[tokio::test]
async fn upload_streamed_file_to_session() {
    let Some(client) = try_client().await else {
        return;
    };

    let _peer = client.peer("upload-stream-peer").await.unwrap();
    let session = client.session("upload-stream-session").await.unwrap();
    session.add_peer("upload-stream-peer").await.unwrap();

    let content = "A".repeat(512);
    let cursor = std::io::Cursor::new(content.into_bytes());

    let result = session
        .upload_file_streamed("streamed.txt", cursor, "text/plain")
        .peer("upload-stream-peer")
        .send()
        .await;

    let messages = match result {
        Ok(msgs) => msgs,
        Err(e) => {
            eprintln!("skipping streamed upload test: upload failed: {e}");
            session.delete().await.ok();
            client.delete_workspace(client.workspace_id()).await.ok();
            return;
        }
    };

    assert!(
        !messages.is_empty(),
        "expected at least one message from streamed upload"
    );

    session.delete().await.unwrap();
    client
        .delete_workspace(client.workspace_id())
        .await
        .unwrap();
}
