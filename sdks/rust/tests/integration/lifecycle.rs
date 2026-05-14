#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::redundant_closure_for_method_calls,
    missing_docs
)]

use std::collections::HashMap;
use std::collections::HashSet;
use std::time::Duration;

use honcho_ai::session::PeerSpec;
use honcho_ai::types::session::{SessionConfiguration, SessionPeerConfig};
use serde_json::json;

use crate::common::try_client;

struct WorkspaceGuard {
    client: Option<honcho_ai::Honcho>,
}

impl WorkspaceGuard {
    fn new(client: honcho_ai::Honcho) -> Self {
        Self {
            client: Some(client),
        }
    }

    fn inner(&self) -> &honcho_ai::Honcho {
        self.client.as_ref().unwrap()
    }
}

impl Drop for WorkspaceGuard {
    fn drop(&mut self) {
        if let Some(client) = self.client.take() {
            let ws_id = client.workspace_id().to_string();
            let rt = tokio::runtime::Handle::current();
            rt.spawn(async move {
                let _ = client.delete_workspace(&ws_id).await;
            });
        }
    }
}

#[tokio::test]
async fn full_lifecycle() {
    let Some(client) = try_client().await else {
        return;
    };
    let guard = WorkspaceGuard::new(client);
    let client = guard.inner();

    let peer_a = client.peer("lifecycle-alice", None, None).await.unwrap();
    assert_eq!(peer_a.id(), "lifecycle-alice");

    let peer_b = client.peer("lifecycle-bob", None, None).await.unwrap();
    assert_eq!(peer_b.id(), "lifecycle-bob");

    let session = client
        .session("lifecycle-session", None, None, None)
        .await
        .unwrap();
    assert_eq!(session.id(), "lifecycle-session");
    assert!(session.is_active());

    session
        .add_peers([PeerSpec::Id("lifecycle-alice".to_owned())])
        .await
        .unwrap();
    session
        .add_peers([PeerSpec::Id("lifecycle-bob".to_owned())])
        .await
        .unwrap();

    let peers = session.peers().await.unwrap();
    let peer_id_set: HashSet<&str> = peers.iter().map(honcho_ai::Peer::id).collect();
    assert_eq!(
        peer_id_set,
        HashSet::from(["lifecycle-alice", "lifecycle-bob"])
    );

    let msg_a = peer_a.message("Hello from Alice").build().unwrap();
    let msg_b = peer_b.message("Hello from Bob").build().unwrap();
    let created = session.add_messages(vec![msg_a, msg_b]).await.unwrap();
    assert_eq!(created.len(), 2);

    let messages = session.messages().await.unwrap();
    assert!(!messages.items().is_empty());

    let _ctx = session.context().await.unwrap();

    let search_results = session.search("Hello").await.unwrap();
    assert!(!search_results.is_empty());

    let mut meta = HashMap::new();
    meta.insert("updated".to_owned(), json!(true));
    peer_a.set_metadata(meta).await.unwrap();
    let refreshed = peer_a.get_metadata().await.unwrap();
    assert_eq!(refreshed.get("updated").unwrap(), &json!(true));

    let peers_page = client.peers().await.unwrap();
    let peer_ids: Vec<String> = peers_page.items().into_iter().map(|p| p.id).collect();
    assert!(peer_ids.contains(&"lifecycle-alice".to_string()));
    assert!(peer_ids.contains(&"lifecycle-bob".to_string()));

    let sessions_page = client.sessions().await.unwrap();
    let session_ids: Vec<String> = sessions_page.items().into_iter().map(|s| s.id).collect();
    assert!(session_ids.contains(&"lifecycle-session".to_string()));

    let fetched = session.get_message(created[0].id()).await.unwrap();
    assert_eq!(fetched.id(), created[0].id());

    let mut update_meta = HashMap::new();
    update_meta.insert("edited".to_owned(), json!(true));
    let updated_msg = session
        .update_message(created[0].id(), update_meta)
        .await
        .unwrap();
    assert_eq!(updated_msg.metadata().get("edited").unwrap(), &json!(true));

    session.delete().await.unwrap();
    drop(guard);
}

#[tokio::test]
async fn peer_metadata_and_configuration_crud() {
    let Some(client) = try_client().await else {
        return;
    };
    let guard = WorkspaceGuard::new(client);
    let client = guard.inner();

    let peer = client.peer("meta-test-peer", None, None).await.unwrap();

    let mut meta = HashMap::new();
    meta.insert("role".to_owned(), json!("tester"));
    meta.insert("version".to_owned(), json!(2));
    peer.set_metadata(meta).await.unwrap();

    let fetched = peer.get_metadata().await.unwrap();
    assert_eq!(fetched.get("role").unwrap(), &json!("tester"));

    let mut config = HashMap::new();
    config.insert("language".to_owned(), json!("en"));
    peer.set_configuration_raw(config).await.unwrap();

    let fetched_config_raw = peer.get_configuration_raw().await.unwrap();
    assert_eq!(fetched_config_raw.get("language").unwrap(), &json!("en"));

    let mut patch_meta = HashMap::new();
    patch_meta.insert("patched".to_owned(), json!(true));
    peer.update(patch_meta).await.unwrap();
    let after_patch = peer.get_metadata().await.unwrap();
    assert_eq!(after_patch.get("patched").unwrap(), &json!(true));

    drop(guard);
}

#[tokio::test]
async fn session_clone_and_summaries() {
    let Some(client) = try_client().await else {
        return;
    };
    let guard = WorkspaceGuard::new(client);
    let client = guard.inner();

    let peer = client.peer("clone-test-peer", None, None).await.unwrap();
    let session = client
        .session("clone-test-session", None, None, None)
        .await
        .unwrap();
    session.add_peer("clone-test-peer").await.unwrap();

    let msg = peer.message("message before clone").build().unwrap();
    let created = session.add_messages(vec![msg]).await.unwrap();

    let cloned = session.clone_session().await.unwrap();
    assert_ne!(cloned.id(), session.id());

    let cloned_with_msg = session
        .clone_session_with_message(created[0].id())
        .await
        .unwrap();
    assert_ne!(cloned_with_msg.id(), session.id());

    let summaries = session.summaries().await.unwrap();
    assert_eq!(summaries.id, session.id());

    session.delete().await.unwrap();
    cloned.delete().await.unwrap();
    cloned_with_msg.delete().await.unwrap();
    drop(guard);
}

#[tokio::test]
async fn session_metadata_and_configuration() {
    let Some(client) = try_client().await else {
        return;
    };
    let guard = WorkspaceGuard::new(client);
    let client = guard.inner();

    let session = client
        .session("meta-test-session", None, None, None)
        .await
        .unwrap();

    let mut meta = HashMap::new();
    meta.insert("topic".to_owned(), json!("integration"));
    session.set_metadata(meta).await.unwrap();

    let fetched_meta = session.get_metadata().await.unwrap();
    assert_eq!(fetched_meta.get("topic").unwrap(), &json!("integration"));

    let config: SessionConfiguration = serde_json::from_value(json!({
        "summary": {"enabled": true}
    }))
    .unwrap();
    session.set_configuration(&config).await.unwrap();

    let fetched_config = session.get_configuration().await.unwrap();
    assert!(fetched_config.summary.is_some());
    assert_eq!(fetched_config.summary.unwrap().enabled, Some(true));

    session.delete().await.unwrap();
    drop(guard);
}

#[tokio::test]
async fn peer_representation_and_context() {
    let Some(client) = try_client().await else {
        return;
    };
    let guard = WorkspaceGuard::new(client);
    let client = guard.inner();

    let peer = client.peer("repr-test-peer", None, None).await.unwrap();
    let session = client
        .session("repr-test-session", None, None, None)
        .await
        .unwrap();
    session.add_peer("repr-test-peer").await.unwrap();

    let msg = peer
        .message("I enjoy hiking and outdoor activities")
        .build()
        .unwrap();
    session.add_messages(vec![msg]).await.unwrap();

    let mut delay = Duration::from_millis(500);
    let max_attempts = 5;
    for attempt in 0..max_attempts {
        match peer.representation().await {
            Ok(_) => break,
            Err(e) if attempt + 1 == max_attempts => {
                panic!("representation never became available after {max_attempts} attempts: {e}");
            }
            Err(_) => {
                tokio::time::sleep(delay).await;
                delay *= 2;
            }
        }
    }

    let _ctx = peer.context().await.unwrap();

    session.delete().await.unwrap();
    drop(guard);
}

#[tokio::test]
async fn workspace_metadata_and_configuration() {
    let Some(client) = try_client().await else {
        return;
    };
    let guard = WorkspaceGuard::new(client);
    let client = guard.inner();

    let mut meta = HashMap::new();
    meta.insert("env".to_owned(), json!("integration-test"));
    client.set_metadata(meta).await.unwrap();
    let fetched_meta = client.get_metadata().await.unwrap();
    assert_eq!(fetched_meta.get("env").unwrap(), &json!("integration-test"));

    let mut config = HashMap::new();
    config.insert("feature_x".to_owned(), json!(true));
    client.set_configuration_raw(config).await.unwrap();
    let fetched_config = client.get_configuration_raw().await.unwrap();
    assert_eq!(fetched_config.get("feature_x").unwrap(), &json!(true));

    drop(guard);
}

#[tokio::test]
async fn session_per_peer_configuration() {
    let Some(client) = try_client().await else {
        return;
    };

    let session = client
        .session("peer-cfg-session", None, None, None)
        .await
        .unwrap();
    session.add_peer("peer-cfg-a").await.unwrap();

    let cfg: SessionPeerConfig =
        serde_json::from_value(json!({"observe_me": true, "observe_others": false})).unwrap();
    session
        .set_peer_configuration("peer-cfg-a", &cfg)
        .await
        .unwrap();

    let fetched = session.get_peer_configuration("peer-cfg-a").await.unwrap();
    assert_eq!(fetched.observe_me, Some(true));
    assert_eq!(fetched.observe_others, Some(false));

    session.delete().await.unwrap();
    client
        .delete_workspace(client.workspace_id())
        .await
        .unwrap();
}
