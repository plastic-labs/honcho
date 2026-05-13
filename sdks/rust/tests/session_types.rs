#![allow(clippy::unwrap_used, clippy::expect_used, missing_docs)]

mod common;

use common::{load_fixture, roundtrip, validate_openapi};
use honcho_ai::types::session::{
    Session, SessionConfiguration, SessionContext, SessionCreate, SessionGet, SessionPage,
    SessionPeerConfig, SessionQueueStatus, SessionSummaries, SessionUpdate, Summary,
    SummaryConfiguration,
};

macro_rules! schema_tests {
    ($name:ident, $schema:expr, $ty:ty) => {
        mod $name {
            use super::*;

            #[test]
            fn validate_min() {
                let fixture = load_fixture($schema, "min");
                validate_openapi(fixture, $schema);
            }

            #[test]
            fn validate_max() {
                let fixture = load_fixture($schema, "max");
                validate_openapi(fixture, $schema);
            }

            #[test]
            fn roundtrip_min() {
                let fixture = load_fixture($schema, "min");
                roundtrip::<$ty>(fixture);
            }

            #[test]
            fn roundtrip_max() {
                let fixture = load_fixture($schema, "max");
                roundtrip::<$ty>(fixture);
            }
        }
    };
}

schema_tests!(session, "Session", Session);
schema_tests!(session_create, "SessionCreate", SessionCreate);
schema_tests!(session_update, "SessionUpdate", SessionUpdate);
schema_tests!(session_get, "SessionGet", SessionGet);
schema_tests!(
    session_configuration,
    "SessionConfiguration",
    SessionConfiguration
);
schema_tests!(session_context, "SessionContext", SessionContext);
schema_tests!(session_peer_config, "SessionPeerConfig", SessionPeerConfig);
schema_tests!(
    session_queue_status,
    "SessionQueueStatus",
    SessionQueueStatus
);
schema_tests!(session_summaries, "SessionSummaries", SessionSummaries);
schema_tests!(summary, "Summary", Summary);
schema_tests!(
    summary_configuration,
    "SummaryConfiguration",
    SummaryConfiguration
);
schema_tests!(page_session, "Page_Session_", SessionPage);

#[test]
fn session_create_builder_minimal() {
    let created = SessionCreate::builder()
        .id("test-session".to_string())
        .build();
    let json = serde_json::to_value(&created).unwrap();
    assert_eq!(json["id"], "test-session");
    assert!(json.get("metadata").is_none());
    assert!(json.get("peers").is_none());
    assert!(json.get("configuration").is_none());
}

#[test]
fn session_create_builder_full() {
    let peers_json = serde_json::json!({
        "peer_a": {"observe_me": true, "observe_others": false}
    });
    let peers: std::collections::HashMap<String, SessionPeerConfig> =
        serde_json::from_value(peers_json).unwrap();
    let config_json = serde_json::json!({
        "reasoning": {"enabled": true}
    });
    let config: SessionConfiguration = serde_json::from_value(config_json).unwrap();

    let created = SessionCreate::builder()
        .id("full-session".to_string())
        .metadata(serde_json::from_value(serde_json::json!({"env": "test"})).unwrap())
        .peers(peers)
        .configuration(config)
        .build();
    let json = serde_json::to_value(&created).unwrap();
    assert_eq!(json["id"], "full-session");
    assert_eq!(json["peers"]["peer_a"]["observe_me"], true);
    assert_eq!(json["configuration"]["reasoning"]["enabled"], true);
}

#[test]
fn session_update_builder_skips_none() {
    let update = SessionUpdate::builder().build();
    let json = serde_json::to_value(&update).unwrap();
    assert_eq!(json, serde_json::json!({}));
}

#[test]
fn session_get_builder_skips_none() {
    let get = SessionGet::builder().build();
    let json = serde_json::to_value(&get).unwrap();
    assert_eq!(json, serde_json::json!({}));
}
