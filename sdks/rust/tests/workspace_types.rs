#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

mod common;

use common::{load_fixture, roundtrip, validate_openapi};
use honcho_ai::types::workspace::{
    DreamConfiguration, PeerCardConfiguration, ReasoningConfiguration, SummaryConfiguration,
    Workspace, WorkspaceConfiguration, WorkspaceCreate, WorkspaceGet, WorkspacePage,
    WorkspaceUpdate,
};

macro_rules! schema_test {
    ($name:ident, $schema:literal, $type:ty) => {
        mod $name {
            use super::*;

            #[test]
            fn min_validates() {
                let fixture = load_fixture($schema, "min");
                validate_openapi(fixture, $schema);
            }

            #[test]
            fn max_validates() {
                let fixture = load_fixture($schema, "max");
                validate_openapi(fixture, $schema);
            }

            #[test]
            fn min_roundtrip() {
                let fixture = load_fixture($schema, "min");
                roundtrip::<$type>(fixture);
            }

            #[test]
            fn max_roundtrip() {
                let fixture = load_fixture($schema, "max");
                roundtrip::<$type>(fixture);
            }
        }
    };
}

schema_test!(workspace, "Workspace", Workspace);
schema_test!(workspace_create, "WorkspaceCreate", WorkspaceCreate);
schema_test!(workspace_update, "WorkspaceUpdate", WorkspaceUpdate);
schema_test!(
    workspace_configuration,
    "WorkspaceConfiguration",
    WorkspaceConfiguration
);
schema_test!(workspace_get, "WorkspaceGet", WorkspaceGet);
schema_test!(workspace_page, "Page_Workspace_", WorkspacePage);
schema_test!(
    reasoning_config,
    "ReasoningConfiguration",
    ReasoningConfiguration
);
schema_test!(
    peer_card_config,
    "PeerCardConfiguration",
    PeerCardConfiguration
);
schema_test!(summary_config, "SummaryConfiguration", SummaryConfiguration);
schema_test!(dream_config, "DreamConfiguration", DreamConfiguration);

#[test]
fn workspace_builder_minimal() {
    let body = WorkspaceCreate::builder().id("test-ws").build();
    let json = serde_json::to_value(&body).unwrap();
    assert_eq!(json["id"], "test-ws");
    assert!(json.get("metadata").is_none() || json["metadata"].is_null());
    assert!(json.get("configuration").is_none() || json["configuration"].is_null());
}

#[test]
fn workspace_update_builder_empty_skips_all() {
    let body = WorkspaceUpdate::builder().build();
    let json = serde_json::to_value(&body).unwrap();
    assert!(json.as_object().unwrap().is_empty());
}

#[test]
fn workspace_get_builder_empty_skips_filters() {
    let body = WorkspaceGet::builder().build();
    let json = serde_json::to_value(&body).unwrap();
    assert!(json.as_object().unwrap().is_empty());
}
