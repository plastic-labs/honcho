//! Validate + round-trip tests for message-related types.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

mod common;
use common::*;

use honcho_ai::types::message::*;
use rstest::rstest;

macro_rules! schema_tests {
    ($type:ty, $schema:literal) => {
        paste::paste! {
            #[rstest]
            fn [<validate_ $schema:snake _min>]() {
                let val = load_fixture($schema, "min");
                let _: $type = serde_json::from_value(val.clone()).unwrap();
                validate_openapi(val, $schema);
            }

            #[rstest]
            fn [<validate_ $schema:snake _max>]() {
                let val = load_fixture($schema, "max");
                let _: $type = serde_json::from_value(val.clone()).unwrap();
                validate_openapi(val, $schema);
            }

            #[rstest]
            fn [<roundtrip_ $schema:snake _min>]() {
                let val = load_fixture($schema, "min");
                roundtrip::<$type>(val);
            }

            #[rstest]
            fn [<roundtrip_ $schema:snake _max>]() {
                let val = load_fixture($schema, "max");
                roundtrip::<$type>(val);
            }
        }
    };
}

mod message_schemas {
    use super::*;

    schema_tests!(Message, "Message");
    schema_tests!(MessageCreate, "MessageCreate");
    schema_tests!(MessageBatchCreate, "MessageBatchCreate");
    schema_tests!(MessageUpdate, "MessageUpdate");
    schema_tests!(MessageGet, "MessageGet");
    schema_tests!(MessageConfiguration, "MessageConfiguration");
    schema_tests!(MessageSearchOptions, "MessageSearchOptions");
}

mod message_upload_form_roundtrip {
    use super::*;

    #[rstest]
    fn roundtrip_message_upload_form_min() {
        let val = load_fixture("MessageUploadForm", "min");
        roundtrip::<MessageUploadForm>(val);
    }

    #[rstest]
    fn roundtrip_message_upload_form_max() {
        let val = load_fixture("MessageUploadForm", "max");
        roundtrip::<MessageUploadForm>(val);
    }
}

mod page_message_roundtrip {
    use super::*;

    #[rstest]
    fn roundtrip_page_message_min() {
        let val = load_fixture("Page_Message", "min");
        roundtrip::<MessagePage>(val);
    }

    #[rstest]
    fn roundtrip_page_message_max() {
        let val = load_fixture("Page_Message", "max");
        roundtrip::<MessagePage>(val);
    }
}

#[test]
fn message_all_fields_present() {
    let val = load_fixture("Message", "max");
    let msg: Message = serde_json::from_value(val).unwrap();
    assert_eq!(msg.id, "msg_02");
    assert_eq!(msg.peer_id, "peer_abc123");
    assert_eq!(msg.session_id, "sess_xyz789");
    assert_eq!(msg.workspace_id, "ws_prod_001");
    assert_eq!(msg.token_count, 127);
    assert!(msg.metadata.contains_key("key"));
}

#[test]
fn message_create_optional_fields_none() {
    let val = load_fixture("MessageCreate", "min");
    let mc: MessageCreate = serde_json::from_value(val).unwrap();
    assert_eq!(mc.content, "hello");
    assert_eq!(mc.peer_id, "peer_01");
    assert!(mc.metadata.is_none());
    assert!(mc.configuration.is_none());
    assert!(mc.created_at.is_none());
}

#[test]
fn message_batch_create_length() {
    let val = load_fixture("MessageBatchCreate", "max");
    let batch: MessageBatchCreate = serde_json::from_value(val).unwrap();
    assert_eq!(batch.messages.len(), 2);
}

#[test]
fn message_update_empty_is_valid() {
    let val = load_fixture("MessageUpdate", "min");
    let upd: MessageUpdate = serde_json::from_value(val).unwrap();
    assert!(upd.metadata.is_none());
}

#[test]
fn message_configuration_with_reasoning() {
    let val = load_fixture("MessageConfiguration", "max");
    let cfg: MessageConfiguration = serde_json::from_value(val).unwrap();
    let r = cfg.reasoning.unwrap();
    assert_eq!(r.enabled, Some(true));
    assert_eq!(
        r.custom_instructions,
        Some("Analyze sentiment carefully".to_string())
    );
}

#[test]
fn message_search_options_default_limit() {
    let val = load_fixture("MessageSearchOptions", "min");
    let opts: MessageSearchOptions = serde_json::from_value(val).unwrap();
    assert_eq!(opts.limit, 10);
    assert_eq!(opts.query, "test");
}

#[test]
fn message_page_empty() {
    let val = load_fixture("Page_Message", "min");
    let page: MessagePage = serde_json::from_value(val).unwrap();
    assert!(page.items.is_empty());
    assert_eq!(page.total, 0);
    assert_eq!(page.pages, 0);
}

#[test]
fn message_page_with_items() {
    let val = load_fixture("Page_Message", "max");
    let page: MessagePage = serde_json::from_value(val).unwrap();
    assert_eq!(page.items.len(), 2);
    assert_eq!(page.total, 42);
    assert_eq!(page.page, 2);
    assert_eq!(page.size, 10);
    assert_eq!(page.pages, 5);
}
