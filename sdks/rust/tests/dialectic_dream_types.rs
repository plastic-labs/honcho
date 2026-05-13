#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::needless_borrows_for_generic_args,
    missing_docs
)]

mod common;

use honcho_ai::types::dialectic::{DialecticOptions, ReasoningLevel, RepresentationResponse};
use honcho_ai::types::dream::{
    DreamConfiguration, DreamType, QueueStatus, ReasoningConfiguration, ScheduleDreamRequest,
};

use common::{load_fixture, roundtrip, validate_openapi};

// ── Dialectic ───────────────────────────────────────────────────────

#[test]
fn dialectic_options_min_validate() {
    let fixture = load_fixture("DialecticOptions", "min");
    validate_openapi(fixture.clone(), "DialecticOptions");
}

#[test]
fn dialectic_options_min_roundtrip() {
    let fixture = load_fixture("DialecticOptions", "min");
    roundtrip::<DialecticOptions>(fixture);
}

#[test]
fn dialectic_options_max_validate() {
    let fixture = load_fixture("DialecticOptions", "max");
    validate_openapi(fixture.clone(), "DialecticOptions");
}

#[test]
fn dialectic_options_max_roundtrip() {
    let fixture = load_fixture("DialecticOptions", "max");
    roundtrip::<DialecticOptions>(fixture);
}

#[test]
fn representation_response_min_validate() {
    let fixture = load_fixture("RepresentationResponse", "min");
    validate_openapi(fixture.clone(), "RepresentationResponse");
}

#[test]
fn representation_response_min_roundtrip() {
    let fixture = load_fixture("RepresentationResponse", "min");
    roundtrip::<RepresentationResponse>(fixture);
}

#[test]
fn representation_response_max_validate() {
    let fixture = load_fixture("RepresentationResponse", "max");
    validate_openapi(fixture.clone(), "RepresentationResponse");
}

#[test]
fn representation_response_max_roundtrip() {
    let fixture = load_fixture("RepresentationResponse", "max");
    roundtrip::<RepresentationResponse>(fixture);
}

#[test]
fn reasoning_level_serde_roundtrip() {
    let levels = ["minimal", "low", "medium", "high", "max"];
    for lvl in levels {
        let v = serde_json::json!(lvl);
        let parsed: ReasoningLevel = serde_json::from_value(v.clone()).unwrap();
        assert_eq!(
            serde_json::to_value(parsed).unwrap(),
            v,
            "mismatch for {lvl}"
        );
    }
}

#[test]
fn reasoning_level_default_is_low() {
    assert_eq!(ReasoningLevel::default(), ReasoningLevel::Low);
}

// ── Dream ───────────────────────────────────────────────────────────

#[test]
fn dream_type_serde_roundtrip() {
    let v = serde_json::json!("omni");
    let parsed: DreamType = serde_json::from_value(v.clone()).unwrap();
    assert_eq!(serde_json::to_value(&parsed).unwrap(), v);
}

#[test]
fn dream_configuration_min_validate() {
    let fixture = load_fixture("DreamConfiguration", "min");
    validate_openapi(fixture.clone(), "DreamConfiguration");
}

#[test]
fn dream_configuration_min_roundtrip() {
    let fixture = load_fixture("DreamConfiguration", "min");
    roundtrip::<DreamConfiguration>(fixture);
}

#[test]
fn dream_configuration_max_validate() {
    let fixture = load_fixture("DreamConfiguration", "max");
    validate_openapi(fixture.clone(), "DreamConfiguration");
}

#[test]
fn dream_configuration_max_roundtrip() {
    let fixture = load_fixture("DreamConfiguration", "max");
    roundtrip::<DreamConfiguration>(fixture);
}

#[test]
fn reasoning_configuration_min_validate() {
    let fixture = load_fixture("ReasoningConfiguration", "min");
    validate_openapi(fixture.clone(), "ReasoningConfiguration");
}

#[test]
fn reasoning_configuration_min_roundtrip() {
    let fixture = load_fixture("ReasoningConfiguration", "min");
    roundtrip::<ReasoningConfiguration>(fixture);
}

#[test]
fn reasoning_configuration_max_validate() {
    let fixture = load_fixture("ReasoningConfiguration", "max");
    validate_openapi(fixture.clone(), "ReasoningConfiguration");
}

#[test]
fn reasoning_configuration_max_roundtrip() {
    let fixture = load_fixture("ReasoningConfiguration", "max");
    roundtrip::<ReasoningConfiguration>(fixture);
}

#[test]
fn schedule_dream_request_min_validate() {
    let fixture = load_fixture("ScheduleDreamRequest", "min");
    validate_openapi(fixture.clone(), "ScheduleDreamRequest");
}

#[test]
fn schedule_dream_request_min_roundtrip() {
    let fixture = load_fixture("ScheduleDreamRequest", "min");
    roundtrip::<ScheduleDreamRequest>(fixture);
}

#[test]
fn schedule_dream_request_max_validate() {
    let fixture = load_fixture("ScheduleDreamRequest", "max");
    validate_openapi(fixture.clone(), "ScheduleDreamRequest");
}

#[test]
fn schedule_dream_request_max_roundtrip() {
    let fixture = load_fixture("ScheduleDreamRequest", "max");
    roundtrip::<ScheduleDreamRequest>(fixture);
}

#[test]
fn queue_status_min_validate() {
    let fixture = load_fixture("QueueStatus", "min");
    validate_openapi(fixture.clone(), "QueueStatus");
}

#[test]
fn queue_status_min_roundtrip() {
    let fixture = load_fixture("QueueStatus", "min");
    roundtrip::<QueueStatus>(fixture);
}

#[test]
fn queue_status_max_validate() {
    let fixture = load_fixture("QueueStatus", "max");
    validate_openapi(fixture.clone(), "QueueStatus");
}

#[test]
fn queue_status_max_roundtrip() {
    let fixture = load_fixture("QueueStatus", "max");
    roundtrip::<QueueStatus>(fixture);
}
