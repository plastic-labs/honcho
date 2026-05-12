#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

mod common;

use common::{load_fixture, roundtrip, validate_openapi};
use honcho_ai::types::validation::{HTTPValidationError, LocationSegment, ValidationError};

use serde_json::json;

#[test]
fn validation_error_min_validates_and_roundtrips() {
    let fixture = load_fixture("ValidationError", "min");
    validate_openapi(fixture.clone(), "ValidationError");
    roundtrip::<ValidationError>(fixture);
}

#[test]
fn validation_error_max_validates_and_roundtrips() {
    let fixture = load_fixture("ValidationError", "max");
    validate_openapi(fixture.clone(), "ValidationError");
    roundtrip::<ValidationError>(fixture);
}

#[test]
fn http_validation_error_min_validates_and_roundtrips() {
    let fixture = load_fixture("HTTPValidationError", "min");
    validate_openapi(fixture.clone(), "HTTPValidationError");
    roundtrip::<HTTPValidationError>(fixture);
}

#[test]
fn http_validation_error_max_validates_and_roundtrips() {
    let fixture = load_fixture("HTTPValidationError", "max");
    validate_openapi(fixture.clone(), "HTTPValidationError");
    roundtrip::<HTTPValidationError>(fixture);
}

#[test]
fn location_segment_string_and_integer() {
    let seg: LocationSegment = serde_json::from_value(json!("field_name")).unwrap();
    assert_eq!(seg, LocationSegment::String("field_name".to_string()));

    let seg: LocationSegment = serde_json::from_value(json!(3)).unwrap();
    assert_eq!(seg, LocationSegment::Integer(3));
}

#[test]
fn validation_error_optional_fields_absent_in_min() {
    let fixture = load_fixture("ValidationError", "min");
    let ve: ValidationError = serde_json::from_value(fixture).unwrap();
    assert!(ve.input.is_none());
    assert!(ve.ctx.is_none());
}

#[test]
fn validation_error_optional_fields_present_in_max() {
    let fixture = load_fixture("ValidationError", "max");
    let ve: ValidationError = serde_json::from_value(fixture).unwrap();
    assert!(ve.input.is_some());
    assert!(ve.ctx.is_some());
}

#[test]
fn http_validation_error_max_has_two_details() {
    let fixture = load_fixture("HTTPValidationError", "max");
    let http: HTTPValidationError = serde_json::from_value(fixture).unwrap();
    assert_eq!(http.detail.len(), 2);
}

#[test]
fn loc_path_with_mixed_segments() {
    let fixture = load_fixture("ValidationError", "max");
    let ve: ValidationError = serde_json::from_value(fixture).unwrap();
    assert_eq!(ve.loc.len(), 4);
    assert_eq!(ve.loc[0], LocationSegment::String("body".to_string()));
    assert_eq!(ve.loc[1], LocationSegment::String("metadata".to_string()));
    assert_eq!(ve.loc[2], LocationSegment::Integer(0));
    assert_eq!(ve.loc[3], LocationSegment::String("key".to_string()));
}

#[test]
fn skip_serializing_none_optional_fields() {
    let ve: ValidationError = serde_json::from_value(json!({
        "loc": ["query"],
        "msg": "field required",
        "type": "value_error.missing"
    }))
    .unwrap();
    let json_val = serde_json::to_value(&ve).unwrap();
    assert!(!json_val.as_object().unwrap().contains_key("input"));
    assert!(!json_val.as_object().unwrap().contains_key("ctx"));
}
