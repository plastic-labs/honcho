#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

mod common;

use common::{load_fixture, roundtrip, validate_openapi};

use honcho_ai::types::conclusion::{
    Conclusion, ConclusionBatchCreate, ConclusionCreate, ConclusionGet, ConclusionPage,
    ConclusionQuery,
};

// ---------------------------------------------------------------------------
// Validate fixtures against OpenAPI JSON Schema
// ---------------------------------------------------------------------------

#[test]
fn conclusion_min_validates() {
    validate_openapi(load_fixture("Conclusion", "min"), "Conclusion");
}

#[test]
fn conclusion_max_validates() {
    validate_openapi(load_fixture("Conclusion", "max"), "Conclusion");
}

#[test]
fn conclusion_create_min_validates() {
    validate_openapi(load_fixture("ConclusionCreate", "min"), "ConclusionCreate");
}

#[test]
fn conclusion_create_max_validates() {
    validate_openapi(load_fixture("ConclusionCreate", "max"), "ConclusionCreate");
}

#[test]
fn conclusion_batch_create_min_validates() {
    validate_openapi(
        load_fixture("ConclusionBatchCreate", "min"),
        "ConclusionBatchCreate",
    );
}

#[test]
fn conclusion_batch_create_max_validates() {
    validate_openapi(
        load_fixture("ConclusionBatchCreate", "max"),
        "ConclusionBatchCreate",
    );
}

#[test]
fn conclusion_get_min_validates() {
    validate_openapi(load_fixture("ConclusionGet", "min"), "ConclusionGet");
}

#[test]
fn conclusion_get_max_validates() {
    validate_openapi(load_fixture("ConclusionGet", "max"), "ConclusionGet");
}

#[test]
fn conclusion_query_min_validates() {
    validate_openapi(load_fixture("ConclusionQuery", "min"), "ConclusionQuery");
}

#[test]
fn conclusion_query_max_validates() {
    validate_openapi(load_fixture("ConclusionQuery", "max"), "ConclusionQuery");
}

#[test]
fn page_conclusion_min_validates() {
    validate_openapi(load_fixture("Page_Conclusion_", "min"), "Page_Conclusion_");
}

#[test]
fn page_conclusion_max_validates() {
    validate_openapi(load_fixture("Page_Conclusion_", "max"), "Page_Conclusion_");
}

// ---------------------------------------------------------------------------
// Round-trip: deserialize → serialize → compare
// ---------------------------------------------------------------------------

#[test]
fn conclusion_roundtrip_min() {
    roundtrip::<Conclusion>(load_fixture("Conclusion", "min"));
}

#[test]
fn conclusion_roundtrip_max() {
    roundtrip::<Conclusion>(load_fixture("Conclusion", "max"));
}

#[test]
fn conclusion_create_roundtrip_min() {
    roundtrip::<ConclusionCreate>(load_fixture("ConclusionCreate", "min"));
}

#[test]
fn conclusion_create_roundtrip_max() {
    roundtrip::<ConclusionCreate>(load_fixture("ConclusionCreate", "max"));
}

#[test]
fn conclusion_batch_create_roundtrip_min() {
    roundtrip::<ConclusionBatchCreate>(load_fixture("ConclusionBatchCreate", "min"));
}

#[test]
fn conclusion_batch_create_roundtrip_max() {
    roundtrip::<ConclusionBatchCreate>(load_fixture("ConclusionBatchCreate", "max"));
}

#[test]
fn conclusion_get_roundtrip_min() {
    roundtrip::<ConclusionGet>(load_fixture("ConclusionGet", "min"));
}

#[test]
fn conclusion_get_roundtrip_max() {
    roundtrip::<ConclusionGet>(load_fixture("ConclusionGet", "max"));
}

#[test]
fn conclusion_query_roundtrip_min() {
    roundtrip::<ConclusionQuery>(load_fixture("ConclusionQuery", "min"));
}

#[test]
fn conclusion_query_roundtrip_max() {
    roundtrip::<ConclusionQuery>(load_fixture("ConclusionQuery", "max"));
}

#[test]
fn page_conclusion_roundtrip_min() {
    roundtrip::<ConclusionPage>(load_fixture("Page_Conclusion_", "min"));
}

#[test]
fn page_conclusion_roundtrip_max() {
    roundtrip::<ConclusionPage>(load_fixture("Page_Conclusion_", "max"));
}
