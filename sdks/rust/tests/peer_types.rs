//! Round-trip and OpenAPI-validation tests for Peer types.

#![allow(clippy::unwrap_used, clippy::expect_used, missing_docs)]

mod common;
use common::*;

use honcho_ai::types::peer::{
    Peer, PeerCardConfiguration, PeerCardResponse, PeerCardSet, PeerContext, PeerCreate, PeerGet,
    PeerPage, PeerRepresentationGet, PeerUpdate,
};
use rstest::rstest;
use serde::de::DeserializeOwned;
use serde::Serialize;

fn do_test<T>(schema_name: &str, variant: &str)
where
    T: Serialize + DeserializeOwned,
{
    let fixture = load_fixture(schema_name, variant);
    validate_openapi(fixture.clone(), schema_name);
    roundtrip::<T>(fixture);
}

// ---------------------------------------------------------------------------
// Per-schema tests
// ---------------------------------------------------------------------------

#[rstest]
#[case::min("min")]
#[case::max("max")]
fn peer(#[case] variant: &str) {
    do_test::<Peer>("Peer", variant);
}

#[rstest]
#[case::min("min")]
#[case::max("max")]
fn peer_create(#[case] variant: &str) {
    do_test::<PeerCreate>("PeerCreate", variant);
}

#[rstest]
#[case::min("min")]
#[case::max("max")]
fn peer_update(#[case] variant: &str) {
    do_test::<PeerUpdate>("PeerUpdate", variant);
}

#[rstest]
#[case::min("min")]
#[case::max("max")]
fn peer_get(#[case] variant: &str) {
    do_test::<PeerGet>("PeerGet", variant);
}

#[rstest]
#[case::min("min")]
#[case::max("max")]
fn peer_card_configuration(#[case] variant: &str) {
    do_test::<PeerCardConfiguration>("PeerCardConfiguration", variant);
}

#[rstest]
#[case::min("min")]
#[case::max("max")]
fn peer_card_response(#[case] variant: &str) {
    do_test::<PeerCardResponse>("PeerCardResponse", variant);
}

#[rstest]
#[case::min("min")]
#[case::max("max")]
fn peer_card_set(#[case] variant: &str) {
    do_test::<PeerCardSet>("PeerCardSet", variant);
}

#[rstest]
#[case::min("min")]
#[case::max("max")]
fn peer_context(#[case] variant: &str) {
    do_test::<PeerContext>("PeerContext", variant);
}

#[rstest]
#[case::min("min")]
#[case::max("max")]
fn peer_representation_get(#[case] variant: &str) {
    do_test::<PeerRepresentationGet>("PeerRepresentationGet", variant);
}

#[rstest]
#[case::min("min")]
#[case::max("max")]
fn peer_page(#[case] variant: &str) {
    do_test::<PeerPage>("Page_Peer_", variant);
}
