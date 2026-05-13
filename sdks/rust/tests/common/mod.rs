#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::needless_pass_by_value,
    clippy::map_unwrap_or,
    clippy::uninlined_format_args,
    clippy::unnecessary_debug_formatting,
    clippy::needless_match,
    clippy::redundant_closure_for_method_calls,
    clippy::needless_borrows_for_generic_args,
    clippy::doc_markdown,
    clippy::unused_async,
    clippy::manual_range_contains,
    missing_docs
)]

use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use serde::Serialize;

static SCHEMAS: std::sync::OnceLock<serde_json::Value> = std::sync::OnceLock::new();

fn openapi_spec() -> &'static serde_json::Value {
    SCHEMAS.get_or_init(|| {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
        let path = std::env::var("HONCHO_OPENAPI_SPEC").map_or_else(
            |_| Path::new(&manifest_dir).join("../../docs/v3/openapi.json"),
            PathBuf::from,
        );
        let content = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read openapi.json at {}: {e}", path.display()));
        serde_json::from_str(&content)
            .unwrap_or_else(|e| panic!("failed to parse openapi.json: {e}"))
    })
}

fn schema_by_name(name: &str) -> serde_json::Value {
    let spec = openapi_spec();
    let schemas = spec["components"]["schemas"].as_object().unwrap();
    let lookup = name;
    schemas
        .get(lookup)
        .unwrap_or_else(|| panic!("schema {name} not found in OpenAPI spec"))
        .clone()
}

fn resolve_refs(value: &serde_json::Value, spec: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            if let Some(ref_path) = map.get("$ref").and_then(|v| v.as_str()) {
                let parts: Vec<&str> = ref_path.split('/').collect();
                let schema_name = parts.last().unwrap();
                let resolved = spec["components"]["schemas"]
                    .get(schema_name)
                    .unwrap_or_else(|| panic!("Unresolved $ref: {schema_name}"))
                    .clone();
                return resolve_refs(&resolved, spec);
            }
            let resolved: serde_json::Map<String, serde_json::Value> = map
                .iter()
                .map(|(k, v)| (k.clone(), resolve_refs(v, spec)))
                .collect();
            serde_json::Value::Object(resolved)
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(|v| resolve_refs(v, spec)).collect())
        }
        other => other.clone(),
    }
}

pub fn load_fixture(name: &str, variant: &str) -> serde_json::Value {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let mut path = PathBuf::from(manifest_dir);
    path.push("tests/fixtures");
    path.push(name);
    path.push(format!("{variant}.json"));
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to load fixture {name}/{variant}.json: {e}"));
    serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("failed to parse fixture {name}/{variant}.json: {e}"))
}

pub fn validate_openapi(fixture: serde_json::Value, schema_name: &str) {
    let spec = openapi_spec();
    let schema = schema_by_name(schema_name);
    let resolved = resolve_refs(&schema, spec);

    let compiled = jsonschema::JSONSchema::compile(&resolved)
        .unwrap_or_else(|e| panic!("failed to compile schema {schema_name}: {e}"));

    let result = compiled.validate(&fixture);
    if let Err(errors) = result {
        let msgs: Vec<String> = errors.map(|e| e.to_string()).collect();
        panic!(
            "fixture failed OpenAPI validation for schema {schema_name}:\n  {}",
            msgs.join("\n  ")
        );
    }
}

pub fn roundtrip<T>(fixture: serde_json::Value)
where
    T: Serialize + DeserializeOwned,
{
    let deserialized: T = serde_json::from_value(fixture.clone())
        .unwrap_or_else(|e| panic!("deserialize failed for {}: {e}", std::any::type_name::<T>()));
    let re_serialized = serde_json::to_string(&deserialized)
        .unwrap_or_else(|e| panic!("serialize failed for {}: {e}", std::any::type_name::<T>()));
    let re_deserialized: T = serde_json::from_str(&re_serialized).unwrap_or_else(|e| {
        panic!(
            "re-deserialize failed for {}: {e}",
            std::any::type_name::<T>()
        )
    });

    let first_json = canonicalize(&serde_json::to_value(&deserialized).unwrap());
    let second_json = canonicalize(&serde_json::to_value(&re_deserialized).unwrap());
    assert_eq!(
        first_json,
        second_json,
        "roundtrip mismatch for {}",
        std::any::type_name::<T>()
    );
}

fn canonicalize(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut sorted: serde_json::Map<String, serde_json::Value> = map
                .iter()
                .map(|(k, v)| (k.clone(), canonicalize(v)))
                .collect();
            sorted.sort_keys();
            serde_json::Value::Object(sorted)
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(canonicalize).collect())
        }
        other => other.clone(),
    }
}
