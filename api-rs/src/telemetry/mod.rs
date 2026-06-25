//! Telemetry foundation, ported from `src/telemetry/`.
//!
//! Port status: the deterministic idempotency-key core (`generate_event_id`) and
//! the Python `datetime.isoformat()` formatting it depends on land first. The
//! CloudEvents envelope, the per-event structs, the emitter transport, and
//! sampling are larger follow-on units; `emit()` is pure observability (it feeds
//! an external sink and never feeds back into control flow), so the worker
//! orchestrators can be wired with the event payloads computed here ahead of the
//! transport.

pub mod events;

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use chrono::{DateTime, Timelike, Utc};
use sha2::{Digest, Sha256};

/// Port of Python `datetime.isoformat()` for a tz-aware UTC datetime: the
/// fractional-second component is omitted when zero, otherwise rendered with
/// 6 digits (microsecond resolution), and the offset is always `+00:00`.
pub fn python_isoformat_utc(dt: DateTime<Utc>) -> String {
    let base = dt.format("%Y-%m-%dT%H:%M:%S").to_string();
    let micros = dt.nanosecond() / 1_000;
    if micros == 0 {
        format!("{base}+00:00")
    } else {
        format!("{base}.{micros:06}+00:00")
    }
}

/// Port of `generate_event_id`: a deterministic, idempotent event id.
///
/// `timestamp_iso` is the already-formatted [`python_isoformat_utc`] string
/// (kept as a parameter so the hash stays pure). `honcho_version` of `None`
/// folds in an empty segment, matching the Python default.
pub fn generate_event_id(
    event_type: &str,
    resource_id: &str,
    timestamp_iso: &str,
    honcho_version: Option<&str>,
) -> String {
    let version_segment = honcho_version.unwrap_or("");
    let payload = format!("{event_type}:{resource_id}:{timestamp_iso}:{version_segment}");
    let digest = Sha256::digest(payload.as_bytes());
    let encoded = URL_SAFE_NO_PAD.encode(&digest[..16]);
    format!("evt_{encoded}")
}

/// Port of the `BaseEvent` interface (`events/base.py`): the metadata + identity
/// methods the emitter reads off every event. Concrete event structs implement
/// it so the emitter can stay generic over event type.
pub trait TelemetryEvent {
    /// The CloudEvents `type` string (`_event_type`).
    fn event_type(&self) -> &'static str;
    /// Schema version for evolution (`_schema_version`).
    fn schema_version(&self) -> i32;
    /// `"work"`, `"activity"`, or `"resource"` — used to build the source path.
    fn category(&self) -> &'static str;
    /// `"ground_truth"` (default) or `"high_volume"` — drives sampling.
    fn volume_class(&self) -> &'static str {
        "ground_truth"
    }
    /// When the event occurred.
    fn timestamp(&self) -> DateTime<Utc>;
    /// Unique identifier for the operation, hashed into the idempotency key.
    fn get_resource_id(&self) -> String;
    /// Deterministic idempotency id. The package version is folded in by the
    /// caller (Python reads `HONCHO_VERSION`).
    fn generate_id(&self, honcho_version: Option<&str>) -> String {
        let iso = python_isoformat_utc(self.timestamp());
        generate_event_id(self.event_type(), &self.get_resource_id(), &iso, honcho_version)
    }
    /// Serialize the event body — the CloudEvent `data` payload.
    fn to_body(&self) -> serde_json::Value;
}

/// Port of the `emit()` entry point (`events/__init__.py`). `emit` is
/// fire-and-forget observability that never feeds back into control flow, so
/// worker orchestrators take an `&dyn Emitter` and call it best-effort.
///
/// The full buffered CloudEvents HTTP transport (`emitter.py`, sampling +
/// retry + batching) is a follow-on unit; until it lands, [`NoopEmitter`]
/// matches Python's "emitter not initialized → drop event" default and
/// [`LogEmitter`] surfaces events at debug level for local worker runs.
pub trait Emitter: Send + Sync {
    fn emit(&self, event: &dyn TelemetryEvent);
}

/// Drops every event. Mirrors the Python default when the telemetry emitter is
/// uninitialized or `TELEMETRY.ENABLED` is false (`emit()` no-ops).
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopEmitter;

impl Emitter for NoopEmitter {
    fn emit(&self, _event: &dyn TelemetryEvent) {}
}

/// Logs each event's type + idempotency id at debug level. A stand-in for the
/// real transport during worker bring-up.
#[derive(Debug, Default, Clone, Copy)]
pub struct LogEmitter;

impl Emitter for LogEmitter {
    fn emit(&self, event: &dyn TelemetryEvent) {
        tracing::debug!(
            event_type = event.event_type(),
            event_id = event.generate_id(None),
            "telemetry event"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn generate_event_id_matches_python() {
        // Golden values captured from src/telemetry/events/base.py.
        assert_eq!(
            generate_event_id(
                "honcho.work.representation.completed",
                "ws:abc",
                "2026-06-21T12:34:56.789012+00:00",
                Some("2.1.0"),
            ),
            "evt_TF64pSrkfi7pvjDfknDfaQ"
        );
        assert_eq!(
            generate_event_id(
                "honcho.work.representation.completed",
                "ws:abc",
                "2026-06-21T12:34:56.789012+00:00",
                None,
            ),
            "evt_3EnAEje_8BYCUw-FhVttsg"
        );
        assert_eq!(
            generate_event_id("e.t", "rid", "2026-01-01T00:00:00+00:00", Some("")),
            "evt_5GAivHXbR8X2w0nrea8Qig"
        );
    }

    #[test]
    fn python_isoformat_omits_zero_microseconds() {
        let with_micros = Utc.with_ymd_and_hms(2026, 6, 21, 12, 34, 56).unwrap()
            + chrono::Duration::microseconds(789_012);
        assert_eq!(python_isoformat_utc(with_micros), "2026-06-21T12:34:56.789012+00:00");

        let no_micros = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        assert_eq!(python_isoformat_utc(no_micros), "2026-01-01T00:00:00+00:00");
    }

    #[test]
    fn generate_event_id_is_deterministic_via_isoformat_helper() {
        let dt = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let iso = python_isoformat_utc(dt);
        assert_eq!(
            generate_event_id("e.t", "rid", &iso, Some("")),
            "evt_5GAivHXbR8X2w0nrea8Qig"
        );
    }
}
