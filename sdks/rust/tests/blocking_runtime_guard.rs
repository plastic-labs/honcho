#![allow(clippy::unwrap_used)]
#[cfg(feature = "blocking")]
#[test]
fn honcho_new_does_not_panic() {
    let honcho = honcho_ai::blocking::Honcho::new("http://localhost:9999", "ws");
    assert!(honcho.is_ok());
    assert_eq!(honcho.unwrap().workspace_id(), "ws");
}

#[cfg(feature = "blocking")]
#[tokio::test]
#[should_panic(expected = "cannot be called from within an async runtime")]
async fn blocking_force_ensure_inside_async_panics() {
    let honcho = honcho_ai::blocking::Honcho::new("http://localhost:9999", "ws").unwrap();
    let _ = honcho.force_ensure();
}
