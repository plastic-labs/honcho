#![allow(clippy::print_stdout)]
//! Queue status: check background processing progress.
//!
//! Demonstrates checking the deriver queue at both the workspace
//! and session levels, and scheduling a dream task.
//!
//! Run with `cargo run --example queue_status`

use honcho_ai::Honcho;

#[tokio::main]
async fn main() -> honcho_ai::error::Result<()> {
    let honcho = Honcho::from_params(
        Honcho::builder()
            .base_url("http://localhost:8000")
            .workspace_id("queue-demo")
            .build(),
    )?;

    let peer = honcho.peer("observer-1", None, None).await?;
    let session = honcho.session("sess-1", None, None, None).await?;

    session
        .add_messages(vec![peer.message("Trigger some background work").build()?])
        .await?;

    let ws_status = honcho.queue_status(None, None, None).await?;
    println!(
        "Workspace queue: total={}, pending={}, in_progress={}, completed={}",
        ws_status.total_work_units,
        ws_status.pending_work_units,
        ws_status.in_progress_work_units,
        ws_status.completed_work_units,
    );

    let sess_status = session.queue_status(None, None).await?;
    println!(
        "Session queue: total={}, pending={}, in_progress={}, completed={}",
        sess_status.total_work_units,
        sess_status.pending_work_units,
        sess_status.in_progress_work_units,
        sess_status.completed_work_units,
    );

    honcho.schedule_dream("observer-1", None, None).await?;
    println!("Dream scheduled for observer-1");

    Ok(())
}
