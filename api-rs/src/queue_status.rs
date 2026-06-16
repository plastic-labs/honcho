use serde_json::{Map, Value, json};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueueStatusCounts {
    pub total: i64,
    pub completed: i64,
    pub in_progress: i64,
    pub pending: i64,
    pub sessions: Vec<(String, i64, i64, i64)>,
}

pub fn build_queue_status(session_name: Option<&str>, counts: QueueStatusCounts) -> Value {
    if session_name.is_some() {
        return json!({
            "sessions": null,
            "total_work_units": counts.total,
            "completed_work_units": counts.completed,
            "in_progress_work_units": counts.in_progress,
            "pending_work_units": counts.pending
        });
    }

    let mut sessions = Map::new();
    for (session_id, completed, in_progress, pending) in counts.sessions {
        sessions.insert(
            session_id.clone(),
            json!({
                "session_id": session_id,
                "total_work_units": completed + in_progress + pending,
                "completed_work_units": completed,
                "in_progress_work_units": in_progress,
                "pending_work_units": pending
            }),
        );
    }

    json!({
        "sessions": if sessions.is_empty() { Value::Null } else { Value::Object(sessions) },
        "total_work_units": counts.total,
        "completed_work_units": counts.completed,
        "in_progress_work_units": counts.in_progress,
        "pending_work_units": counts.pending
    })
}
