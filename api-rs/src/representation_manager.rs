//! `RepresentationManager.save_representation`, ported from
//! `src/crud/representation.py`.
//!
//! Persists a [`Representation`]'s observations as collection documents:
//! filter+order them, embed their texts (the only external call — performed
//! before any DB work, per the "never hold a DB session during external calls"
//! rule), shape them into documents, ensure the collection exists, then
//! bulk-create with dedup. The embedding seam is the [`Embedder`] trait so this
//! is testable without OpenAI.

use chrono::{DateTime, Utc};
use sqlx::PgPool;

use crate::db;
use crate::dialectic::Embedder;
use crate::representation::{Representation, build_documents_to_create};

/// Failure modes of [`save_representation`].
#[derive(Debug)]
pub enum SaveRepresentationError {
    /// Embedding failed (token-limit or transport). Python maps the
    /// token-limit case to a `ValidationException`.
    Embed(String),
    Database(sqlx::Error),
}

impl From<sqlx::Error> for SaveRepresentationError {
    fn from(error: sqlx::Error) -> Self {
        SaveRepresentationError::Database(error)
    }
}

/// Port of `RepresentationManager.save_representation`: save a representation's
/// observations as documents in the `(observer, observed)` collection, returning
/// the number of new documents written. Returns `0` without touching the DB or
/// the embedder when there are no non-empty explicit/deductive observations.
///
/// `deduplicate` mirrors `settings.DERIVER.DEDUPLICATE` (the caller supplies it).
/// Dream scheduling (`check_and_schedule_dream`, gated on
/// `configuration.dream.enabled`) is not yet wired and is a documented follow-up.
#[allow(clippy::too_many_arguments)]
pub async fn save_representation<E: Embedder + Sync>(
    pool: &PgPool,
    embedder: &E,
    workspace_name: &str,
    observer: &str,
    observed: &str,
    representation: &Representation,
    message_ids: &[i64],
    session_name: &str,
    message_created_at: DateTime<Utc>,
    deduplicate: bool,
) -> Result<usize, SaveRepresentationError> {
    // Ordered, stripped, non-empty observations (deductive then explicit).
    let observations = representation.observations_for_save();
    if observations.is_empty() {
        return Ok(0);
    }

    // Embed first — external call, no DB session held.
    let texts: Vec<String> = observations.iter().map(|obs| obs.content.clone()).collect();
    let embeddings = embedder
        .batch_embed(&texts)
        .await
        .map_err(SaveRepresentationError::Embed)?;

    let documents = build_documents_to_create(
        &observations,
        &embeddings,
        message_ids,
        session_name,
        message_created_at,
    );

    // Ensure the collection exists, then write (with optional dedup).
    db::get_or_create_collection(pool, workspace_name, observer, observed).await?;
    let new_documents =
        db::create_documents(pool, documents, workspace_name, observer, observed, deduplicate)
            .await?;

    Ok(new_documents)
}
