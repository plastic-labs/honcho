# Audio Upload Ingestion Design

## Summary

Add audio-only ingestion to the existing `/v3/workspaces/{workspace_id}/sessions/{session_id}/messages/upload` endpoint so `.mp3` and `.wav` uploads are accepted and converted into ordinary Honcho message rows. The endpoint remains one file per request. Audio uploads are transcribed during upload using an external provider, normalized into transcript text, chunked into message-sized text blocks, stored as regular messages, and then enqueued into the existing deriver path without changing the deriver message format.

This design is intended to support migrat     Pions into Honcho without introducing a new ingestion API surface. Throughput comes from two places: clients may issue many upload requests concurrently, and the server may transcribe large single files by splitting them into provider-safe audio segments and processing those segments with bounded parallelism.

## Goals

- Keep `/messages/upload` as the single ingestion path.
- Keep the public request contract at one file per request.
- Support audio-only uploads for `.mp3` and `.wav`.
- Convert audio into the same text message format the deriver already consumes.
- Prevent large audio files from failing due to provider or server-side size constraints.
- Improve upload throughput for long recordings with bounded parallel transcription.
- Preserve deterministic transcript ordering and deterministic message creation.

## Non-Goals

- No transcript-text upload mode.
- No new bulk upload endpoint.
- No per-request tuning for audio chunking or transcription concurrency.
- No managed-hosting customer-facing configuration surface for audio limits.
- No change to deriver prompt/message formatting.

## Existing Behavior

Today `/messages/upload` accepts a single `UploadFile`, validates it against the global `MAX_FILE_SIZE`, extracts text for supported file types, splits the resulting text with `split_text_into_chunks(...)`, creates `MessageCreate` records, stores file metadata in `internal_metadata`, and enqueues those messages for the deriver. The deriver then consumes ordinary message content and formats it as timestamped `peer: content` strings.

This means audio support can fit cleanly if the upload path produces ordinary text messages before enqueueing.

## Proposed API Behavior

The public endpoint remains:

- `POST /v3/workspaces/{workspace_id}/sessions/{session_id}/messages/upload`

The request shape remains multipart form data with:

- one `file`
- `peer_id`
- optional `metadata`
- optional `configuration`
- optional `created_at`

Supported file types for this work:

- `audio/mpeg` and `.mp3`
- `audio/wav`, `audio/x-wav`, and `.wav`

Non-audio transcript uploads are explicitly out of scope for this feature.

Responses remain `201` with a list of created `Message` objects. From the client perspective, audio uploads behave like current text/PDF/JSON uploads: one file may create one or more stored messages.

## Architecture

### Upload Router

The router continues to accept one `UploadFile` and delegates to shared upload processing. The route still performs a coarse upload-size validation before any expensive work begins.

### File Processing Service

The current `FileProcessingService` grows a dedicated audio processor path rather than a separate ingestion system. The processor selection flow becomes:

- text/json/pdf processors for existing behavior
- audio processor for `.mp3` and `.wav`

The audio processor is responsible for:

- validating audio media type and extension
- reading the file bytes
- deciding whether a single provider transcription call is safe
- splitting large audio files into provider-safe segments when necessary
- transcribing those segments with bounded concurrency
- reassembling the final transcript in original segment order

Its output is still just transcript text.

### Message Creation

Once transcript text is produced, the system reuses the existing text chunking and message creation flow:

1. transcript text is split with the existing message-size chunker
2. `MessageCreate` objects are built
3. created DB messages receive file and processing metadata in `internal_metadata`
4. the existing enqueue flow sends the resulting text messages into the deriver

No deriver-specific branching is introduced for audio uploads.

## Limits And Configuration

### Public Behavior

Managed Honcho should treat audio limits as fixed platform behavior. Clients should not be able to tune them per request, and SDKs should not expose audio-specific request knobs for this feature.

### Self-Hosted Configuration

Self-hosted deployments may configure audio processing through server settings. These settings are deployment-level only and are not exposed in the HTTP contract. The likely settings are:

- `MAX_AUDIO_FILE_SIZE_BYTES`
- `MAX_AUDIO_CHUNK_DURATION_SECONDS`
- `MAX_AUDIO_CHUNK_BYTES`
- `AUDIO_TRANSCRIPTION_CONCURRENCY`

The existing global `MAX_FILE_SIZE` remains the generic upload gate. Audio-specific settings refine behavior after the route recognizes an audio upload. For managed deployments, Honcho provides fixed defaults for these values.

### Why Separate Audio Limits

Audio needs larger and more specialized limits than text/PDF uploads. Raising the existing global file-size setting alone would make all uploads looser and would not solve provider-specific transcription constraints. Audio-specific limits allow Honcho to keep current behavior for existing file types while safely supporting long recordings.

## Chunking Strategy

Chunking happens in two phases.

### Phase 1: Raw Audio Segmentation

Large audio files are split before transcription into contiguous, ordered segments based on provider-safe size or duration thresholds. Segment boundaries are derived from deployment settings or managed defaults. This prevents a large upload from failing at the transcription provider due to size or duration caps.

For small audio files, the server should skip segmentation and issue a single transcription request. The behavior is therefore hybrid:

- small safe audio: one provider request
- large audio: many provider requests across ordered segments

### Phase 2: Transcript Text Chunking

After transcription, the ordered transcript text is merged and then passed through the existing `split_text_into_chunks(...)` logic so that every stored message still satisfies `MAX_MESSAGE_SIZE` and the existing `MessageCreate` validation.

This produces ordinary text messages the rest of Honcho already understands.

## Parallelism Model

Parallelism exists at two distinct layers.

### Intra-File Parallelism

Within a single upload request, the server may process multiple audio segments concurrently. This is bounded by a semaphore or equivalent concurrency limiter. The server should never fan out unbounded requests for one upload.

Example model:

- one audio file is split into ordered segments `0..N`
- at most `AUDIO_TRANSCRIPTION_CONCURRENCY` segments are sent to the provider at once
- segments may complete out of order
- results are reassembled strictly by original segment index
- only then is the final transcript turned into messages

This improves latency for long uploads without changing message ordering.

### Cross-Request Parallelism

For migration-scale ingestion, throughput should primarily come from the migration client sending many one-file upload requests concurrently. Honcho already allows this pattern because the endpoint contract is one file per request. Audio support should preserve that model rather than introducing a bulk-file API.

These two layers combine cleanly:

- many uploads may be in flight across the migration client
- each upload may also process a few audio segments in parallel internally

## Ordering Guarantees

Deterministic ordering is required so migration results are reproducible.

- Audio segments must be indexed at segmentation time.
- Transcription results must be sorted by segment index before merging.
- Final transcript text must preserve original temporal order.
- Message chunk creation must preserve transcript order.
- `created_at` handling should remain consistent with the caller-provided value or current upload behavior.

Parallel processing must never change the final transcript ordering.

## Metadata

Existing file metadata should be preserved and extended for observability. Each created message should include the existing file metadata plus enough audio provenance to debug migration behavior. Likely additions in `internal_metadata` include:

- `processing_type: "audio_transcription"`
- `audio_segment_count`
- `audio_segment_index_range` or equivalent provenance for the source portion
- `transcription_provider`
- `transcript_char_range`
- existing `chunk_index` and `total_chunks`

The exact metadata keys should stay compact and operationally useful. The goal is to reconstruct how a final message was derived from an uploaded audio file without exposing a new public schema.

## Error Handling

Expected failure cases:

- unsupported media type
- file too large for the route-level gate
- file too large for audio-specific policy
- provider transcription failure
- partial segment transcription failure
- empty or unusable transcript output

Design rules:

- unsupported types return the existing unsupported media behavior
- upload-size violations fail before transcription work starts
- if any required segment fails transcription, the upload should fail rather than creating partial transcript messages by default
- error responses should remain aligned with the existing upload route semantics

Failing the whole upload on segment failure keeps migration behavior easier to reason about than partial success within one file.

## Testing Strategy

Add focused tests around the existing upload route and file processing helpers.

Coverage should include:

- `.mp3` and `.wav` accepted through `/messages/upload`
- transcript output becomes ordinary created messages
- small audio triggers single-request transcription behavior
- large audio triggers segmentation before transcription
- bounded parallel transcription preserves final ordering
- merged transcript is text-chunked into multiple messages when needed
- audio metadata is stored in `internal_metadata`
- route rejects oversized audio files
- route rejects unsupported non-audio binary formats
- transcription provider failures fail the upload cleanly

Where practical, tests should mock the transcription provider and audio segmenter so the suite validates behavior without requiring real external calls.

## Rollout Notes

The smallest coherent implementation is:

1. add audio processor support under the existing file processing path
2. add deployment-level audio settings with managed defaults
3. add provider client wrapper for audio transcription
4. add segmentation + bounded concurrency + ordered merge
5. add route/helper tests
6. update docs for self-hosted configuration and supported file types

This keeps the change local to the existing upload architecture and avoids a second ingestion system.
