import asyncio
import io
import math
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import UploadFile
from openai import AsyncOpenAI
from starlette.datastructures import Headers

from src.config import settings
from src.exceptions import ValidationException
from src.utils.clients import CLIENTS, transcribe_audio
from src.utils.files import (
    AudioProcessor,
    AudioSegment,
    FileProcessingService,
)


def _generate_test_audio_bytes(
    audio_format: str,
    duration_seconds: int = 1,
    *,
    audio_bitrate: str | None = None,
) -> bytes:
    suffix = f".{audio_format}"
    with tempfile.TemporaryDirectory(prefix="honcho-audio-test-") as temp_dir:
        output_path = Path(temp_dir) / f"tone{suffix}"
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration_seconds}",
        ]
        if audio_bitrate is not None:
            command.extend(["-b:a", audio_bitrate])
        command.append(str(output_path))
        subprocess.run(command, check=True, capture_output=True, text=True)
        return output_path.read_bytes()


def test_audio_processor_supports_mp3_and_wav_content_types():
    processor = AudioProcessor()

    assert processor.supports_file_type("audio/mpeg")
    assert processor.supports_file_type("audio/wave")
    assert processor.supports_file_type("audio/wav")
    assert processor.supports_file_type("audio/x-wav")
    assert not processor.supports_file_type("text/plain")


def test_audio_defaults_use_openai_whisper_without_backup():
    assert settings.AUDIO.PROVIDER == "openai"
    assert settings.AUDIO.MODEL == "whisper-1"


@pytest.mark.asyncio
async def test_audio_upload_requires_openai_client_before_processing():
    file = UploadFile(
        file=io.BytesIO(b"audio-bytes"),
        filename="voice.mp3",
        headers=Headers({"content-type": "audio/mpeg"}),
    )
    service = FileProcessingService()

    with (
        patch.dict(CLIENTS, {}, clear=True),
        patch.object(service.audio_processor, "extract_text", new=AsyncMock()) as mock_extract,
        pytest.raises(
            ValidationException,
            match="Audio uploads require OpenAI transcription credentials",
        ),
    ):
        await service.extract_text_from_upload(file)

    mock_extract.assert_not_awaited()


@pytest.mark.asyncio
async def test_filename_audio_extension_does_not_override_explicit_text_plain_mime():
    file = UploadFile(
        file=io.BytesIO(b"plain text body"),
        filename="notes.mp3",
        headers=Headers({"content-type": "text/plain"}),
    )
    service = FileProcessingService()

    with patch.object(service.audio_processor, "extract_text", new=AsyncMock()) as mock_extract:
        extracted = await service.extract_text_from_upload(file)

    assert extracted.text == "plain text body"
    mock_extract.assert_not_awaited()


@pytest.mark.asyncio
async def test_transcribe_audio_uses_openai_whisper():
    mock_openai = AsyncMock(spec=AsyncOpenAI)
    mock_openai.audio.transcriptions.create = AsyncMock(return_value="hello from whisper")

    with patch.dict(CLIENTS, {"openai": mock_openai}, clear=False):
        text = await transcribe_audio(
            b"audio-bytes",
            filename="clip.mp3",
            content_type="audio/mpeg",
        )

    assert text == "hello from whisper"
    mock_openai.audio.transcriptions.create.assert_awaited_once()
    call = mock_openai.audio.transcriptions.create.await_args
    assert call is not None
    assert call.kwargs["model"] == "whisper-1"
    assert call.kwargs["response_format"] == "text"


@pytest.mark.asyncio
async def test_transcribe_audio_allows_empty_transcript_for_silence():
    mock_openai = AsyncMock(spec=AsyncOpenAI)
    mock_openai.audio.transcriptions.create = AsyncMock(return_value="")

    with patch.dict(CLIENTS, {"openai": mock_openai}, clear=False):
        text = await transcribe_audio(
            b"audio-bytes",
            filename="clip.mp3",
            content_type="audio/mpeg",
        )

    assert text == ""


@pytest.mark.asyncio
async def test_transcribe_audio_raises_when_openai_fails():
    mock_openai = AsyncMock(spec=AsyncOpenAI)
    mock_openai.audio.transcriptions.create = AsyncMock(side_effect=RuntimeError("openai failed"))

    with (
        patch.dict(CLIENTS, {"openai": mock_openai}, clear=False),
        pytest.raises(RuntimeError, match="openai failed"),
    ):
        await transcribe_audio(
            b"audio-bytes",
            filename="clip.mp3",
            content_type="audio/mpeg",
        )


@pytest.mark.asyncio
async def test_transcribe_segments_preserves_order_when_tasks_finish_out_of_order():
    processor = AudioProcessor()
    segments = [
        AudioSegment(index=0, filename="seg-0.mp3", content=b"0"),
        AudioSegment(index=1, filename="seg-1.mp3", content=b"1"),
        AudioSegment(index=2, filename="seg-2.mp3", content=b"2"),
    ]

    async def fake_transcribe(
        _content: bytes,
        filename: str,
        content_type: str,
        **_: object,
    ):
        assert content_type == "audio/mpeg"
        if filename == "seg-0.mp3":
            await asyncio.sleep(0.03)
            return "first"
        if filename == "seg-1.mp3":
            await asyncio.sleep(0.0)
            return "second"
        await asyncio.sleep(0.01)
        return "third"

    with patch("src.utils.files.transcribe_audio", side_effect=fake_transcribe):
        extracted = await processor.transcribe_segments(
            segments,
            content_type="audio/mpeg",
            concurrency=3,
        )

    assert extracted.text == "first\nsecond\nthird"
    assert extracted.metadata["processing_type"] == "audio_transcription"
    assert extracted.metadata["audio_segment_count"] == 3
    assert extracted.metadata["transcription_provider"] == "openai"
    assert "transcription_fallback_used" not in extracted.metadata


@pytest.mark.asyncio
async def test_transcribe_segments_ignores_empty_silent_segments():
    processor = AudioProcessor()
    segments = [
        AudioSegment(index=0, filename="seg-0.mp3", content=b"0"),
        AudioSegment(index=1, filename="seg-1.mp3", content=b"1"),
        AudioSegment(index=2, filename="seg-2.mp3", content=b"2"),
    ]

    async def fake_transcribe(
        _content: bytes,
        filename: str,
        content_type: str,
        **_: object,
    ):
        assert content_type == "audio/mpeg"
        if filename == "seg-0.mp3":
            return "first"
        if filename == "seg-1.mp3":
            return ""
        return "third"

    with patch("src.utils.files.transcribe_audio", side_effect=fake_transcribe):
        extracted = await processor.transcribe_segments(
            segments,
            content_type="audio/mpeg",
            concurrency=3,
        )

    assert extracted.text == "first\nthird"
    assert extracted.metadata["audio_segment_count"] == 3


@pytest.mark.asyncio
async def test_audio_processor_normalizes_octet_stream_mp3_uploads():
    processor = AudioProcessor()

    async def fake_transcribe(
        _content: bytes,
        filename: str,
        content_type: str,
        **_: object,
    ) -> str:
        assert filename == "voice-note.mp3"
        assert content_type == "audio/mpeg"
        return "normalized"

    with patch.object(
        processor,
        "split_audio_segments",
        return_value=[AudioSegment(index=0, filename="voice-note.mp3", content=b"bytes")],
    ), patch("src.utils.files.transcribe_audio", side_effect=fake_transcribe):
        extracted = await processor.extract_text(
            b"bytes",
            filename="voice-note.mp3",
            content_type="application/octet-stream",
        )

    assert extracted.text == "normalized"


@pytest.mark.asyncio
async def test_empty_audio_upload_is_rejected_before_transcription():
    processor = AudioProcessor()

    with patch.object(
        processor,
        "transcribe_segments",
        new=AsyncMock(),
    ) as mock_transcribe, pytest.raises(
        ValidationException,
        match="Audio upload is empty",
    ):
        await processor.extract_text(
            b"",
            filename="empty.mp3",
            content_type="audio/mpeg",
        )

    mock_transcribe.assert_not_awaited()


@pytest.mark.asyncio
async def test_audio_wave_mime_is_accepted_for_wav_uploads():
    processor = AudioProcessor()

    async def fake_transcribe(
        _content: bytes,
        filename: str,
        content_type: str,
        **_: object,
    ) -> str:
        assert filename == "recording.wav"
        assert content_type == "audio/wave"
        return "wav accepted"

    with patch.object(
        processor,
        "split_audio_segments",
        return_value=[AudioSegment(index=0, filename="recording.wav", content=b"bytes")],
    ), patch("src.utils.files.transcribe_audio", side_effect=fake_transcribe):
        extracted = await processor.extract_text(
            b"bytes",
            filename="recording.wav",
            content_type="audio/wave",
        )

    assert extracted.text == "wav accepted"


def test_split_audio_segments_raises_validation_when_ffprobe_missing():
    processor = AudioProcessor()

    with (
        patch("src.utils.files.subprocess.run", side_effect=FileNotFoundError("ffprobe")),
        pytest.raises(
            ValidationException,
            match="Audio uploads require ffmpeg and ffprobe to be installed",
        ),
    ):
        processor.split_audio_segments(
            b"audio-bytes",
            filename="voice.mp3",
            content_type="audio/mpeg",
        )


def test_split_audio_segments_splits_long_wav_when_duration_exceeds_limit():
    processor = AudioProcessor()
    wav_bytes = _generate_test_audio_bytes("wav", duration_seconds=3)

    original_duration_limit = settings.AUDIO.MAX_CHUNK_DURATION_SECONDS
    original_chunk_bytes = settings.AUDIO.MAX_CHUNK_BYTES
    settings.AUDIO.MAX_CHUNK_DURATION_SECONDS = 1
    settings.AUDIO.MAX_CHUNK_BYTES = max(len(wav_bytes) * 2, 1)
    try:
        segments = processor.split_audio_segments(
            wav_bytes,
            filename="long.wav",
            content_type="audio/wav",
        )
    finally:
        settings.AUDIO.MAX_CHUNK_DURATION_SECONDS = original_duration_limit
        settings.AUDIO.MAX_CHUNK_BYTES = original_chunk_bytes

    assert len(segments) >= 3
    assert [segment.index for segment in segments] == list(range(len(segments)))
    assert all(segment.filename.endswith(".wav") for segment in segments)
    assert all(segment.content for segment in segments)


def test_split_audio_segments_raises_validation_for_invalid_audio_bytes():
    processor = AudioProcessor()

    with pytest.raises(
        ValidationException,
        match="Uploaded audio is invalid or unreadable",
    ):
        processor.split_audio_segments(
            b"not-valid-audio",
            filename="broken.mp3",
            content_type="audio/mpeg",
        )


def test_split_audio_segments_keep_low_bitrate_mp3_chunks_under_limit():
    processor = AudioProcessor()
    mp3_bytes = _generate_test_audio_bytes(
        "mp3",
        duration_seconds=12,
        audio_bitrate="64k",
    )

    original_duration_limit = settings.AUDIO.MAX_CHUNK_DURATION_SECONDS
    original_chunk_bytes = settings.AUDIO.MAX_CHUNK_BYTES
    max_chunk_bytes = max(math.ceil(len(mp3_bytes) / 2), 1)
    settings.AUDIO.MAX_CHUNK_DURATION_SECONDS = 60
    settings.AUDIO.MAX_CHUNK_BYTES = max_chunk_bytes
    try:
        segments = processor.split_audio_segments(
            mp3_bytes,
            filename="voice-note.mp3",
            content_type="audio/mpeg",
        )
    finally:
        settings.AUDIO.MAX_CHUNK_DURATION_SECONDS = original_duration_limit
        settings.AUDIO.MAX_CHUNK_BYTES = original_chunk_bytes

    assert len(segments) >= 3
    assert all(len(segment.content) <= max_chunk_bytes for segment in segments)
