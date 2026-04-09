import asyncio
import io
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import UploadFile
from openai import AsyncOpenAI
from starlette.datastructures import Headers

from src.config import settings
from src.exceptions import ValidationException
from src.utils.clients import CLIENTS, transcribe_audio
from src.utils.files import AudioProcessor, FileProcessingService


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
async def test_audio_processor_extract_text_transcribes_directly():
    processor = AudioProcessor()

    async def fake_transcribe(
        _content: bytes,
        filename: str,
        content_type: str,
        **_: object,
    ):
        assert content_type == "audio/mpeg"
        assert filename == "seg-0.mp3"
        await asyncio.sleep(0.01)
        return "first"

    with (
        patch.object(processor, "_probe_audio_duration_seconds", return_value=1.0),
        patch("src.utils.files.transcribe_audio", side_effect=fake_transcribe),
    ):
        extracted = await processor.extract_text(
            b"audio-bytes",
            filename="seg-0.mp3",
            content_type="audio/mpeg",
        )

    assert extracted.text == "first"
    assert extracted.metadata["processing_type"] == "audio_transcription"
    assert extracted.metadata["audio_segment_count"] == 1
    assert extracted.metadata["transcription_provider"] == "openai"
    assert "transcription_fallback_used" not in extracted.metadata


@pytest.mark.asyncio
async def test_audio_processor_extract_text_allows_empty_transcript():
    processor = AudioProcessor()

    async def fake_transcribe(
        _content: bytes,
        filename: str,
        content_type: str,
        **_: object,
    ):
        assert content_type == "audio/mpeg"
        assert filename == "voice-note.mp3"
        return ""

    with (
        patch.object(processor, "_probe_audio_duration_seconds", return_value=1.0),
        patch("src.utils.files.transcribe_audio", side_effect=fake_transcribe),
    ):
        extracted = await processor.extract_text(
            b"bytes",
            filename="voice-note.mp3",
            content_type="audio/mpeg",
        )

    assert extracted.text == ""
    assert extracted.metadata["audio_segment_count"] == 1


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

    with (
        patch.object(processor, "_probe_audio_duration_seconds", return_value=1.0),
        patch("src.utils.files.transcribe_audio", side_effect=fake_transcribe),
    ):
        extracted = await processor.extract_text(
            b"bytes",
            filename="voice-note.mp3",
            content_type="application/octet-stream",
        )

    assert extracted.text == "normalized"


@pytest.mark.asyncio
async def test_empty_audio_upload_is_rejected_before_transcription():
    processor = AudioProcessor()

    with patch("src.utils.files.transcribe_audio", new=AsyncMock()) as mock_transcribe, pytest.raises(
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

    with (
        patch.object(processor, "_probe_audio_duration_seconds", return_value=1.0),
        patch("src.utils.files.transcribe_audio", side_effect=fake_transcribe),
    ):
        extracted = await processor.extract_text(
            b"bytes",
            filename="recording.wav",
            content_type="audio/wave",
        )

    assert extracted.text == "wav accepted"


@pytest.mark.asyncio
async def test_audio_processor_normalizes_small_mime_only_audio_filename():
    processor = AudioProcessor()

    async def fake_transcribe(
        _content: bytes,
        filename: str,
        content_type: str,
        **_: object,
    ) -> str:
        assert filename == "blob.mp3"
        assert content_type == "audio/mpeg"
        return "normalized"

    with (
        patch.object(processor, "_probe_audio_duration_seconds", return_value=1.0),
        patch("src.utils.files.transcribe_audio", side_effect=fake_transcribe),
    ):
        extracted = await processor.extract_text(
            b"audio-bytes",
            filename="blob",
            content_type="audio/mpeg",
        )

    assert extracted.text == "normalized"
    assert extracted.metadata["audio_segment_count"] == 1
