# ruff: noqa: I001
import asyncio
import datetime
from dataclasses import dataclass, field
import logging
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

from fastapi import UploadFile
from nanoid import generate as generate_nanoid
from sqlalchemy import Integer, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import schemas
from src.config import settings
from src.exceptions import FileProcessingError, UnsupportedFileTypeError, ValidationException
from src.schemas import Message
from src.utils.clients import CLIENTS, transcribe_audio

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_CONTENT_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wave",
    "audio/wav",
    "audio/x-wav",
}
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav"}
AUDIO_EXTENSION_CONTENT_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
}
UPLOAD_VALIDATION_CHUNK_BYTES = 1024 * 1024
GENERIC_CONTENT_TYPES = {
    "",
    "application/octet-stream",
}


@dataclass
class FileExtractionResult:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class FileProcessor(Protocol):
    async def extract_text(self, content: bytes) -> str: ...
    def supports_file_type(self, content_type: str) -> bool: ...


class PDFProcessor:
    def supports_file_type(self, content_type: str) -> bool:
        return content_type == "application/pdf"

    async def extract_text(self, content: bytes) -> str:
        import pdfplumber

        with pdfplumber.open(BytesIO(content)) as pdf_reader:
            text_parts: list[str] = []
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    text_parts.append(f"[Page {page_num + 1}]\n{text}")
            return "\n\n".join(text_parts)


class TextProcessor:
    def supports_file_type(self, content_type: str) -> bool:
        return content_type.startswith("text/")

    async def extract_text(self, content: bytes) -> str:
        # Try different encodings
        for encoding in ["utf-8", "utf-16", "latin-1"]:
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode text file")


class JSONProcessor:
    def supports_file_type(self, content_type: str) -> bool:
        return content_type == "application/json"

    async def extract_text(self, content: bytes) -> str:
        import json

        try:
            decoded_content = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValidationException("JSON uploads must be UTF-8 encoded") from exc

        if not decoded_content.strip():
            return ""

        try:
            data = json.loads(decoded_content)
        except json.JSONDecodeError as exc:
            raise ValidationException("Uploaded JSON is invalid") from exc

        # Convert JSON to readable text format
        return json.dumps(data, ensure_ascii=False)


class AudioProcessor:
    def supports_file_type(self, content_type: str) -> bool:
        return content_type in SUPPORTED_AUDIO_CONTENT_TYPES

    def supports_filename(self, filename: str | None) -> bool:
        if not filename:
            return False
        return Path(filename).suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS

    def supports_content(self, *, filename: str | None, content_type: str) -> bool:
        if self.supports_file_type(content_type):
            return True
        if content_type not in GENERIC_CONTENT_TYPES:
            return False
        return self.supports_filename(filename)

    def supports_upload(self, file: UploadFile) -> bool:
        return self.supports_content(
            filename=file.filename,
            content_type=file.content_type or "",
        )

    async def extract_text(
        self,
        content: bytes,
        *,
        filename: str | None,
        content_type: str,
    ) -> FileExtractionResult:
        if not filename:
            raise ValidationException("Audio upload requires a filename")
        if not content:
            raise ValidationException("Audio upload is empty")

        suffix = self.get_output_suffix(filename, content_type)
        normalized_filename = self.ensure_audio_filename(filename, suffix)
        normalized_content_type = self.normalize_content_type(filename, content_type)
        await asyncio.to_thread(
            self._probe_audio_duration_seconds,
            content,
            suffix,
        )
        text = await transcribe_audio(
            content,
            filename=normalized_filename,
            content_type=normalized_content_type,
        )
        return FileExtractionResult(
            text=text,
            metadata={
                "processing_type": "audio_transcription",
                "audio_segment_count": 1,
                "transcription_provider": settings.AUDIO.PROVIDER,
            },
        )

    def normalize_content_type(self, filename: str, content_type: str) -> str:
        if content_type in SUPPORTED_AUDIO_CONTENT_TYPES:
            return content_type
        extension = Path(filename).suffix.lower()
        return AUDIO_EXTENSION_CONTENT_TYPES.get(extension, content_type)

    def get_output_suffix(self, filename: str, content_type: str) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix in SUPPORTED_AUDIO_EXTENSIONS:
            return suffix
        if content_type in {"audio/wave", "audio/wav", "audio/x-wav"}:
            return ".wav"
        return ".mp3"

    def ensure_audio_filename(self, filename: str, suffix: str) -> str:
        path = Path(filename)
        if path.suffix.lower() == suffix:
            return filename
        return f"{filename}{suffix}"

    def _probe_audio_duration_seconds(self, content: bytes, suffix: str) -> float:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        try:
            return self.probe_audio_duration_seconds_from_path(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

    def probe_audio_duration_seconds_from_path(self, path: Path) -> float:
        try:
            command = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
            )
            return max(float(result.stdout.strip()), 0.0)
        except FileNotFoundError as exc:
            raise ValidationException(
                "Audio uploads require ffmpeg and ffprobe to be installed on the server"
            ) from exc
        except (subprocess.CalledProcessError, ValueError) as exc:
            raise ValidationException("Uploaded audio is invalid or unreadable") from exc


def is_audio_upload(file: UploadFile) -> bool:
    return AudioProcessor().supports_upload(file)


def is_audio_transcription_enabled() -> bool:
    return settings.AUDIO.PROVIDER == "openai" and "openai" in CLIENTS


async def is_validated_audio_upload(file: UploadFile) -> bool:
    processor = AudioProcessor()
    if not processor.supports_upload(file):
        return False

    filename = file.filename
    if not filename:
        return False

    content_type = processor.normalize_content_type(filename, file.content_type or "")
    suffix = processor.get_output_suffix(filename, content_type)
    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            while chunk := await file.read(UPLOAD_VALIDATION_CHUNK_BYTES):
                temp_file.write(chunk)

        await asyncio.to_thread(
            processor.probe_audio_duration_seconds_from_path,
            temp_path,
        )
        return True
    except ValidationException as exc:
        if str(exc) == "Uploaded audio is invalid or unreadable":
            return False
        raise
    finally:
        await file.seek(0)
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


class FileProcessingService:
    def __init__(self):
        self.audio_processor: AudioProcessor = AudioProcessor()
        self.processors: list[FileProcessor] = [
            PDFProcessor(),
            TextProcessor(),
            JSONProcessor(),
            # Add more processors as needed
        ]

    async def extract_text_from_upload(self, file: UploadFile) -> FileExtractionResult:
        """Extract text from uploaded file without saving to disk."""
        content = await file.read()
        await file.seek(0)
        normalized_content_type = file.content_type or ""

        if self.audio_processor.supports_content(
            filename=file.filename,
            content_type=normalized_content_type,
        ):
            if "openai" not in CLIENTS:
                raise ValidationException(
                    "Audio uploads require OpenAI transcription credentials"
                )
            return await self.audio_processor.extract_text(
                content,
                filename=file.filename,
                content_type=normalized_content_type,
            )

        processor = self._get_processor(normalized_content_type)
        if not processor:
            raise UnsupportedFileTypeError(
                f"Unsupported file type: {file.content_type}. Supported types: {[p.__class__.__name__ for p in self.processors]}"
            )

        return FileExtractionResult(text=await processor.extract_text(content))

    def _get_processor(self, content_type: str) -> FileProcessor | None:
        for processor in self.processors:
            if processor.supports_file_type(content_type):
                return processor
        return None


def split_text_into_chunks(text: str, max_chars: int = 49500) -> list[str]:
    """Split text into chunks that fit within message limits."""
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current_pos = 0

    while current_pos < len(text):
        # Try to break at paragraph boundaries first
        end_pos = current_pos + max_chars

        if end_pos >= len(text):
            chunks.append(text[current_pos:])
            break

        # Look for good break points (paragraph, sentence, word)
        break_pos = end_pos
        for delimiter in ["\n\n", "\n", ". ", " "]:
            last_delimiter = text.rfind(delimiter, current_pos, end_pos)
            if last_delimiter > current_pos:
                break_pos = last_delimiter + len(delimiter)
                break

        chunks.append(text[current_pos:break_pos])
        current_pos = break_pos

    return chunks


async def get_file_messages(
    db: AsyncSession,
    workspace_name: str,
    file_id: str,
    session_name: str | None = None,
) -> list[Message]:
    """Get all messages for a specific document, ordered by chunk_index."""
    from sqlalchemy import and_, func

    from src.models import Message

    query = select(Message).where(
        and_(
            Message.workspace_name == workspace_name,
            func.jsonb_extract_path_text(Message.internal_metadata, "file_id")
            == file_id,
        )
    )

    if session_name:
        query = query.where(Message.session_name == session_name)

    # Order by chunk_index
    query = query.order_by(
        func.jsonb_extract_path_text(Message.internal_metadata, "chunk_index").cast(
            Integer
        )
    )

    result = await db.execute(query)
    return list(result.scalars().all())


async def process_file_uploads_for_messages(
    file: UploadFile,
    peer_id: str,
    max_chars: int = settings.MAX_MESSAGE_SIZE,
    metadata: dict[str, Any] | None = None,
    configuration: schemas.MessageConfiguration | None = None,
    created_at: datetime.datetime | None = None,
) -> list[dict[str, Any]]:
    """
    Process an uploaded file and prepare message creation data.

    This function extracts text from a file, splits it into chunks, and prepares
    the data needed to create messages.

    Args:
        file: Uploaded file to process
        peer_id: ID of the peer creating the messages
        max_chars: Maximum characters per message chunk
        metadata: Optional metadata to associate with all messages created from this file
        configuration: Optional configuration to associate with all messages created from this file
        created_at: Optional created_at timestamp to use for all messages created from this file

    Returns:
        List of dictionaries containing message_create and file_metadata

    Raises:
        HTTPException: If file processing fails
    """

    file_processor = FileProcessingService()
    all_message_data: list[dict[str, Any]] = []

    extracted = await file_processor.extract_text_from_upload(file)
    extracted_text = extracted.text

    # Split into chunks and create messages
    chunks = split_text_into_chunks(extracted_text, max_chars=max_chars)
    file_id = generate_nanoid()

    for i, chunk in enumerate(chunks):
        # Build message content properly handling empty files
        message_content = chunk or ""

        # Create message with optional metadata, configuration, and created_at
        message_create = schemas.MessageCreate(
            content=message_content,
            peer_id=peer_id,
            metadata=metadata,
            configuration=configuration,
            created_at=created_at,
        )

        # Store file metadata separately to add to internal_metadata later
        file_metadata = {
            "file_id": file_id,
            "filename": file.filename,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "original_file_size": file.size,
            "content_type": file.content_type,
            "chunk_character_range": [
                i * max_chars,
                min((i + 1) * max_chars, len(extracted_text)),
            ],
        }
        file_metadata.update(extracted.metadata)

        all_message_data.append(
            {
                "message_create": message_create,
                "file_metadata": file_metadata,
            }
        )

    if not all_message_data:
        raise FileProcessingError()

    return all_message_data
