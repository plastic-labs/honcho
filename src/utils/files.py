import datetime
import base64
import logging
from io import BytesIO
from typing import Any, Protocol

import httpx
from fastapi import UploadFile
from nanoid import generate as generate_nanoid
from sqlalchemy import Integer, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import schemas
from src.config import settings
from src.exceptions import (
    FileProcessingError,
    UnsupportedFileTypeError,
    ValidationException,
)
from src.schemas import Message

logger = logging.getLogger(__name__)


class FileProcessor(Protocol):
    async def extract_text(self, content: bytes, content_type: str) -> str: ...
    def supports_file_type(self, content_type: str) -> bool: ...


def _native_pdf_text(content: bytes) -> str:
    import pdfplumber

    with pdfplumber.open(BytesIO(content)) as pdf_reader:
        text_parts: list[str] = []
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text and text.strip():
                text_parts.append(f"[Page {page_num + 1}]\n{text}")
        return "\n\n".join(text_parts)


def _ocr_endpoint() -> str:
    if settings.OCR.PROVIDER == "mistral":
        return "https://api.mistral.ai/v1/ocr"

    base_url = settings.OCR.DEEPSEEK_BASE_URL
    if not base_url:
        raise UnsupportedFileTypeError("DeepSeek-compatible OCR endpoint not configured")
    return (
        base_url
        if base_url.rstrip("/").endswith("/ocr")
        else f"{base_url.rstrip('/')}/ocr"
    )


def _ocr_headers() -> dict[str, str]:
    if settings.OCR.PROVIDER == "mistral":
        return {
            "Authorization": f"Bearer {settings.OCR.MISTRAL_API_KEY}",
            "Content-Type": "application/json",
        }

    headers = {"Content-Type": "application/json"}
    if settings.OCR.DEEPSEEK_API_KEY:
        headers["Authorization"] = f"Bearer {settings.OCR.DEEPSEEK_API_KEY}"
    return headers


def _ocr_payload(content: bytes, content_type: str) -> dict[str, Any]:
    encoded = base64.b64encode(content).decode("ascii")
    data_url = f"data:{content_type};base64,{encoded}"

    document: dict[str, Any]
    if content_type.startswith("image/"):
        document = {"type": "image_url", "image_url": data_url}
    else:
        document = {"type": "document_url", "document_url": data_url}

    payload = {"document": document}
    if settings.OCR.PROVIDER == "mistral":
        payload["model"] = settings.OCR.MISTRAL_MODEL
        payload["include_image_base64"] = False
    elif settings.OCR.DEEPSEEK_MODEL:
        payload["model"] = settings.OCR.DEEPSEEK_MODEL
    return payload


def _coerce_ocr_text(response_json: dict[str, Any]) -> str:
    pages = response_json.get("pages")
    if isinstance(pages, list):
        text_parts: list[str] = []
        for page in pages:
            if not isinstance(page, dict):
                continue
            text = page.get("markdown") or page.get("text")
            if isinstance(text, str) and text.strip():
                index = len(text_parts) + 1
                text_parts.append(f"[Page {index}]\n{text.strip()}")
        if text_parts:
            return "\n\n".join(text_parts)

    for key in ("markdown", "text", "content"):
        value = response_json.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    raise ValueError("OCR response did not contain extracted text")


async def _ocr_extract_text(content: bytes, content_type: str) -> str:
    if settings.OCR.MODE == "off":
        raise UnsupportedFileTypeError("OCR is not enabled")

    async with httpx.AsyncClient(timeout=settings.OCR.TIMEOUT_SECONDS) as client:
        response = await client.post(
            _ocr_endpoint(),
            headers=_ocr_headers(),
            json=_ocr_payload(content, content_type),
        )
        response.raise_for_status()
        return _coerce_ocr_text(response.json())


class PDFProcessor:
    def supports_file_type(self, content_type: str) -> bool:
        return content_type == "application/pdf"

    async def extract_text(self, content: bytes, content_type: str) -> str:
        if settings.OCR.MODE == "off":
            return _native_pdf_text(content)

        if settings.OCR.MODE == "force":
            return await _ocr_extract_text(content, content_type)

        native_text = ""
        try:
            native_text = _native_pdf_text(content)
        except Exception:
            logger.warning("Native PDF extraction failed; trying OCR", exc_info=True)

        if (
            settings.OCR.MODE == "fallback"
            and len(native_text.strip()) >= settings.OCR.MIN_EXTRACTED_TEXT_CHARS
        ):
            return native_text

        try:
            return await _ocr_extract_text(content, content_type)
        except Exception as e:
            if settings.OCR.MODE == "fallback" and native_text.strip():
                logger.warning(
                    "OCR failed for PDF upload, falling back to native text: %s", e
                )
                return native_text
            raise


class TextProcessor:
    def supports_file_type(self, content_type: str) -> bool:
        return content_type.startswith("text/")

    async def extract_text(self, content: bytes, content_type: str) -> str:
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

    async def extract_text(self, content: bytes, content_type: str) -> str:
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


class ImageProcessor:
    def supports_file_type(self, content_type: str) -> bool:
        return content_type.startswith("image/") and settings.OCR.MODE != "off"

    async def extract_text(self, content: bytes, content_type: str) -> str:
        return await _ocr_extract_text(content, content_type)


class FileProcessingService:
    def __init__(self):
        self.processors: list[FileProcessor] = [
            PDFProcessor(),
            TextProcessor(),
            JSONProcessor(),
        ]
        if settings.OCR.MODE != "off":
            self.processors.append(ImageProcessor())

    async def extract_text_from_upload(self, file: UploadFile) -> str:
        """Extract text from uploaded file without saving to disk."""
        content = await file.read()

        # Reset file position in case it's needed again
        await file.seek(0)

        processor = self._get_processor(file.content_type or "")
        if not processor:
            raise UnsupportedFileTypeError(
                f"Unsupported file type: {file.content_type}. Supported types: {self._supported_types()}"
            )

        return await processor.extract_text(content, file.content_type or "")

    def _get_processor(self, content_type: str) -> FileProcessor | None:
        for processor in self.processors:
            if processor.supports_file_type(content_type):
                return processor
        return None

    def _supported_types(self) -> list[str]:
        supported_types = ["application/pdf", "text/*", "application/json"]
        if any(isinstance(processor, ImageProcessor) for processor in self.processors):
            supported_types.append("image/*")
        return supported_types


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

    # Process the uploaded file
    extracted_text = await file_processor.extract_text_from_upload(file)

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

        all_message_data.append(
            {
                "message_create": message_create,
                "file_metadata": file_metadata,
            }
        )

    if not all_message_data:
        raise FileProcessingError()

    return all_message_data
