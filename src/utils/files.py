import logging
from io import BytesIO
from typing import Any, Protocol

import pdfplumber
from fastapi import UploadFile
from nanoid import generate as generate_nanoid
from sqlalchemy import Integer, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import schemas
from src.config import settings
from src.exceptions import FileProcessingError, UnsupportedFileTypeError
from src.schemas import Message

logger = logging.getLogger(__name__)


class FileProcessor(Protocol):
    async def extract_text(self, content: bytes) -> str: ...
    def supports_file_type(self, content_type: str) -> bool: ...


class PDFProcessor:
    def supports_file_type(self, content_type: str) -> bool:
        return content_type == "application/pdf"

    async def extract_text(self, content: bytes) -> str:
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

        data = json.loads(content.decode("utf-8"))
        # Convert JSON to readable text format
        return json.dumps(data, ensure_ascii=False)


class FileProcessingService:
    def __init__(self):
        self.processors: list[FileProcessor] = [
            PDFProcessor(),
            TextProcessor(),
            JSONProcessor(),
            # Add more processors as needed
        ]

    async def extract_text_from_upload(self, file: UploadFile) -> str:
        """Extract text from uploaded file without saving to disk."""
        content = await file.read()

        # Reset file position in case it's needed again
        await file.seek(0)

        processor = self._get_processor(file.content_type or "")
        if not processor:
            raise UnsupportedFileTypeError(
                f"Unsupported file type: {file.content_type}. Supported types: {[p.__class__.__name__ for p in self.processors]}"
            )

        return await processor.extract_text(content)

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
) -> list[dict[str, Any]]:
    """
    Process an uploaded file and prepare message creation data.

    This function extracts text from a file, splits it into chunks, and prepares
    the data needed to create messages.

    Args:
        file: Uploaded file to process
        peer_id: ID of the peer creating the messages
        max_chars: Maximum characters per message chunk

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

        # Create message
        message_create = schemas.MessageCreate(
            content=message_content,
            peer_id=peer_id,
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
