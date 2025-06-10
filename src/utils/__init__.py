"""
Utility modules for the Honcho app.
"""

import re
from datetime import datetime, timezone
from typing import Optional

from src import crud, models, schemas


def parse_xml_content(text: str, tag: str) -> str:
    """
    Extract content from XML-like tags in a string.

    Args:
        text: The text containing XML-like tags
        tag: The tag name to extract content from

    Returns:
        The content between the opening and closing tags, or an empty string if not found
    """
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


async def update_document_access_metadata(
    db, 
    document: models.Document, 
    app_id: str, 
    user_id: str, 
    collection_id: str,
    session_id: Optional[str] = None,
    message_id: Optional[str] = None
):
    """Update access metadata for a document when it's accessed/retrieved."""
    try:
        # Update the document's metadata with new access information
        updated_metadata = document.h_metadata.copy() if document.h_metadata else {}
        
        # Increment access count
        updated_metadata["access_count"] = updated_metadata.get("access_count", 0) + 1
        updated_metadata["last_accessed"] = datetime.now(timezone.utc).isoformat()
        
        # Track accessed sessions
        if session_id:
            accessed_sessions = updated_metadata.get("accessed_sessions", [])
            if session_id not in accessed_sessions:
                accessed_sessions.append(session_id)
                updated_metadata["accessed_sessions"] = accessed_sessions
        
        # Update the document
        document_update = schemas.DocumentUpdate(metadata=updated_metadata)
        await crud.update_document(
            db,
            document=document_update,
            app_id=app_id,
            user_id=user_id,
            collection_id=collection_id,
            document_id=document.public_id
        )
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Error updating document access metadata: {e}")
