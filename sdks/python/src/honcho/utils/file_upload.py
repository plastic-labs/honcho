"""
File upload utilities for the Honcho Python SDK.

This module provides shared functionality for handling file uploads across
both sync and async client implementations.
"""

import mimetypes
from io import BytesIO, IOBase


def normalize_file_input(
    file: tuple[str, bytes, str] | tuple[str, IOBase, str] | IOBase,
) -> tuple[str, IOBase, str]:
    """
    Normalize various file input formats to a standard tuple format.

    Args:
        file: File to normalize. Can be:
            - a file object (must have .name and .read())
            - a tuple (filename, bytes, content_type)
            - a tuple (filename, fileobj, content_type)

    Returns:
        A normalized tuple of (filename, fileobj, content_type)

    Raises:
        ValueError: If the file input format is not supported
    """
    # If it's a tuple (filename, bytes, content_type)
    if isinstance(file, tuple) and len(file) == 3:
        filename, file_content, content_type = file
        if isinstance(file_content, bytes):
            fileobj = BytesIO(file_content)
            fileobj.name = filename
            return (filename, fileobj, content_type)
        elif isinstance(file_content, IOBase):  # pyright: ignore -- needed for return type
            return (filename, file_content, content_type)
        else:
            raise ValueError("File content must be bytes or a file-like object.")

    # If it's a file object (not str/bytes/bytearray/memoryview)
    elif isinstance(file, IOBase):  # pyright: ignore -- needed for return type
        filename = getattr(file, "name", None)
        if not filename:
            raise ValueError("File object must have a .name attribute.")
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        return (filename, file, content_type)


def prepare_file_for_upload(
    file: tuple[str, bytes, str] | tuple[str, IOBase, str] | IOBase,
) -> tuple[str, bytes, str]:
    """
    Prepare a file for upload by normalizing and reading its content.

    Args:
        file: File to prepare. Can be:
            - a file object (must have .name and .read())
            - a tuple (filename, bytes, content_type)
            - a tuple (filename, fileobj, content_type)

    Returns:
        A tuple of (filename, content_bytes, content_type) ready for API upload

    Raises:
        ValueError: If the file input format is not supported
    """
    normalized_file = normalize_file_input(file)

    # Read the file content
    normalized_file[1].seek(0)  # Reset file position
    content_bytes = normalized_file[1].read()

    return (normalized_file[0], content_bytes, normalized_file[2])
