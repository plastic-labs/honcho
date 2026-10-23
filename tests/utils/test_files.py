import json

import pytest

from src.exceptions import ValidationException
from src.utils.files import JSONProcessor, PDFProcessor


@pytest.mark.asyncio
async def test_json_processor_returns_empty_string_for_blank_content():
    processor = JSONProcessor()

    assert await processor.extract_text(b"") == ""
    assert await processor.extract_text(b"   \n\t") == ""


@pytest.mark.asyncio
async def test_json_processor_preserves_valid_json_behavior():
    processor = JSONProcessor()

    result = await processor.extract_text(b'{"name": "test", "count": 1}')

    assert json.loads(result) == {"name": "test", "count": 1}


@pytest.mark.asyncio
async def test_json_processor_rejects_non_utf8_content():
    processor = JSONProcessor()

    with pytest.raises(ValidationException, match="UTF-8"):
        await processor.extract_text(b"\xff\xfe\x00{")


@pytest.mark.asyncio
async def test_json_processor_rejects_invalid_json_content():
    processor = JSONProcessor()

    with pytest.raises(ValidationException, match="invalid"):
        await processor.extract_text(b'{"name": }')


@pytest.mark.asyncio
async def test_pdf_processor_rejects_builtin_error_raised_by_pypdf():
    # pypdf raises a bare AttributeError on this truncated object table
    malformed = b"%PDF-1.4\n1 0 obj\n00 n \ntrailer\n<</Size 6 /Root 1 0 R>>\nstartxref\n394\n%%EOF"

    with pytest.raises(ValidationException, match="Uploaded PDF could not be read"):
        await PDFProcessor().extract_text(malformed)


@pytest.mark.asyncio
async def test_pdf_processor_does_not_mask_builtin_errors_from_other_code(
    monkeypatch: pytest.MonkeyPatch,
):
    def broken_reader(_stream: object) -> None:
        raise TypeError("bug outside pypdf")

    monkeypatch.setattr("pypdf.PdfReader", broken_reader)

    with pytest.raises(TypeError, match="bug outside pypdf"):
        await PDFProcessor().extract_text(b"%PDF-1.4\n")
