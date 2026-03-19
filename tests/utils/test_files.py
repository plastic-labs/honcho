import json

import pytest

from src.exceptions import ValidationException
from src.utils.files import JSONProcessor


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
