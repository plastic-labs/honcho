import pytest

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

    assert '"name": "test"' in result
    assert '"count": 1' in result
