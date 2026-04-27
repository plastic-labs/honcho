import json
from typing import Any

import httpx
import pytest

from src.config import settings
from src.exceptions import ValidationException
from src.utils.files import JSONProcessor, PDFProcessor


class _FakeMistralOCRResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {
            "pages": [
                {"index": 0, "markdown": "# Page 1\nHello"},
                {"index": 1, "markdown": "Page 2 text"},
            ],
            "usage_info": {"pages_processed": 2},
        }


class _FakeAsyncClient:
    posted_json: dict[str, Any] | None = None
    posted_headers: dict[str, str] | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> _FakeMistralOCRResponse:
        assert url == "https://api.mistral.ai/v1/ocr"
        self.__class__.posted_headers = headers
        self.__class__.posted_json = json
        return _FakeMistralOCRResponse()


class _FailingAsyncClient(_FakeAsyncClient):
    async def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> _FakeMistralOCRResponse:
        raise httpx.ConnectError("Mistral unavailable")


class _FakePDFPage:
    def __init__(self, text: str | None) -> None:
        self._text = text

    def extract_text(self) -> str | None:
        return self._text


class _FakePDFReader:
    pages = [_FakePDFPage("First page"), _FakePDFPage(None), _FakePDFPage("Second page")]

    def __enter__(self) -> "_FakePDFReader":
        return self

    def __exit__(self, *args: Any) -> None:
        return None


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
async def test_pdf_processor_extracts_markdown_with_mistral_ocr(monkeypatch):
    processor = PDFProcessor()
    _FakeAsyncClient.posted_json = None
    _FakeAsyncClient.posted_headers = None
    monkeypatch.setattr(settings, "MISTRAL_OCR_API_KEY", "test-mistral-key")
    monkeypatch.setattr(settings, "MISTRAL_OCR_MODEL", "mistral-ocr-test")
    monkeypatch.setattr(settings, "MISTRAL_OCR_TIMEOUT_SECONDS", 12.5)
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    result = await processor.extract_text(b"%PDF test bytes")

    assert result == "# Page 1\nHello\n\nPage 2 text"
    assert _FakeAsyncClient.posted_headers == {
        "Authorization": "Bearer test-mistral-key",
        "Content-Type": "application/json",
    }
    assert _FakeAsyncClient.posted_json == {
        "model": "mistral-ocr-test",
        "document": {
            "type": "document_url",
            "document_url": "data:application/pdf;base64,JVBERiB0ZXN0IGJ5dGVz",
        },
        "include_image_base64": False,
    }


@pytest.mark.asyncio
async def test_pdf_processor_falls_back_to_pdfplumber_without_mistral_key(
    monkeypatch,
):
    processor = PDFProcessor()
    monkeypatch.setattr(settings, "MISTRAL_OCR_API_KEY", None)
    monkeypatch.setattr("src.utils.files.pdfplumber.open", lambda *args: _FakePDFReader())

    result = await processor.extract_text(b"%PDF test bytes")

    assert result == "[Page 1]\nFirst page\n\n[Page 3]\nSecond page"


@pytest.mark.asyncio
async def test_pdf_processor_falls_back_to_pdfplumber_when_mistral_fails(
    monkeypatch,
):
    processor = PDFProcessor()
    monkeypatch.setattr(settings, "MISTRAL_OCR_API_KEY", "test-mistral-key")
    monkeypatch.setattr(httpx, "AsyncClient", _FailingAsyncClient)
    monkeypatch.setattr("src.utils.files.pdfplumber.open", lambda *args: _FakePDFReader())

    result = await processor.extract_text(b"%PDF test bytes")

    assert result == "[Page 1]\nFirst page\n\n[Page 3]\nSecond page"
