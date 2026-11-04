import pytest
import tiktoken
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from src.config import settings
from src.schemas import ConclusionCreate, MessageCreate
from src.utils.tokens import estimate_tokens
from tests.tokenizer_strategies import TOKENIZER_PROPERTIES, special_token_content


@TOKENIZER_PROPERTIES
@given(content=special_token_content("o200k_base"), include_nul=st.booleans())
def test_message_special_token_content_is_counted_as_ordinary_text(
    content: str, include_nul: bool
) -> None:
    submitted = "\x00" + content + "\x00" if include_nul else content
    message = MessageCreate(content=submitted, peer_id="alice")

    assert message.content == content
    assert message.encoded_message == tiktoken.get_encoding(
        "o200k_base"
    ).encode_ordinary(content)


@TOKENIZER_PROPERTIES
@given(content=special_token_content("o200k_base"))
def test_conclusion_special_token_content_respects_token_limit(content: str) -> None:
    count = len(tiktoken.get_encoding("o200k_base").encode_ordinary(content))
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(settings.EMBEDDING, "MAX_INPUT_TOKENS", count)
        conclusion = ConclusionCreate(
            content=content, observer_id="alice", observed_id="bob"
        )
        assert conclusion.content == content

        monkeypatch.setattr(settings.EMBEDDING, "MAX_INPUT_TOKENS", count - 1)
        with pytest.raises(ValidationError, match="maximum embedding token limit"):
            ConclusionCreate(content=content, observer_id="alice", observed_id="bob")


@TOKENIZER_PROPERTIES
@given(contents=st.lists(special_token_content("o200k_base"), min_size=1, max_size=3))
def test_special_token_estimates_use_exact_ordinary_text_counts(
    contents: list[str],
) -> None:
    encoding = tiktoken.get_encoding("o200k_base")
    for content in contents:
        assert estimate_tokens(content) == len(encoding.encode_ordinary(content))
    assert estimate_tokens(contents) == len(
        encoding.encode_ordinary("\n".join(contents))
    )
