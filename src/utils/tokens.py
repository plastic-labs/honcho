import tiktoken

tokenizer = tiktoken.get_encoding("o200k_base")


def estimate_tokens(text: str | list[str] | None) -> int:
    """Estimate token count using tiktoken for text or list of strings."""
    if not text:
        return 0
    if isinstance(text, list):
        text = "\n".join(text)
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return len(text) // 4
