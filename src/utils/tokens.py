import tiktoken

from src.telemetry import prometheus

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


def track_deriver_input_tokens(
    task_type: prometheus.DeriverTaskTypes,
    components: dict[prometheus.DeriverComponents, int],
) -> None:
    """
    Helper method to track input token components for a given task type.

    Args:
        task_type: The type of task
        components: Dict mapping component names to token counts
    """
    for component, token_count in components.items():
        prometheus.DERIVER_TOKENS_PROCESSED.labels(
            task_type=task_type.value,
            token_type=prometheus.TokenTypes.INPUT.value,
            component=component.value,
        ).inc(token_count)
