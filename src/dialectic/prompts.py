from src.config import settings
from src.utils.templates import render_template


def dialectic_prompt(
    query: str,
    working_representation: str,
    recent_conversation_history: str | None,
    observer_peer_card: list[str] | None,
    observed_peer_card: list[str] | None = None,
    *,
    observer: str,
    observed: str,
) -> str:
    """
    Generate the main dialectic prompt for context synthesis.

    Args:
        query: The specific question or request from the application about the user
        working_representation: Conclusions from recent conversation analysis AND historical conclusions from the user's global representation
        recent_conversation_history: Recent conversation history
        observer_peer_card: Known biographical information about the observer
        observed_peer_card: Known biographical information about the target, if applicable
        observer: Name of the observer peer
        observed: Name of the observed peer

    Returns:
        Formatted prompt string for the dialectic model
    """
    return render_template(
        settings.DIALECTIC.DIALECTIC_TEMPLATE,
        {
            "query": query,
            "working_representation": working_representation,
            "recent_conversation_history": recent_conversation_history,
            "observer_peer_card": observer_peer_card,
            "observed_peer_card": observed_peer_card,
            "observer": observer,
            "observed": observed,
        },
    )


def query_generation_prompt(query: str, observed: str) -> str:
    """
    Generate the prompt for semantic query expansion.

    Args:
        query: The original user query
        observed: Name of the target peer

    Returns:
        Formatted prompt string for query generation
    """
    return render_template(
        settings.DIALECTIC.QUERY_GENERATION_TEMPLATE,
        {
            "query": query,
            "observed": observed,
        },
    )
