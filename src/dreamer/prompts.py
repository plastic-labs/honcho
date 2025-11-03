from src.config import settings
from src.utils.representation import Representation
from src.utils.templates import render_template


def consolidation_prompt(
    representation: Representation,
) -> str:
    """
    Generate the prompt for user representation consolidation.

    Args:
        representation: The user representation to consolidate

    Returns:
        A consolidated user representation
    """
    representation_as_json = representation.model_dump_json(indent=2)

    return render_template(
        settings.DREAM.CONSOLIDATION_TEMPLATE,
        {"representation_as_json": representation_as_json},
    )
