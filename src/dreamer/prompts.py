from inspect import cleandoc as c

from src.utils.representation import Representation


def consolidation_prompt(
    representation: Representation,
) -> str:
    """
    Generate the prompt for user reprensentation consolidation.

    Args:
        representation: The user reprensentation to consolidate

    Returns:
        A consolidated user reprensentation
    """
    representation_as_json = representation.model_dump_json(indent=2)

    return c(
        f"""
You are an agent that consolidates observations about an entity. You will be presented with a list of EXPLICIT and DEDUCTIVE observations. **Reduce** the number of observations, if possible, by combining similar observations. **ONLY** include information that is **GIVEN**. Create the highest-quality observations with the given information. Observations must always be maximally concise.

{representation_as_json}
"""
    )
