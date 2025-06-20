from .conversational import (
    get_user_representation_conversational,
)
from .long_term import get_user_representation_long_term
from .single_prompt import (
    get_tom_inference_single_prompt,
    get_user_representation_single_prompt,
)


async def get_user_representation(
    chat_history: str,
    user_representation: str = "None",
    tom_inference: str = "None",
    method: str = "conversational",
    **kwargs,
) -> str:
    if method == "conversational":
        return await get_user_representation_conversational(
            chat_history, user_representation, tom_inference, **kwargs
        )
    elif method == "single_prompt":
        return await get_user_representation_single_prompt(
            chat_history, user_representation, tom_inference, **kwargs
        )
    elif method == "long_term":
        return await get_user_representation_long_term(
            chat_history, user_representation, tom_inference, **kwargs
        )
    else:
        raise ValueError(f"Invalid method: {method}")
