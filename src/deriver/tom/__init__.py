from .conversational import (
    get_tom_inference_conversational,
    get_user_representation_conversational,
)
from .long_term import get_user_representation_long_term
from .single_prompt import (
    get_tom_inference_single_prompt,
    get_user_representation_single_prompt,
)


async def get_tom_inference(
    chat_history: str,
    user_representation: str = "None",
    method: str = "conversational",
) -> str:
    if method == "conversational":
        return await get_tom_inference_conversational(chat_history, user_representation)
    elif method == "single_prompt":
        response = await get_tom_inference_single_prompt(
            chat_history, user_representation
        )
        return response.model_dump_json()
    else:
        raise ValueError(f"Invalid method: {method}")


async def get_user_representation(
    chat_history: str,
    user_representation: str = "None",
    tom_inference: str = "None",
    method: str = "conversational",
) -> str:
    if method == "conversational":
        return await get_user_representation_conversational(
            chat_history, user_representation, tom_inference
        )
    elif method == "single_prompt":
        return await get_user_representation_single_prompt(
            chat_history, user_representation, tom_inference
        )
    elif method == "long_term":
        response = await get_user_representation_long_term(
            chat_history, user_representation, tom_inference
        )
        return response.model_dump_json()
    else:
        raise ValueError(f"Invalid method: {method}")
