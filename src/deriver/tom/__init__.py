from .conversational import (
    tom_inference_conversational,
    user_representation_conversational,
)
from .long_term import get_user_representation_long_term
from .single_prompt import (
    TomInferenceOutput,
    UserRepresentationOutput,
)
from .single_prompt import (
    tom_inference as tom_inference_single_prompt,
)
from .single_prompt import (
    user_representation as user_representation_single_prompt,
)


async def get_tom_inference(
    chat_history: str,
    user_representation: str = "None",
    method: str = "conversational",
) -> TomInferenceOutput:
    if method == "conversational":
        return await tom_inference_conversational(chat_history, user_representation)
    elif method == "single_prompt":
        return await tom_inference_single_prompt(chat_history, user_representation)

    else:
        raise ValueError(f"Invalid method: {method}")


async def get_user_representation(
    chat_history: str,
    user_representation: str = "None",
    tom_inference: str = "None",
    method: str = "conversational",
) -> UserRepresentationOutput:
    if method == "conversational":
        return await user_representation_conversational(
            chat_history, user_representation, tom_inference
        )
    elif method == "single_prompt":
        return await user_representation_single_prompt(
            chat_history, user_representation, tom_inference
        )
    elif method == "long_term":
        return await get_user_representation_long_term(
            chat_history, user_representation, tom_inference
        )
    else:
        raise ValueError(f"Invalid method: {method}")
