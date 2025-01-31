from .conversational import get_tom_inference_conversational, get_user_representation_conversational
from .single_prompt import get_tom_inference_single_prompt, get_user_representation_single_prompt

async def get_tom_inference(chat_history: str,
                            session_id: str,
                            user_representation: str = "None",
                            method: str = "conversational",
                            **kwargs
                            ) -> str:
    if method == "conversational":
        return await get_tom_inference_conversational(chat_history, session_id, user_representation, **kwargs)
    elif method == "single_prompt":
        return await get_tom_inference_single_prompt(chat_history, session_id, user_representation, **kwargs)
    else:
        raise ValueError(f"Invalid method: {method}")


async def get_user_representation(chat_history: str,
                            session_id: str,
                            user_representation: str = "None", 
                            tom_inference: str = "None",
                            method: str = "conversational",
                            **kwargs
                            ) -> str:
    if method == "conversational":
        return await get_user_representation_conversational(chat_history, session_id, user_representation, tom_inference, **kwargs)
    elif method == "single_prompt":
        return await get_user_representation_single_prompt(chat_history, session_id, user_representation, tom_inference, **kwargs)
    else:
        raise ValueError(f"Invalid method: {method}")