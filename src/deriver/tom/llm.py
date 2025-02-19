import os

from dotenv import load_dotenv
from anthropic import Anthropic
from langfuse.decorators import observe, langfuse_context

load_dotenv()

DEF_PROVIDER = "anthropic"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEF_ANTHROPIC_MODEL = "claude-3-5-sonnet-20240620"

anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

@observe(as_type="generation")
def get_anthropic_response(messages: list[dict[str, str]], model: str = DEF_ANTHROPIC_MODEL, system_prompt: str = "") -> str:
    message = anthropic.messages.create(
        model=model,
        max_tokens=1000,
        temperature=0,
        messages=messages,
        system=system_prompt,
    )
    langfuse_context.update_current_observation(
        input=messages, model=model
    )
    return message.content[0].text

def get_response(messages: list[dict[str, str]], provider: str = DEF_PROVIDER, model: str = DEF_ANTHROPIC_MODEL, system_prompt: str = "") -> str:
    if provider == "anthropic":
        return get_anthropic_response(messages, model, system_prompt)
    else:
        raise ValueError(f"Invalid provider: {provider}")
