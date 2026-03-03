"""DeepSeek API utilities."""

import os
from openai import OpenAI
from openai.types.chat import ChatCompletion

from .retry_utils import chat_completion_with_retry


def setup_deepseek(key_path: str = "keys/deepseek-key/key.env") -> str:
    """Load DeepSeek API key from file."""
    with open(key_path, "r") as f:
        api_key = f.read().strip()
    os.environ["DEEPSEEK_API_KEY"] = api_key
    return api_key

try:
    client = OpenAI(
            api_key=setup_deepseek(),
            base_url="https://api.deepseek.com"
        )
except Exception as e:
    print(f"Exception occurred while setting up DeepSeek client: {e}")
    client = None

def chat_completion_request(
    messages,
    model: str = "deepseek-reasoner",
    temperature: float = 1.0,
    max_tokens: int = 8192,
    stop=None,
    stream: bool = False,
    **kwargs
) -> ChatCompletion:
    """Send a chat completion request to DeepSeek with retry logic."""
    return chat_completion_with_retry(
        client,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        stop=stop,
        **kwargs,
    )
