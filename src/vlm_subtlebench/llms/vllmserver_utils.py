"""vLLM server API utilities."""

import os
from openai import OpenAI
from openai.types.chat import ChatCompletion

from .retry_utils import chat_completion_with_retry


def setup_vllmserver(key_path: str = "./keys/vllmserver-key/key.env") -> str:
    """Load vLLM server API key from file."""
    with open(key_path, "r") as f:
        api_key = f.read().strip()
    os.environ["OPENROUTER_API_KEY"] = api_key
    return api_key


try:
    client = OpenAI(api_key=setup_vllmserver(), base_url="http://0.0.0.0:8000/v1")
except Exception as e:
    print(f"Exception occurred while setting up OpenRouter client: {e}")
    client = None

_VLLM_HEADERS = {
    "HTTP-Referer": "https://github.com/None/None",
    "X-Title": "None",
}


def chat_completion_request(
    messages,
    model: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    temperature: float = 1.0,
    max_tokens: int = 8192,
    stop=None,
    stream: bool = False,
    **kwargs,
) -> ChatCompletion:
    """Send a chat completion request to vLLM server with retry logic."""
    return chat_completion_with_retry(
        client,
        check_model_extra=True,
        extra_headers=_VLLM_HEADERS,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        stop=stop,
        **kwargs,
    )
