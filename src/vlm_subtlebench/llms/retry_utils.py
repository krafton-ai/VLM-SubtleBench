"""Shared retry logic for LLM API calls."""

import time
from openai.types.chat import ChatCompletion


def chat_completion_with_retry(
    client,
    max_retries: int = 10,
    delay: float = 0.5,
    check_model_extra: bool = False,
    extra_headers: dict = None,
    **kwargs,
) -> ChatCompletion:
    """Send a chat completion request with retry logic.

    Args:
        client: OpenAI-compatible client instance.
        max_retries: Maximum number of retry attempts.
        delay: Delay in seconds between retries.
        check_model_extra: If True, check response.model_extra for errors (used by OpenRouter/vLLM).
        extra_headers: Optional extra HTTP headers to include in the request.
        **kwargs: Arguments passed to client.chat.completions.create().
    """
    for attempt in range(max_retries):
        try:
            create_kwargs = {**kwargs}
            if extra_headers:
                create_kwargs["extra_headers"] = extra_headers
            response = client.chat.completions.create(**create_kwargs)
            if check_model_extra and "error" in response.model_extra:
                if "message" in response.model_extra["error"]:
                    raise Exception(response.model_extra["error"]["message"])
            return response
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[Warning] chat_completion_request failed (attempt {attempt+1}), retrying...")
                time.sleep(delay)
            else:
                print(f"[Error] chat_completion_request failed after {max_retries} attempts.")
                raise e
