"""OpenAI API utilities, cost tracking, and logging."""

import json
import os
from datetime import datetime
from pathlib import Path
from traceback import print_stack
from typing import Dict, Iterable, List

from openai import OpenAI, Stream
from openai.types import Completion, Embedding
from openai.types.chat import ChatCompletion
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random,
    wait_random_exponential,
)
from termcolor import colored

from .utils import (
    CompletionFunc,
    CompletionFuncCall,
    Message,
)


def setup_openai(
    key_path: os.PathLike = "keys/openai-key/key.env",
) -> Dict[str, str]:
    """Load OpenAI API key from file."""
    with open(key_path, "r") as f:
        key_list = f.readlines()

    if len(key_list) > 1:
        api_key = key_list[0].strip()
        organization_key = key_list[1].strip()
        os.environ["OPENAI_ORGANIZATION"] = organization_key
        os.environ["OPENAI_API_KEY"] = api_key
        return {"organization": organization_key, "api_key": api_key}
    else:
        api_key = key_list[0].strip()
        os.environ["OPENAI_API_KEY"] = api_key
        return {"api_key": api_key}


try:
    client = OpenAI(**setup_openai())
except Exception as e:
    print(f"Exception occurred while setting up OpenAI client: {e}")
    client = None


@retry(wait=wait_random(min=1, max=10), stop=stop_after_attempt(5))
def chat_completion_request(
    messages: List[Message],
    functions: Iterable[CompletionFunc] | None = None,
    function_call: CompletionFuncCall | None = None,
    model: str = "gpt-4o",
    client: OpenAI = client,
    **kwargs,
) -> Stream[ChatCompletion] | None:
    """Send a chat completion request to OpenAI."""
    json_data = {"model": model, "messages": messages}
    if functions is not None:
        json_data.update({"functions": functions})
    if function_call is not None:
        json_data.update({"function_call": function_call})

    if "stop" in kwargs.keys() and kwargs["stop"] is not None:
        json_data.update({"stop": kwargs["stop"]})
    if "temperature" in kwargs.keys() and kwargs["temperature"] is not None:
        json_data.update({"temperature": kwargs["temperature"]})
    if "n" in kwargs.keys() and kwargs["n"] is not None:
        json_data.update({"n": kwargs["n"]})
    if "max_tokens" in kwargs.keys() and kwargs["max_tokens"] is not None:
        json_data.update({"max_tokens": kwargs["max_tokens"]})
    if "json_mode" in kwargs.keys() and kwargs["json_mode"] is not None:
        if kwargs["json_mode"]:
            json_data.update({"response_format": {"type": "json_object"}})

    if "response_format" in kwargs.keys() and kwargs["response_format"] is not None:
        # Use beta parse() endpoint for structured output
        response_format_model = kwargs["response_format"]
        try:
            response = client.beta.chat.completions.parse(
                response_format=response_format_model,
                **json_data
            )
            return response
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            raise e
    else:
        try:
            response = client.chat.completions.create(**json_data)
            return response
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            raise e


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def completion_request(
    prompt, model: str = "gpt-4o", client: OpenAI = client, **kwargs
) -> ChatCompletion | None:
    """Send a completion request to an OpenAI-compatible API."""
    json_data = {"model": model, "prompt": prompt}
    extra_data = {}

    args_keys = [
        "stop",
        "temperature",
        "n",
        "max_tokens",
        "top_k",
        "top_p",
        "do_sample",
    ]
    extra_keys = [
        "repetition_penalty",
        "guided_regex",
        "guided_json",
        "guided_decoding_backend",
    ]

    for args_key in args_keys:
        if args_key in kwargs.keys() and kwargs[args_key] is not None:
            json_data.update({args_key: kwargs[args_key]})

    for args_key in extra_keys:
        if args_key in kwargs.keys() and kwargs[args_key] is not None:
            extra_data.update({args_key: kwargs[args_key]})

    if len(extra_data) > 0:
        json_data.update({"extra_body": extra_data})

    try:
        response = client.completions.create(**json_data)
        return response
    except Exception as e:
        print("Unable to generate Completion response")
        print(f"Exception: {e}")
        raise e


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def embedding_request(
    text: str, model: str = "text-embedding-3-small"
) -> Embedding | None:
    """Send an embedding request to OpenAI."""
    text = text.replace("\n", " ")

    try:
        return client.embeddings.create(input=[text], model=model).data[0].embedding
    except Exception as e:
        print("Unable to generate Embedding response")
        print(f"Exception: {e}")
        raise e


# (input_cost_per_1k_tokens, output_cost_per_1k_tokens)
MODEL_PRICING = {
    "gpt-4-turbo-preview": (0.01, 0.03),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-4-1106-preview": (0.01, 0.03),
    "gpt-4": (0.03, 0.06),
    "text-embedding-ada-002": (0.0001, 0.0),
    "claude-3-opus-20240229": (0.015, 0.075),
    "claude-opus-4-20250514": (15 / 1000, 75 / 1000),
    "claude-sonnet-4-20250514": (3 / 1000, 15 / 1000),
    "anthropic/claude-sonnet-4": (3 / 1000, 15 / 1000),
    "gpt-4o": (2.5 / 1000, 10 / 1000),
    "gpt-4o-mini": (0.15 / 1000, 0.6 / 1000),
    "gpt-4o-2024-08-06": (2.5 / 1000, 10 / 1000),
    "gpt-4o-2024-05-13": (5 / 1000, 15 / 1000),
    "gpt-5": (1.25 / 1000, 10 / 1000),
    "gpt-5-chat-latest": (1.25 / 1000, 10 / 1000),
    "o1-preview": (15 / 1000, 60 / 1000),
    "o1-preview-2024-09-12": (15 / 1000, 60 / 1000),
    "o1-2024-12-17": (15 / 1000, 60 / 1000),
    "o1-mini": (1.1 / 1000, 4.4 / 1000),
    "o1-mini-2024-09-12": (1.1 / 1000, 4.4 / 1000),
    "o3-mini": (1.1 / 1000, 4.4 / 1000),
    "o3": (2 / 1000, 8 / 1000),
    "o3-mini-2025-01-31": (1.1 / 1000, 4.4 / 1000),
    "gpt-4.1": (2 / 1000, 8 / 1000),
    "gpt-4.1-mini": (0.4 / 1000, 1.6 / 1000),
    "gpt-4.1-2025-04-14": (2 / 1000, 8 / 1000),
    "o4-mini": (1.1 / 1000, 4.4 / 1000),
    "o4-mini-2025-04-16": (1.1 / 1000, 4.4 / 1000),
    "gemini-2.5-flash": (0.3 / 1000, 2.5 / 1000),
    # TODO: cost changes when tokens > 200k
    "gemini-2.5-pro": (1.25 / 1000, 10 / 1000),
    "qwen/qwen-2.5-vl-7b-instruct": (0.2 / 1000, 0.2 / 1000),
    "qwen/qwen2.5-vl-32b-instruct": (0.04 / 1000, 0.14 / 1000),
    "qwen/qwen2.5-vl-72b-instruct": (0.07 / 1000, 0.28 / 1000),
    "opengvlab/internvl3-78b": (0.03 / 1000, 0.13 / 1000),
}


class MoneyManager:
    """Tracks API costs per model based on token usage."""

    def __init__(self, model: str = "gpt-4o"):
        self.total_cost = 0.0
        self.model = model
        pricing = MODEL_PRICING.get(self.model)
        if pricing:
            self.input_cost, self.output_cost = pricing
        else:
            print(
                f"MoneyManager: Model {self.model} not found. If you are using a new model, please add the cost to the MoneyManager class."
            )
            self.input_cost = 0.0
            self.output_cost = 0.0

    def __call__(self, response: Completion | None = None) -> None:
        """Update total cost based on response token usage."""
        if hasattr(response, "usage") and response.usage is None:
            print("No usage in response")
            print(response)
            return

        if self.model == "gemini-2.5-flash" or self.model == "gemini-2.5-pro":
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = (
                response.usage_metadata.candidates_token_count
                + response.usage_metadata.thoughts_token_count
            )
        else:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            # Reasoning models include additional token categories
            if "o1" in self.model or "o3" in self.model or "o4" in self.model:
                output_tokens += (
                    response.usage.completion_tokens_details.accepted_prediction_tokens
                    + response.usage.completion_tokens_details.reasoning_tokens
                    + response.usage.completion_tokens_details.rejected_prediction_tokens
                )

        input_cost = input_tokens / 1000 * self.input_cost
        output_cost = output_tokens / 1000 * self.output_cost

        self.total_cost += input_cost + output_cost

    def refresh(self) -> None:
        """Reset total cost to zero."""
        self.total_cost = 0.0


class Logger:
    """Logs LLM conversations to JSON files."""

    def __init__(
        self,
        log_path: os.PathLike = "logs",
    ):
        self.log_path = log_path
        os.makedirs(self.log_path, exist_ok=True)

    def __call__(self, messages: List[Message], path=None):
        nowtime = (
            datetime.now().time().strftime("%H%M%S")
            + f"_{int(datetime.now().microsecond / 100):04d}"
        )
        if path is not None:
            folder_name = os.path.join(self.log_path, path)
            os.makedirs(folder_name, exist_ok=True)
            filepath = os.path.join(folder_name, nowtime + ".json")
        else:
            filepath = os.path.join(self.log_path, nowtime + ".json")

        with open(filepath, "w+", encoding="utf-8") as f:
            json.dump(
                messages,
                f,
                ensure_ascii=False,
                indent=4,
            )


def pretty_print_conversation(messages: List[Message]):
    """Print a conversation with color-coded roles."""
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }
    formatted_messages = []
    for message in messages:
        if "role" not in message:
            continue

        if message["role"] == "system":
            formatted_messages.append(f"system: {message['content']}\n")
        elif message["role"] == "user":
            formatted_messages.append(f"user: {message['content']}\n")
        elif message["role"] == "assistant" and message.get("function_call"):
            formatted_messages.append(f"assistant: {message['function_call']}\n")
        elif message["role"] == "assistant" and not message.get("function_call"):
            formatted_messages.append(f"assistant: {message['content']}\n")
        elif message["role"] == "function":
            formatted_messages.append(
                f"function ({message['name']}): {message['content']}\n"
            )
    for formatted_message in formatted_messages:
        print(
            colored(
                formatted_message,
                role_to_color[
                    messages[formatted_messages.index(formatted_message)]["role"]
                ],
            )
        )


if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "Say hi."},
        {"role": "user", "content": "Hello."},
    ]
    message = chat_completion_request(
        messages, model="gpt-4-turbo-preview", max_tokens=100, temperature=0.5
    )
    print(message)
