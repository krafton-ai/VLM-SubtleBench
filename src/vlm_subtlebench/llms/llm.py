"""LLM backend factory and model classes."""

import os
import logging
from typing import Any, Dict, Iterable, List, Union

import numpy as np
import tiktoken
from anthropic.types import MessageParam
from openai import OpenAI

from .openai_utils import (
    MoneyManager,
    chat_completion_request,
)
from .anthropic_utils import (
    chat_completion_request as anthropic_chat_completion_request,
)
from .google_utils import (
    chat_completion_request as google_chat_completion_request,
)
from .openrouter_utils import (
    chat_completion_request as openrouter_chat_completion_request,
)
from .vllmserver_utils import (
    chat_completion_request as vllmserver_chat_completion_request,
)

from .utils import CompletionFunc, Message

logger = logging.getLogger(__name__)


class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class Choice:
    def __init__(self, message: Message):
        self.message = message


class GeminiChatCompletionResponse:
    """Wraps Gemini response to match OpenAI ChatCompletion interface."""

    def __init__(self, text: str, role: str):
        self.choices = [Choice(Message(role=role, content=text))]


class ChatGPTBase:
    """OpenAI ChatGPT backend (gpt-4, gpt-5, o1, o3)."""

    def __init__(
        self,
        model: str,
        tool: Iterable[CompletionFunc] | None = None,
        ctx_manager: MoneyManager | None = None,
        desired_output_length: int = 512,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
    ):
        self.model = model
        self.tool = tool
        assert ctx_manager is not None
        self.ctx_manager = ctx_manager
        self.enc = tiktoken.get_encoding("cl100k_base")
        if "gpt-4-1106-preview" in self.model:
            self.max_budget = 128000
        elif "gpt-4" in self.model:
            self.max_budget = 128000
        elif "gpt-5" in self.model:
            self.max_budget = 128000
        elif "o1" in self.model:
            self.max_budget = 128000
        elif "o3" in self.model:
            self.max_budget = 128000
        else:
            raise NotImplementedError()
        self.desired_output_length = desired_output_length
        self.temperature = temperature
        self.repetition_penalty = (repetition_penalty - 1.0,)

        # Reasoning models don't support temperature
        if "o1" in self.model or "o3" in self.model or self.model == "gpt-5":
            self.temperature = None
            logger.info(
                f"Temperature is not supported and set to None for reasoning models (model name: {self.model})."
            )

    def cutoff(self, message: Union[str, dict], budget: int) -> str:
        if isinstance(message, dict):
            # TODO: implement cutoff for visual input
            return message

        tokens = self.enc.encode(message)
        if len(tokens) > budget:
            message = self.enc.decode(tokens[:budget])
        return message

    def manage_length(self, messages: List[Message]) -> None:
        # TODO: implement context length management for multimodal messages
        pass

    def chat(
        self,
        messages: List[Message],
        function: List[CompletionFunc | None] = [None],
        disable_function: bool = False,
        **kwargs,
    ):
        self.manage_length(messages)
        if self.tool is not None and not disable_function:
            response = chat_completion_request(
                messages,
                self.tool.functions,
                model=self.model,
                temperature=self.temperature,
                frequency_penalty=self.repetition_penalty,
                **kwargs,
            )
        else:
            response = chat_completion_request(
                messages,
                model=self.model,
                temperature=self.temperature,
                **kwargs,
            )
        self.ctx_manager(response)
        return response

    def __call__(
        self,
        messages: List[Message],
        disable_function: bool = False,
        stop: List[str] | str | None = None,
        n: int = 1,
        max_tokens: int | None = None,
        **kwargs,
    ):
        response = self.chat(
            messages,
            disable_function=disable_function,
            stop=stop,
            n=n,
            max_tokens=max_tokens,
            **kwargs,
        )

        full_message = response.choices[0]
        if full_message.finish_reason == "function_call":
            messages.append(full_message["message"])
            func_results = self.tool.call_function(messages, full_message)

            try:
                response = self.chat(messages, disable_function=True)
                return {
                    "response": response,
                    "function_results": func_results,
                }
            except Exception as e:
                print(type(e))
                raise Exception("Function chat request failed")
        else:
            return {
                "response": response,
                "function_results": None,
            }


class ClaudeBase:
    """Anthropic Claude backend (via direct API)."""

    def __init__(
        self,
        model: str,
        ctx_manager: MoneyManager = None,
        desired_output_length: int = 512,
        temperature: float = 1.0,
        **kwargs,
    ):
        self.model = model
        assert ctx_manager is not None
        self.ctx_manager = ctx_manager
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = 8192
        self.desired_output_length = desired_output_length
        self.temperature = temperature

    def chat(self, messages: List[MessageParam], *args, **kwargs):
        response = anthropic_chat_completion_request(
            messages, model=self.model, temperature=self.temperature, **kwargs
        )
        self.ctx_manager(response)
        return response

    def __call__(
        self,
        messages: List[MessageParam],
        disable_function: bool = False,
        stop: List[str] | str | None = None,
        **kwargs,
    ):
        response = self.chat(
            messages,
            disable_function=disable_function,
            stop=stop,
            max_tokens=self.max_tokens,
        )
        return {
            "response": response,
            "function_results": None,
        }


class GeminiBase:
    """Google Gemini backend."""

    def __init__(
        self,
        model: str,
        ctx_manager=None,
        desired_output_length: int = 512,
        temperature: float = 1.0,
    ):
        self.model = model
        assert ctx_manager is not None
        self.ctx_manager = ctx_manager
        self.max_tokens = 8192
        self.desired_output_length = desired_output_length
        self.temperature = temperature

    def chat(self, messages: List[dict], *args, **kwargs):
        response = google_chat_completion_request(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )
        self.ctx_manager(response)
        return response

    def __call__(
        self,
        messages: List[dict],
        disable_function: bool = False,
        **kwargs,
    ):
        response = self.chat(
            messages,
            **kwargs,
        )

        full_text = ""
        for candidate in response.candidates:
            if candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        full_text += part.text

        return {
            "response": GeminiChatCompletionResponse(text=full_text, role="assistant"),
            "function_results": None,
        }


class OpenRouterBase:
    """OpenRouter backend (for Claude, Qwen, InternVL, etc.)."""

    def __init__(
        self,
        model: str,
        ctx_manager=None,
        desired_output_length: int = 512,
        temperature: float = 1.0,
    ):
        self.model = model
        assert ctx_manager is not None
        self.ctx_manager = ctx_manager
        self.max_tokens = desired_output_length
        self.temperature = temperature
        self.stream = False

    def chat(self, messages: List[dict], *args, **kwargs):
        response = openrouter_chat_completion_request(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=self.stream,
            **kwargs,
        )
        self.ctx_manager(response)
        return response

    def __call__(
        self,
        messages: List[dict],
        disable_function: bool = False,
        stop: Union[List[str], str, None] = None,
        **kwargs,
    ):
        response = self.chat(
            messages,
            stop=stop,
            **kwargs,
        )
        return {
            "response": response,
            "function_results": None,
        }


class VLLMServerBase:
    """vLLM server backend (for LLaVA, etc.)."""

    def __init__(
        self,
        model: str,
        ctx_manager=None,
        desired_output_length: int = 512,
        temperature: float = 1.0,
    ):
        self.model = model
        assert ctx_manager is not None
        self.ctx_manager = ctx_manager
        self.max_tokens = desired_output_length
        self.temperature = temperature
        self.stream = False

    def chat(self, messages: List[dict], *args, **kwargs):
        response = vllmserver_chat_completion_request(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=self.stream,
            **kwargs,
        )
        self.ctx_manager(response)
        return response

    def __call__(
        self,
        messages: List[dict],
        disable_function: bool = False,
        stop: Union[List[str], str, None] = None,
        **kwargs,
    ):
        response = self.chat(
            messages,
            stop=stop,
            **kwargs,
        )
        return {
            "response": response,
            "function_results": None,
        }


class LocalBase:
    """Local model backend via OpenAI-compatible chat completions API."""

    def __init__(
        self,
        model,
        api_key,
        api_base_url,
        ctx_manager: MoneyManager | None = None,
        desired_output_length: int = 512,
        temperature: float = 1.0,
        **kwargs,
    ):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=api_base_url)
        assert ctx_manager is not None
        self.ctx_manager = ctx_manager
        self.max_tokens = desired_output_length
        self.temperature = temperature

    def chat(self, messages, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )
        self.ctx_manager(response)
        return response

    def __call__(self, messages, **kwargs):
        response = self.chat(messages, **kwargs)
        return {"response": response, "function_results": None}


# (substring_match, model_type, output_budget) — checked in order
# output_budget can be an int or a callable(model_name) -> int
MODEL_ROUTING = [
    ("gpt-4", "chatgpt", 64000),
    ("o1", "chatgpt", 64000),
    ("o3", "chatgpt", 64000),
    ("o4", "chatgpt", 64000),
    ("gpt-5", "chatgpt", 128000),
    ("claude", "openrouter", 64000),
    ("gemini", "gemini", 64000),
    ("llava", "vllmserver", 32000),
    ("qwen", "openrouter", 32000),
    ("internvl", "openrouter", 32000),
    ("local_", "local", 32000),
]

BACKEND_CLASSES = {
    "chatgpt": ChatGPTBase,
    "claude": ClaudeBase,
    "gemini": GeminiBase,
    "openrouter": OpenRouterBase,
    "vllmserver": VLLMServerBase,
    "local": LocalBase,
}


def load_model(
    model: str,
    fine_tuned_model: str | None = None,
    temperature: float = 1.0,
    repetition_penalty: float = 0,
    api_key: str = None,
    api_base_url: str = None,
) -> Dict[str, Any]:
    """Load and configure an LLM backend based on model name.

    Routes by model name substring to the appropriate backend class.

    Returns:
        Dict with 'model_name', 'llm', 'tokenizer', 'ctx_manager'.
    """
    # Find matching backend via routing table
    model_type = "local"
    output_budget = 32000
    for substring, mtype, budget in MODEL_ROUTING:
        if substring in model:
            model_type = mtype
            output_budget = budget(model) if callable(budget) else budget
            break

    if fine_tuned_model is not None:
        model = fine_tuned_model

    logger.info(f"model: {model}")
    logger.info(f"model_type: {model_type}")

    ctx_manager = MoneyManager(model=model)
    enc = tiktoken.get_encoding("cl100k_base")

    # Build kwargs common to all backends
    kwargs = dict(
        model=model,
        ctx_manager=ctx_manager,
        desired_output_length=output_budget,
        temperature=temperature,
    )

    if model_type == "chatgpt":
        kwargs["repetition_penalty"] = repetition_penalty
    elif model_type == "local":
        assert (api_key is not None) and (
            api_base_url is not None
        ), "API key and base URL must be provided for local models."
        kwargs["repetition_penalty"] = repetition_penalty
        kwargs["api_key"] = api_key
        kwargs["api_base_url"] = api_base_url

    backend_cls = BACKEND_CLASSES[model_type]
    llm = backend_cls(**kwargs)

    return {
        "model_name": model,
        "llm": llm,
        "tokenizer": enc,
        "ctx_manager": ctx_manager,
    }
