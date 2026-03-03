"""Type aliases for LLM interfaces."""

from typing import Any, Dict, TypeAlias

try:
    from openai.resources.chat.completions import completion_create_params
except ImportError:
    from openai.resources.chat.completions.completions import (
        completion_create_params,
    )

CompletionFunc: TypeAlias = completion_create_params.Function

CompletionFuncCall: TypeAlias = completion_create_params.FunctionCall

Message: TypeAlias = Dict[str, Any]
