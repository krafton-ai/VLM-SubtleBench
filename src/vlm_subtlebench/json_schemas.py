"""Pydantic schemas for structured LLM output."""

from typing import List
from pydantic import BaseModel

SCHEMA_REGISTRY = {}

def register_schema(cls):
    """Register a schema class in the global registry."""
    SCHEMA_REGISTRY[cls.__name__] = cls.model_json_schema()
    return cls

@register_schema
class ImageComparisonSchema(BaseModel):
    differences: List[str]

@register_schema
class DistractorSchema(BaseModel):
    plausible_negatives: List[str]
