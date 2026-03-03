"""Response parsing utilities for evaluation tasks."""

import re
import string
from typing import Optional


class ResponseParser:
    """Parses model responses for different task types."""

    @staticmethod
    def extract_multiple_choice_answer(response_content: str) -> Optional[str]:
        """Extract the answer letter (A-E) from a multiple choice response.

        Args:
            response_content: Raw response text from the model.

        Returns:
            Single letter answer (A-E) or empty string if not found.
        """
        if not response_content:
            return None

        response = response_content.strip()

        # Try "### Answer X" format first
        ans_match = re.search(r"### Answer\s*([A-E])", response)
        if ans_match:
            return ans_match.group(1).strip()

        # Try lines that are a single letter
        valid_choices = {"A", "B", "C", "D", "E"}
        lines = [line.strip(string.punctuation) for line in response.split("\n")]
        selected = [line for line in lines if line in valid_choices]
        if selected:
            return selected[-1]

        # Try words that are a single letter
        words = [word.strip(string.punctuation) for word in response.split()]
        selected = [word for word in words if word in valid_choices]
        if selected:
            return selected[-1]

        # Try word boundary match
        letter_match = re.search(r"\b[A-E]\b", response)
        if letter_match:
            return letter_match.group()

        return ""

    @staticmethod
    def extract_free_form_answer(response_content: str) -> str:
        """Extract the answer from a free-form response.

        Args:
            response_content: Raw response text from the model.

        Returns:
            The extracted answer text.
        """
        if not response_content:
            return ""
        return response_content.strip()

    @staticmethod
    def extract_structured_answer(
        response_content: str, answer_format: str
    ) -> Optional[str]:
        """Extract answer from a structured response format.

        Args:
            response_content: Raw response text from the model.
            answer_format: Expected format (e.g., "json").

        Returns:
            Extracted answer or None if parsing fails.
        """
        if answer_format == "json":
            pass
        return response_content.strip() if response_content else None
