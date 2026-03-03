# flake8: noqa
"""Prompts for free-form image comparison evaluation."""


def get_image_comparison_system_prompt() -> str:
    """Get the standard system prompt for image comparison."""
    return """You are an expert at visual analysis and image comparison. Your task is to compare two images and describe the differences between them.

Describe what you observe that's different between the two images. This might include differences in attributes, states, emotions, temporal aspects, spatial aspects, existence of objects, quantities, image quality, viewpoints, actions, or interactions. Write your response as one or several sentences describing what you notice."""


def get_image_comparison_system_prompt_simple() -> str:
    """Get a concise system prompt for brief image comparisons."""
    return """You are an expert at visual analysis and image comparison. Compare the two images and briefly describe what's different between them.

Keep your response concise and direct. Use simple phrases like "In the first image, X, however in the second image, Y" or "X appeared in the second image" or "In the first image, X is relatively Y". Avoid detailed explanations or structured lists."""


def get_image_comparison_system_prompt_line_by_line() -> str:
    """Get the structured system prompt that returns differences as a JSON list."""
    return """You are an expert at visual analysis and image comparison. Your task is to compare two images and identify all the differences between them. Return the differences as a list of strings in the specified format.

Adhere to the following guidelines:
- Analyze two images to identify the differences.
- Ensure each string clearly states a specific difference with a focus on the second image as compared to the first.
- The description should be detailed.

# Steps
1. **Analyze the Differences**: Review the images and identify the differences.
2. **Formulate Descriptions**: Write clear and concise descriptions for each identified difference, focusing on changes in the second image compared to the first.
3. **Output in Required Format**: Return the descriptions in the specified structured format.

# Output Format
Return a dictionary with a "differences" key containing a list of strings, where each string describes a specific difference between the images, emphasizing adjustments in the second image.

# Examples

### Example 1
**Output**:
{
    "differences": [
        "The person with the umbrella has appeared in the first image.",
        "The person in the red coat has been moved to the right in the second image.",
        "In the first image, the person in blue shirt remains stationary, whereas in the second image, he is running."
    ]
}

### Example 2
**Output**:
{
    "differences": [
        "The cat on the couch in Image 1 is replaced by a dog in Image 2."
    ]
}"""


def get_image_comparison_user_prompt() -> str:
    """Get the standard user prompt for image comparison."""
    return """Describe the differences between the two images"""


def get_image_comparison_user_prompt_line_by_line() -> str:
    """Get the line-by-line user prompt for structured comparison."""
    return """What is the difference between these two images?"""


SYSTEM_PROMPT_VARIANTS = {
    "standard": get_image_comparison_system_prompt,
    "simple": get_image_comparison_system_prompt_simple,
    "line_by_line": get_image_comparison_system_prompt_line_by_line,
}

USER_PROMPT_VARIANTS = {
    "standard": get_image_comparison_user_prompt,
    "simple": get_image_comparison_system_prompt,
}


def get_system_prompt_by_type(prompt_type: str = "standard") -> str:
    """Get a system prompt variant by type.

    Args:
        prompt_type: One of "standard", "simple", "line_by_line".

    Returns:
        The system prompt text.
    """
    if prompt_type in SYSTEM_PROMPT_VARIANTS:
        return SYSTEM_PROMPT_VARIANTS[prompt_type]()
    else:
        print(
            f"Warning: Unknown prompt type '{prompt_type}'. Using standard system prompt."
        )
        return SYSTEM_PROMPT_VARIANTS["standard"]()


def get_user_prompt_by_type(prompt_type: str = "standard") -> str:
    """Get a user prompt variant by type.

    Args:
        prompt_type: One of "standard", "simple".

    Returns:
        The user prompt text.
    """
    if prompt_type in USER_PROMPT_VARIANTS:
        return USER_PROMPT_VARIANTS[prompt_type]()
    else:
        print(
            f"Warning: Unknown prompt type '{prompt_type}'. Using standard user prompt."
        )
        return USER_PROMPT_VARIANTS["standard"]()
