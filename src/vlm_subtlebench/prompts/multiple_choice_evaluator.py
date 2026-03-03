# flake8: noqa
"""Prompts and message builders for multiple-choice evaluation."""

import base64
import io
from typing import List, Dict, Any
from PIL import Image

from vlm_subtlebench.utils import encode_image_to_base64

# --- System prompts ---

_BASE_GUIDELINES = """Guidelines:
- Unless specified in the options, the difference is described in terms of the second image relative to the first."""

MULTIPLE_CHOICE_SYSTEM_PROMPT = f"""You are a helpful assistant that answers multiple-choice questions about differences between two images.
Your task is to carefully analyze both images and identify the main difference between them.

{_BASE_GUIDELINES}
- Respond **only** in the following format. The answer should be a single letter.

### Reasoning
[explanation of the key visual difference between the two images]

### Answer
[answer (single letter)]"""

MULTIPLE_CHOICE_NO_REASONING_SYSTEM_PROMPT = f"""You are a helpful assistant that answers multiple-choice questions about differences between two images.
Your task is to carefully analyze both images and identify the main difference between them.

{_BASE_GUIDELINES}
- Respond **only** with the answer letter (A, B, C, D, etc.). Do not provide any reasoning or explanation."""

MULTIPLE_CHOICE_CAMERA_AUGMENTED_SYSTEM_PROMPT = f"""You are a helpful assistant that answers multiple-choice questions about differences between two images.
Your task is to carefully analyze both images and identify the main difference between them.

{_BASE_GUIDELINES}
- For camera movement analysis, remember that the camera moves in the opposite direction to the perceived motion of objects in the scene. If objects appear to move left, the camera is actually moving right. If the scene appears to rotate clockwise, the camera is rotating counterclockwise.
- Respond **only** with the answer letter (A, B, C, D, etc.). Do not provide any reasoning or explanation."""

MULTIPLE_CHOICE_CONCATENATED_SYSTEM_PROMPT = f"""You are a helpful assistant that answers multiple-choice questions about differences between two images that are concatenated horizontally (first image on the left and second image on the right, separated by a black line).
Your task is to carefully analyze both images and identify the main difference between them.

{_BASE_GUIDELINES}
- Respond **only** with the answer letter (A, B, C, D, etc.). Do not provide any reasoning or explanation."""

MULTIPLE_CHOICE_OVERLAPPED_SYSTEM_PROMPT = f"""You are a helpful assistant that answers multiple-choice questions about differences between two images.
Your task is to carefully analyze first and second images and identify the main difference between them. The third image is the overlay of the first and second images. You may use the third image to help you analyze the difference between the first and second images.

{_BASE_GUIDELINES}
- Respond **only** with the answer letter (A, B, C, D, etc.). Do not provide any reasoning or explanation."""

MULTIPLE_CHOICE_SUBSTRACT_SYSTEM_PROMPT = f"""You are a helpful assistant that answers multiple-choice questions about differences between two images.
Your task is to carefully analyze first and second images and identify the main difference between them. The third image is a black-and-white difference map between the first and second images, where brighter areas indicate larger differences. You may use the third image to help you analyze the difference between the first and second images.

{_BASE_GUIDELINES}
- Respond **only** with the answer letter (A, B, C, D, etc.). Do not provide any reasoning or explanation."""


# --- Helpers ---


def _image_block(base64_str: str) -> Dict[str, Any]:
    """Create an image_url content block from a base64-encoded string."""
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"},
    }


def _build_messages(
    system_prompt: str,
    image_blocks: List[Dict[str, Any]],
    user_prompt: str,
) -> List[Dict[str, Any]]:
    """Build a chat message list from a system prompt, image blocks, and user text."""
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [*image_blocks, {"type": "text", "text": user_prompt}],
        },
    ]


def _encode_image_to_base64_jpeg(image: Image.Image) -> str:
    """Encode a PIL Image to a base64 JPEG string."""
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


# --- User prompts ---


def _get_user_prompt(
    options_text: str,
    question_text: str = None,
    preamble: str = None,
) -> str:
    """Build the user prompt, optionally with a preamble for multi-image setups."""
    if question_text is None:
        question_text = "What is the difference between these two images?"

    parts = []
    if preamble:
        parts.append(preamble)
    parts.append(f"Question: {question_text}")
    parts.append("")
    if preamble:
        parts.append(
            "Carefully examine the images and choose the best description of the key visual difference of first and second images."
        )
    else:
        parts.append(
            "Carefully examine the images and choose the best description of the key visual difference."
        )
    parts.append(f"\nOptions:\n{options_text}")
    return "\n".join(parts)


# --- Message creators ---


def _create_two_image_messages(
    first_image_path: str,
    second_image_path: str,
    options_text: str,
    question_text: str,
    system_prompt: str,
) -> List[Dict[str, Any]]:
    """Create messages with two separate images (standard, no_reasoning, camera_augmented, grid)."""
    first_b64 = encode_image_to_base64(first_image_path)
    second_b64 = encode_image_to_base64(second_image_path)
    user_prompt = _get_user_prompt(options_text, question_text)
    return _build_messages(
        system_prompt,
        [_image_block(first_b64), _image_block(second_b64)],
        user_prompt,
    )


def create_standard_messages(
    first_image_path: str,
    second_image_path: str,
    options_text: str,
    question_text: str,
) -> List[Dict[str, Any]]:
    """Create messages for standard multiple choice evaluation."""
    return _create_two_image_messages(
        first_image_path, second_image_path, options_text, question_text,
        MULTIPLE_CHOICE_SYSTEM_PROMPT,
    )


def create_no_reasoning_messages(
    first_image_path: str,
    second_image_path: str,
    options_text: str,
    question_text: str,
) -> List[Dict[str, Any]]:
    """Create messages for no-reasoning prompt (answer only)."""
    return _create_two_image_messages(
        first_image_path, second_image_path, options_text, question_text,
        MULTIPLE_CHOICE_NO_REASONING_SYSTEM_PROMPT,
    )


def create_camera_augmented_messages(
    first_image_path: str,
    second_image_path: str,
    options_text: str,
    question_text: str,
) -> List[Dict[str, Any]]:
    """Create messages for camera-augmented multiple choice evaluation."""
    return _create_two_image_messages(
        first_image_path, second_image_path, options_text, question_text,
        MULTIPLE_CHOICE_CAMERA_AUGMENTED_SYSTEM_PROMPT,
    )


def create_concatenated_messages(
    first_image_path: str,
    second_image_path: str,
    options_text: str,
    question_text: str,
) -> List[Dict[str, Any]]:
    """Create messages with horizontally concatenated images."""
    first_image = Image.open(first_image_path)
    second_image = Image.open(second_image_path)

    if first_image.mode != "RGB":
        first_image = first_image.convert("RGB")
    if second_image.mode != "RGB":
        second_image = second_image.convert("RGB")

    target_height = min(first_image.height, second_image.height)
    first_width = int(first_image.width * target_height / first_image.height)
    second_width = int(second_image.width * target_height / second_image.height)

    first_resized = first_image.resize((first_width, target_height), Image.LANCZOS)
    second_resized = second_image.resize((second_width, target_height), Image.LANCZOS)

    # 1px black separator between images
    separator_width = 1
    total_width = first_width + separator_width + second_width
    concatenated = Image.new("RGB", (total_width, target_height))
    concatenated.paste(first_resized, (0, 0))
    concatenated.paste(second_resized, (first_width + separator_width, 0))

    concatenated_b64 = _encode_image_to_base64_jpeg(concatenated)

    first_image.close()
    second_image.close()
    first_resized.close()
    second_resized.close()
    concatenated.close()

    user_prompt = _get_user_prompt(options_text, question_text)
    return _build_messages(
        MULTIPLE_CHOICE_CONCATENATED_SYSTEM_PROMPT,
        [_image_block(concatenated_b64)],
        user_prompt,
    )


def create_grid_messages(
    first_image_path: str,
    second_image_path: str,
    options_text: str,
    question_text: str,
) -> List[Dict[str, Any]]:
    """Create messages with 4x4 grid overlay on both images."""
    from PIL import ImageDraw

    def add_grid_to_image(image_path: str) -> str:
        """Add 4x4 grid overlay to an image and return base64."""
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        image_with_grid = image.copy()
        draw = ImageDraw.Draw(image_with_grid)
        width, height = image.size

        # 30% opacity gray lines for the 4x4 grid (3 horizontal + 3 vertical)
        grid_color = (76, 76, 76)
        for i in range(1, 4):
            y = height * i // 4
            draw.line([(0, y), (width, y)], fill=grid_color, width=1)
        for i in range(1, 4):
            x = width * i // 4
            draw.line([(x, 0), (x, height)], fill=grid_color, width=1)

        grid_b64 = _encode_image_to_base64_jpeg(image_with_grid)
        image.close()
        image_with_grid.close()
        return grid_b64

    first_b64 = add_grid_to_image(first_image_path)
    second_b64 = add_grid_to_image(second_image_path)
    user_prompt = _get_user_prompt(options_text, question_text)
    return _build_messages(
        MULTIPLE_CHOICE_SYSTEM_PROMPT,
        [_image_block(first_b64), _image_block(second_b64)],
        user_prompt,
    )


def create_overlapped_messages(
    first_image_path: str,
    second_image_path: str,
    options_text: str,
    question_text: str,
) -> List[Dict[str, Any]]:
    """Create messages with first, second, and 50/50 blended images."""
    image1 = Image.open(first_image_path).convert("RGBA")
    image2 = Image.open(second_image_path).convert("RGBA")

    target_size = (min(image1.width, image2.width), min(image1.height, image2.height))
    image1_resized = image1.resize(target_size, Image.LANCZOS)
    image2_resized = image2.resize(target_size, Image.LANCZOS)
    blended = Image.blend(image1_resized, image2_resized, 0.5).convert("RGB")

    first_b64 = encode_image_to_base64(first_image_path)
    second_b64 = encode_image_to_base64(second_image_path)
    overlapped_b64 = _encode_image_to_base64_jpeg(blended)

    image1.close()
    image2.close()
    image1_resized.close()
    image2_resized.close()
    blended.close()

    preamble = (
        "I am showing you three images:\n"
        "1. First image\n"
        "2. Second image\n"
        "3. Overlapped image (50/50 blend of first and second images)"
    )
    user_prompt = _get_user_prompt(options_text, question_text, preamble=preamble)
    return _build_messages(
        MULTIPLE_CHOICE_OVERLAPPED_SYSTEM_PROMPT,
        [_image_block(first_b64), _image_block(second_b64), _image_block(overlapped_b64)],
        user_prompt,
    )


def create_substract_messages(
    first_image_path: str,
    second_image_path: str,
    options_text: str,
    question_text: str,
) -> List[Dict[str, Any]]:
    """Create messages with first, second, and black-and-white difference map images."""
    import numpy as np

    image1 = Image.open(first_image_path).convert("RGB")
    image2 = Image.open(second_image_path).convert("RGB")

    target_size = (min(image1.width, image2.width), min(image1.height, image2.height))
    image1_resized = image1.resize(target_size, Image.LANCZOS)
    image2_resized = image2.resize(target_size, Image.LANCZOS)

    array1 = np.array(image1_resized, dtype=np.float32)
    array2 = np.array(image2_resized, dtype=np.float32)
    diff = np.abs(array1 - array2)
    diff_gray = np.mean(diff, axis=2)
    diff_normalized = (diff_gray / diff_gray.max() * 255).astype(np.uint8)
    diff_image = Image.fromarray(diff_normalized, mode="L").convert("RGB")

    first_b64 = encode_image_to_base64(first_image_path)
    second_b64 = encode_image_to_base64(second_image_path)
    diff_b64 = _encode_image_to_base64_jpeg(diff_image)

    image1.close()
    image2.close()
    image1_resized.close()
    image2_resized.close()
    diff_image.close()

    preamble = (
        "I am showing you three images:\n"
        "1. First image\n"
        "2. Second image\n"
        "3. Black-and-white difference map between the first and second images"
    )
    user_prompt = _get_user_prompt(options_text, question_text, preamble=preamble)
    return _build_messages(
        MULTIPLE_CHOICE_SUBSTRACT_SYSTEM_PROMPT,
        [_image_block(first_b64), _image_block(second_b64), _image_block(diff_b64)],
        user_prompt,
    )


PROMPT_TYPE_HANDLERS = {
    "standard": create_standard_messages,
    "camera_augmented": create_camera_augmented_messages,
    "concatenated": create_concatenated_messages,
    "grid": create_grid_messages,
    "overlapped": create_overlapped_messages,
    "no_reasoning": create_no_reasoning_messages,
    "substract": create_substract_messages,
}


def create_multiple_choice_messages(
    first_image_path: str,
    second_image_path: str,
    options_text: str,
    question_text: str,
    prompt_type: str = "standard",
) -> List[Dict[str, Any]]:
    """Create messages for a multiple choice question with images.

    Args:
        first_image_path: Path to the first image.
        second_image_path: Path to the second image.
        options_text: Formatted options text (e.g., "A. option1\\nB. option2").
        question_text: The question to ask.
        prompt_type: Prompt variant to use.

    Returns:
        List of message dicts for the LLM API.

    Raises:
        ValueError: If prompt_type is not supported.
    """
    if prompt_type not in PROMPT_TYPE_HANDLERS:
        available_types = ", ".join(PROMPT_TYPE_HANDLERS.keys())
        raise ValueError(
            f"Unknown prompt_type: '{prompt_type}'. Available types: {available_types}"
        )

    handler = PROMPT_TYPE_HANDLERS[prompt_type]
    return handler(first_image_path, second_image_path, options_text, question_text)
