"""Google Gemini API utilities."""

import json
import logging
import os
import base64
from typing import List, Dict, Union

from google.oauth2 import service_account
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


def setup_gemini(
    service_account_path: str = "keys/google-key/gemini_gcp.json",
    project_id: str | None = None,
    location: str = "us-central1",
) -> genai.Client:
    """Initialize Gemini client with service account credentials.

    project_id is resolved in order: argument, GOOGLE_CLOUD_PROJECT env var,
    then the project_id field in the service account JSON. No default is
    hardcoded to avoid leaking project identifiers.
    """
    project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id and os.path.isfile(service_account_path):
        with open(service_account_path) as f:
            project_id = json.load(f).get("project_id")
    if not project_id:
        raise ValueError(
            "Gemini project_id must be set via GOOGLE_CLOUD_PROJECT env var, "
            "passed as project_id, or present in the service account JSON."
        )
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    credentials = service_account.Credentials.from_service_account_file(
        service_account_path, scopes=scopes
    )
    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
        credentials=credentials,
    )
    return client


try:
    client = setup_gemini()
except Exception as e:
    print(f"Exception occurred while setting up Gemini client: {e}")
    client = None


def chat_completion_request(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    top_p: float = 0.8,
    max_tokens: int = 1024,
    stream: bool = False,
    response_format: str = None,
):
    """Send a chat completion request to Gemini, converting from OpenAI message format."""
    system_prompt = None
    contents = []

    for m in messages:
        if m["role"] == "system" and system_prompt is None:
            system_prompt = m["content"]
        else:
            if isinstance(m["content"], str):
                contents.append(
                    types.Content(
                        role=m["role"], parts=[types.Part.from_text(text=m["content"])]
                    )
                )
            elif isinstance(m["content"], list):
                for part in m["content"]:
                    if part["type"] == "text":
                        contents.append(
                            types.Content(
                                role=m["role"],
                                parts=[types.Part.from_text(text=part["text"])],
                            )
                        )
                    elif part["type"] == "image_url":
                        base64_image = part["image_url"]["url"][
                            len("data:image/png;base64,") :
                        ]
                        image_bytes = base64.b64decode(base64_image)
                        contents.append(
                            types.Content(
                                role=m["role"],
                                parts=[
                                    types.Part.from_bytes(
                                        data=image_bytes, mime_type="image/png"
                                    )
                                ],
                            )
                        )
                    else:
                        raise ValueError("Content must be a string or list of strings.")

    if system_prompt is None:
        system_prompt = ""

    generate_content_config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
        response_modalities=["TEXT"],
        safety_settings=[],
        system_instruction=[types.Part(text=system_prompt)],
    )

    if response_format:
        generate_content_config.response_mime_type = "application/json"
        generate_content_config.response_schema = response_format

    full_text = ""
    response_role = ""

    if stream:
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if hasattr(chunk, "text") and chunk.text:
                full_text += chunk.text
                response_role = "assistant"
    else:
        max_retries = 10
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                )

                found_text = False
                if hasattr(response, "candidates") and response.candidates:
                    for candidate in response.candidates:
                        if candidate.content.parts:
                            for part in candidate.content.parts:
                                if hasattr(part, "text") and part.text:
                                    found_text = True
                                    break
                        if found_text:
                            break
                if found_text:
                    break

                print(f"[Retry {attempt+1}] Empty response. Retrying...")
            except Exception as e:
                import time

                print(f"[Retry {attempt + 1}] Unexpected error: {e}. Retrying...")
                time.sleep(2**attempt)

    return response
