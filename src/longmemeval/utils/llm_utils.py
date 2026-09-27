from __future__ import annotations
"""Model client configuration."""
import json
import os
import re
from typing import Any, Dict, Optional

from openai import OpenAI

try:
    from json_repair import repair_json
except ImportError:
    repair_json = None

DEFAULT_DISABLE_THINKING_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}




# No key is bundled. Set T2MEM_LLM_API_KEY (or pass --api-key) when the endpoint
# requires one; servers without auth accept the "EMPTY" placeholder.
DEFAULT_API_KEY = os.getenv("T2MEM_LLM_API_KEY", "")


def build_client(
    base_url: str,
    timeout: float | None = None,
    api_key: str | None = None,
) -> OpenAI:
    """Create a client for a local or remote OpenAI-compatible endpoint.

    Falls back to DEFAULT_API_KEY, then to the "EMPTY" placeholder used by
    servers that do not check the key.
    """
    key = api_key or DEFAULT_API_KEY or "EMPTY"
    if timeout is None:
        return OpenAI(base_url=base_url, api_key=key)
    return OpenAI(base_url=base_url, api_key=key, timeout=timeout)

def call_llm(
    *,
    client: OpenAI,
    model_name: str,
    system_prompt: str | None = None,
    user_prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    extra_body: Optional[Dict[str, Any]] = None,
) -> str:
    """Send one chat turn and return the plain text content."""
    messages = [{"role": "user", "content": user_prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    return response.choices[0].message.content


def call_json_llm(
    *,
    client: OpenAI,
    model_name: str,
    system_prompt: str | None = None,
    user_prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Call the model and parse the first structured object out of its reply."""

    return parse_json_response(
        call_llm(
            client=client,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra_body,
        )
    )

# Normalization helpers
def strip_code_fences(text: str) -> str:
    """Strip the code-fence wrapper models commonly add."""

    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_first_json_object(text: str) -> str:
    """Extract the first bracket-balanced structured object from free text."""

    cleaned = strip_code_fences(text)
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("No JSON object start found.")
    depth = 0
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : index + 1]
    raise ValueError("No complete JSON object found.")


def parse_json_response(text: str) -> Dict[str, Any]:
    """Parse a structured object out of raw model output."""
    cleaned = strip_code_fences(text)
    try:
        return json.loads(extract_first_json_object(cleaned))
    except (ValueError, json.JSONDecodeError):
        if repair_json is None:
            raise ImportError("json_repair is required to repair malformed JSON responses.")
        repaired = repair_json(cleaned, return_objects=True)
        if not isinstance(repaired, dict):
            raise TypeError(f"Repaired JSON is not an object: {type(repaired).__name__}")
        return repaired
