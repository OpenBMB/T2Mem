from __future__ import annotations
"""Model client configuration."""
import json
import os
import re
import threading
from typing import Any, Dict, Optional
from openai import OpenAI
DEFAULT_DISABLE_THINKING_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}

# --- Force an all-ASCII wire body (ensure_ascii=True) ---
# The openai SDK hardcodes ensure_ascii=False in openapi_dumps (openai/_utils/_json.py),
# so non-ASCII text is sent as raw UTF-8 bytes (1 char = 3 bytes). If any proxy truncates
# or validates Content-Length, the character count and byte count disagree and the server
# 400 "Unterminated string"。
# raises an error. We force ensure_ascii=True at the serialization layer: every non-ASCII
# character becomes \uXXXX, so bytes == characters and the mismatch cannot occur. The
# server decodes \uXXXX back, leaving the model-visible text unchanged.
_patch_lock = threading.Lock()
_patch_done = False


def force_ascii_json_body() -> None:
    """Force ensure_ascii=True for the openai SDK JSON body (runs once per process)."""
    global _patch_done
    if _patch_done:
        return
    with _patch_lock:
        if _patch_done:
            return
        import openai._utils._json as _oj
        import openai._base_client as _bc

        def _openapi_dumps_ascii(obj: Any) -> bytes:
            return json.dumps(
                obj,
                cls=_oj._CustomEncoder,
                ensure_ascii=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")

        # _base_client imports openapi_dumps by value, so both bindings must be replaced.
        _oj.openapi_dumps = _openapi_dumps_ascii
        _bc.openapi_dumps = _openapi_dumps_ascii
        _patch_done = True

_lock = threading.Lock()
_prompt_tokens = 0
_completion_tokens = 0


def reset_token_counters() -> None:
    global _prompt_tokens, _completion_tokens
    with _lock:
        _prompt_tokens = 0
        _completion_tokens = 0


def get_token_usage() -> dict[str, int]:
    with _lock:
        return {"prompt_tokens": _prompt_tokens, "completion_tokens": _completion_tokens}


def _record_usage(usage: Any) -> None:
    global _prompt_tokens, _completion_tokens
    if usage is None:
        return
    pt = getattr(usage, "prompt_tokens", 0) or 0
    ct = getattr(usage, "completion_tokens", 0) or 0
    with _lock:
        _prompt_tokens += pt
        _completion_tokens += ct


def build_client(base_url: str, timeout: float | None = None,
                 api_key: str | None = None) -> OpenAI:
    """Create a client for a local or remote OpenAI-compatible endpoint."""
    force_ascii_json_body()
    key = api_key or os.getenv("T2MEM_LLM_API_KEY", "") or "EMPTY"
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
    _record_usage(response.usage)
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
    return json.loads(extract_first_json_object(text))


def _close_truncated_braces(s: str) -> str:
    """Complete truncated JSON with a stack: unclosed { / [ get } / ] appended in reverse.

    ``s`` must start at the first {. String contents are preserved; only nesting is fixed.
    """
    stack: list[str] = []
    in_string = False
    escaped = False
    for ch in s:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack and (
                (ch == "}" and stack[-1] == "{") or (ch == "]" and stack[-1] == "[")
            ):
                stack.pop()
    closing = "".join("}" if c == "{" else "]" for c in reversed(stack))
    return s + closing


def _cut_unterminated_tail(s: str) -> str:
    """If ``s`` ends inside an unterminated string, cut back to the last complete quote.

    Truncation usually lands mid-string in snippet/reasoning, where appending brackets
    alone still fails to parse.
    """
    in_string = False
    escaped = False
    last_complete_quote = -1
    for index, ch in enumerate(s):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
                last_complete_quote = index + 1
            continue
        if ch == '"':
            in_string = True
    if in_string and last_complete_quote >= 0:
        return s[:last_complete_quote]
    return s


def parse_json_response_lenient(text: str) -> Dict[str, Any]:
    """Truncation-tolerant JSON parsing.

    When the model hits max_tokens its output is cut mid-JSON. Try strict parsing first,
    then repair in escalating steps: close brackets -> trim the unterminated string tail ->
    close brackets again.
    """
    try:
        return parse_json_response(text)
    except Exception:
        cleaned = strip_code_fences(text)
        start = cleaned.find("{")
        if start == -1:
            raise ValueError("No JSON object start found.")
        body = cleaned[start:]
        for candidate in (
            _close_truncated_braces(body),
            _close_truncated_braces(_cut_unterminated_tail(body)),
            _close_truncated_braces(
                re.sub(r',?\s*"[^"]*":?\s*$', "", _cut_unterminated_tail(body))
            ),
        ):
            try:
                return json.loads(candidate)
            except Exception:
                continue
    raise ValueError(f"JSON parse failed even after truncation repair: {text[:120]!r}")
