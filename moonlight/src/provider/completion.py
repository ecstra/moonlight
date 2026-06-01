import httpx, json, re, random, asyncio
from dataclasses import dataclass

from typing import Dict, List, Optional, Tuple
from .main import Provider, EndpointType, ANTHROPIC_VERSION

RETRYABLE_STATUS = { 408, 429, 500, 502, 503, 504 }

class GetCompletionError(Exception): pass

@dataclass
class Completion:
    content: Optional[str] = None
    reasoning: Optional[str] = None
    images: Optional[List[str]] = None
    error: Optional[str] = None
    total_tokens: Optional[int] = None

    def model_dump(self) -> dict:
        return {
            "content": json.loads(self.content) if self.content else None,
            "reasoning": self.reasoning,
            "images": self.images,
            "error": self.error,
            "total_tokens": self.total_tokens
        }

    def __str__(self) -> str:
        if self.error:
            return f"CompletionError({self.error})"

        content_preview = (self.content[:50] + "...") if self.content and len(self.content) > 50 else self.content
        return (
            f"{self.content if self.content is not None else ""}\n\n"
            f"---\n\n"
            f"Completion("
            f"content={content_preview!r}, "
            f"images={len(self.images) if self.images else 0}, "
            f"error={self.error!r}, "
            f"total_tokens={self.total_tokens})"
        )

    def __repr__(self) -> str:
        content_preview = (self.content[:50] + "...") if self.content and len(self.content) > 50 else self.content
        return (
            f"Completion("
            f"content={content_preview!r}, "
            f"images={len(self.images) if self.images else 0}, "
            f"error={self.error!r}, "
            f"total_tokens={self.total_tokens})"
        )

def _retry_delay(
    attempt: int,
    backoff: float,
    response: Optional[httpx.Response] = None
) -> float:
    # Honor server's Retry-After (seconds) when present, else exponential backoff w/ full jitter (cap 30s)
    if response is not None:
        retry_after = response.headers.get("retry-after", "")
        if retry_after.isdigit():
            return min(float(retry_after), 30.0)
    return random.uniform(0, min(30.0, backoff * (2 ** attempt)))

def _strip_tags(
    text: str,
    tags: List[str]
) -> str:
    for tag in tags:
        # remove opening tags
        text = re.sub(rf"<{tag}[^>]*?>", "", text, flags=re.IGNORECASE)

        # remove closing tags
        text = re.sub(rf"</{tag}>", "", text, flags=re.IGNORECASE)

    return text

def _check_for_errors(request) -> str:
    status_code = request.status_code
    error_msg = None
    provider_name = None

    try:
        error_response = request.json()

        if "error" in error_response and isinstance(error_response["error"], dict):
            api_error = error_response["error"]

            # Get provider name if available
            if "metadata" in api_error and isinstance(api_error["metadata"], dict):
                provider_name = api_error["metadata"].get("provider_name")

            # Handle metadata first for more specific errors
            if "metadata" in api_error and isinstance(api_error["metadata"], dict):
                metadata = api_error["metadata"]

                # Handle moderation errors (reasons and flagged content)
                if status_code == 403 and "reasons" in metadata:
                    reasons = ", ".join(metadata["reasons"])
                    error_msg = f"Content flagged for: {reasons}"

                # Handle provider specific raw errors
                elif "raw" in metadata:
                    raw_data = metadata["raw"]

                    if isinstance(raw_data, str):
                        # First try to parse as JSON (nested error structures)
                        try:
                            parsed_raw = json.loads(raw_data)

                            if isinstance(parsed_raw, dict):
                                # Extract the main error (e.g. "Invalid request parameters")
                                outer_msg = parsed_raw.get("error")

                                # Try to extract deeper details (e.g. "max_tokens ... is too large")
                                inner_msg = ""
                                if "details" in parsed_raw and isinstance(parsed_raw["details"], str):
                                    try:
                                        details_json = json.loads(parsed_raw["details"])
                                        if "error" in details_json and "message" in details_json["error"]:
                                            inner_msg = details_json["error"]["message"]
                                    except: pass

                                # Combine messages for clarity
                                if outer_msg and inner_msg:
                                    error_msg = f"{outer_msg}: {inner_msg}"
                                elif inner_msg:
                                    error_msg = inner_msg
                                elif outer_msg:
                                    error_msg = str(outer_msg)
                        except json.JSONDecodeError:
                            # Raw is a plain string error message - use it directly
                            error_msg = raw_data

            # Fallback to top-level message if no specific error was extracted.
            # Anthropic errors ({"type": "error", "error": {"message": ...}}) land here.
            if not error_msg and "message" in api_error:
                error_msg = api_error["message"]

    except json.JSONDecodeError:
        pass  # Response body is not JSON
    except Exception:
        pass  # Unexpected structure

    # Fallback to status code based messages if no specific error found
    if not error_msg:
        match status_code:
            case 400: error_msg = "Bad Request (invalid or missing params, CORS)"
            case 401: error_msg = "Invalid credentials (expired OAuth, disabled/invalid API key)"
            case 402: error_msg = "Insufficient credits - add more credits and retry"
            case 403: error_msg = "Input flagged by moderation system"
            case 404: raise GetCompletionError("Provider endpoint is incompatible (completion endpoint not found).")
            case 408: error_msg = "Request timed out"
            case 429: error_msg = "Rate limited - too many requests"
            case 502: error_msg = "Model is down or returned invalid response"
            case 503: error_msg = "No available model provider meets your routing requirements"
            case _: error_msg = "Unknown error"

    # Build final error message with status code and optional provider
    prefix = f"[{status_code}]"
    if provider_name:
        prefix += f" [{provider_name}]"

    return f"{prefix} {error_msg}"

def _build_request(
    provider: Provider,
    model: str,
    messages: List[Dict[str, str]],
    kwargs: dict
) -> Tuple[str, dict, dict]:
    """
    Build the (url, headers, payload) for a completion request.

    OpenAI-compatible providers POST /chat/completions with Bearer auth.
    Anthropic POSTs /messages with x-api-key, lifts the system message(s) out
    into a top-level "system" field, and requires max_tokens. See structures.txt.
    """
    if provider.get_endpoint_type() == EndpointType.ANTHROPIC:
        # Anthropic keeps the system prompt out of the message list.
        system_parts = [
            m.get("content", "")
            for m in messages
            if m.get("role") == "system" and m.get("content")
        ]
        convo = [m for m in messages if m.get("role") != "system"]

        # max_tokens is required; accept the OpenAI spellings or fall back.
        max_tokens = (
            kwargs.pop("max_tokens", None)
            or kwargs.pop("max_completion_tokens", None)
            or kwargs.pop("max_output_tokens", None)
            or 4096
        )

        payload = {
            "model": model,
            "messages": convo,
            "max_tokens": max_tokens,
            **kwargs,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)

        headers = {
            "x-api-key": provider.get_api(),
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }
        return f"{provider.get_source()}/messages", headers, payload

    # OpenAI-compatible
    headers = {
        "Authorization": f"Bearer {provider.get_api()}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": messages, **kwargs}
    return f"{provider.get_source()}/chat/completions", headers, payload

def _parse_openai_completion(response: dict) -> Completion:
    usage = response.get("usage", {})
    total_tokens = usage.get("total_tokens") if usage else None

    if not response.get("choices"):
        return Completion(error="Response contained no choices")

    message = response["choices"][0]["message"]
    content = message.get("content") or ""

    reasoning = _strip_tags(
        text=message.get("reasoning") or "",
        tags=["think", "thought", "reason"]
    )

    images_data = message.get("images", [])
    images = []
    if isinstance(images_data, list):
        images = [
            item["image_url"]["url"]
            for item in images_data
            if isinstance(item, dict) and item.get("type") == "image_url"
        ]

    return Completion(
        total_tokens=total_tokens,
        content=content.strip(),
        reasoning=reasoning,
        images=images if len(images) > 0 else None
    )

def _parse_anthropic_completion(response: dict) -> Completion:
    blocks = response.get("content")
    if not isinstance(blocks, list):
        return Completion(error="Response contained no content")

    # Anthropic returns a list of typed blocks (text / thinking / tool_use ...).
    text = "".join(
        b.get("text", "")
        for b in blocks
        if isinstance(b, dict) and b.get("type") == "text"
    )
    thinking = "".join(
        b.get("thinking", "")
        for b in blocks
        if isinstance(b, dict) and b.get("type") == "thinking"
    )

    # Anthropic reports token counts separately and gives no total. Sum every
    # input variant (fresh + cache read + cache write) with output to match the
    # OpenAI total_tokens semantic the agent tracks; cache fields are 0 when
    # prompt caching is inactive.
    usage = response.get("usage", {}) or {}
    total = (
        (usage.get("input_tokens") or 0)
        + (usage.get("cache_read_input_tokens") or 0)
        + (usage.get("cache_creation_input_tokens") or 0)
        + (usage.get("output_tokens") or 0)
    )

    return Completion(
        total_tokens=total or None,
        content=text.strip(),
        reasoning=_strip_tags(thinking, tags=["think", "thought", "reason"]),
        images=None
    )

def _parse_completion(response: dict, endpoint_type: EndpointType) -> Completion:
    if endpoint_type == EndpointType.ANTHROPIC:
        return _parse_anthropic_completion(response)
    return _parse_openai_completion(response)

async def GetCompletion(
    provider: Provider,
    model: str,
    messages: List[Dict[str, str]],
    max_retries: int = 2,
    retry_backoff: float = 1.0,
    **kwargs
) -> Completion:
    """
    Asynchronously retrieves a chat completion from the specified LLM provider.

    This function validates inputs, sends a request to the provider's API, handles various
    HTTP error codes (including moderation flags and rate limits), and processes the
    response to extract content, reasoning, and usage statistics.

    Supports OpenAI-compatible providers (OpenAI, DeepSeek, Groq, OpenRouter, ...) and
    Anthropic, which uses a different endpoint, auth scheme and request/response shape.
    The raw structures are documented in structures.txt (same directory).

    Args:
        provider (Provider): The LLM provider instance containing API credentials and base URL.
        model (str): The specific model identifier to use for generation.
        messages (List[Dict[str, str]]): A list of message dictionaries representing the conversation history.
        max_retries (int): Number of extra attempts on transient/network errors.
        retry_backoff (float): Base seconds for exponential backoff between retries.
        **kwargs: Optional configuration parameters allowed by the API (e.g., temperature, top_p, tools).

    Returns:
        Completion: A dataclass containing the generated content, stripped reasoning traces,
                    generated images, error messages (if any), and total token usage.

    Raises:
        GetCompletionError: If required arguments are missing, messages are empty, or invalid kwargs are passed.
    """
    if not provider:
        raise GetCompletionError("LLM Provider must be given")

    if not model or model == "":
        raise GetCompletionError("Model must be provided")

    if not isinstance(messages, list):
        raise GetCompletionError("Messages must be a list of objects")

    if not messages or len(messages) == 0:
        raise GetCompletionError("Messages is empty")

    # Allowed kwargs differ per wire format.
    if provider.get_endpoint_type() == EndpointType.ANTHROPIC:
        allowed = {
            "temperature", "top_p", "top_k", "tools", "tool_choice",
            "stop_sequences", "thinking", "metadata",
            "max_tokens", "max_completion_tokens", "max_output_tokens"
        }
    else:
        allowed = {
            "tools", "temperature", "top_p", "top_k", "plugins",
            "tool_choice", "text", "reasoning", "max_output_tokens",
            "frequency_penalty", "presence_penalty", "repetition_penalty",
            "response_format", "verbosity", "modalities", "max_completion_tokens"
        }

    unknown = set(kwargs) - allowed
    if unknown: raise GetCompletionError(f"Unknown properties: {unknown}")

    # The request is identical across retries, so build it once.
    url, headers, payload = _build_request(provider, model, messages, kwargs)
    body = json.dumps(payload)

    async with httpx.AsyncClient(timeout=300.0) as client:
        # Retry Mechanism
        for attempt in range(max_retries + 1):
            try:
                request = await client.post(url=url, headers=headers, data=body)
            except Exception as e:
                if attempt < max_retries:
                    await asyncio.sleep(_retry_delay(attempt, retry_backoff))
                    continue
                return Completion(error=f"Request failed after {attempt + 1} attempts: {e}")

            # Retry one last time in case of transient server errors
            if request.status_code in RETRYABLE_STATUS and attempt < max_retries:
                await asyncio.sleep(_retry_delay(attempt, retry_backoff, request))
                continue

            break

        if request.status_code != 200:
            return Completion(error=_check_for_errors(request))

        try:
            response = request.json()
        except json.JSONDecodeError:
            return Completion(error=f"Failed to decode JSON response from provider. Raw output: {request.text[:200]}")

        return _parse_completion(response, provider.get_endpoint_type())
