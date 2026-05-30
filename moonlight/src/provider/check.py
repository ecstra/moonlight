import httpx
from dataclasses import dataclass

from typing import List, Optional
from .main import Provider, EndpointType, ANTHROPIC_VERSION

class CheckModelError(Exception): pass

@dataclass
class ModelInfo:
    model_exists: bool
    context_length: Optional[int]
    max_completion_tokens: Optional[int]
    reasoning: bool
    input_modalities: List[str]
    output_modalities: List[str]

    @staticmethod
    def create_empty() -> ModelInfo:
        return ModelInfo(
            model_exists=False,
            context_length=None,
            max_completion_tokens=None,
            reasoning=False,
            input_modalities=[],
            output_modalities=[]
        )

    def __str__(self) -> str:
        if not self.model_exists:
            return "ModelInfo(model does not exist)"

        inputs = ", ".join(self.input_modalities) if self.input_modalities else "none"
        outputs = ", ".join(self.output_modalities) if self.output_modalities else "none"
        return (
            f"Context Length: {self.context_length}\n"
            f"Max Completion Tokens: {self.max_completion_tokens}\n"
            f"Reasoning: {self.reasoning}\n"
            f"Modalities: {inputs} -> {outputs}\n\n"
            f"---\n\n"
            f"{self!r}"
        )

    def __repr__(self) -> str:
        return (
            f"ModelInfo("
            f"model_exists={self.model_exists}, "
            f"context_length={self.context_length}, "
            f"max_completion_tokens={self.max_completion_tokens}, "
            f"reasoning={self.reasoning}, "
            f"input_modalities={self.input_modalities}, "
            f"output_modalities={self.output_modalities})"
        )

def _parse_openai_compatible(m: dict) -> ModelInfo:
    """
    Map an OpenAI-compatible /models entry into ModelInfo.

    Covers OpenAI, DeepSeek, Groq, OpenRouter and any other provider that
    speaks the OpenAI /models shape. Minimal providers (OpenAI, DeepSeek) only
    report an id, so everything except existence stays unknown (None / []).
    """
    arch = m.get("architecture", {}) or {}

    # Prefer OpenRouter's nested top_provider limits; fall back to the flat
    # fields other providers use (e.g. Groq's context_window).
    top = m.get("top_provider") or {}
    context_length = (
        top.get("context_length")
        or m.get("context_length")
        or m.get("context_window")
    )
    max_completion_tokens = (
        top.get("max_completion_tokens")
        or m.get("max_completion_tokens")
    )

    # Heuristic for reasoning capability
    pricing = m.get("pricing", {}) or {}
    supported = m.get("supported_parameters", []) or []
    reasoning = (
        ("reasoning" in supported)
        or (pricing.get("internal_reasoning", "0") != "0")
    )

    return ModelInfo(
        model_exists=True,
        context_length=context_length,
        max_completion_tokens=max_completion_tokens,
        reasoning=reasoning,
        input_modalities=arch.get("input_modalities", []) or [],
        output_modalities=arch.get("output_modalities", []) or []
    )

def _parse_anthropic(m: dict) -> ModelInfo:
    """
    Map an Anthropic /models entry into ModelInfo.

    Anthropic exposes features as a nested tree of {"supported": bool} flags
    rather than flat lists. Token limits are sometimes reported as 0 (unknown)
    and are normalized to None. When the capabilities block is absent, the
    modalities are left empty (unknown) so the agent fails open.
    """
    caps = m.get("capabilities") or {}

    def supported(name: str) -> bool:
        return bool((caps.get(name) or {}).get("supported"))

    if caps:
        input_modalities = ["text"]
        if supported("image_input"):
            input_modalities.append("image")
        if supported("pdf_input"):
            input_modalities.append("file")
        output_modalities = ["text"]  # Claude models are text-out only
    else:
        input_modalities = []
        output_modalities = []

    return ModelInfo(
        model_exists=True,
        context_length=m.get("max_input_tokens") or None,
        max_completion_tokens=m.get("max_tokens") or None,
        reasoning=supported("thinking"),
        input_modalities=input_modalities,
        output_modalities=output_modalities
    )

async def CheckModel(
    provider: Provider,
    model: str
) -> ModelInfo:
    """
    Validate model existence and retrieve its capabilities from the provider.

    Queries the provider's model registry to determine if the specified model exists
    and extracts key configuration details including context limits, supported modalities,
    and reasoning capabilities.

    Supports OpenAI-compatible providers (OpenAI, DeepSeek, Groq, OpenRouter, ...)
    and Anthropic, which uses a different auth scheme and response shape.

    Args:
        provider: LLM provider instance with get_source() and get_api() methods
        model (str): Model identifier (e.g., "openai/gpt-4-0314")

    Returns:
        model_info (ModelInfo): ModelInfo class with information.

    Raises:
        CheckModelError: If provider or model parameter is missing
        httpx.HTTPStatusError: If the API request fails

    Note:
        Each provider's raw /models response shape is documented in
        structures.txt (same directory).

    Example:
        >>> info = await CheckModel(provider, "openai/gpt-4")
        >>> if info.model_exists:
        ...     print(f"Context: {info.context_length} tokens")
    """

    if not provider:
        raise CheckModelError("LLM Provider must be given")

    if not model:
        raise CheckModelError("Model must be provided")

    # Anthropic authenticates differently and must be detected before the call.
    is_anthropic = provider.get_endpoint_type() == EndpointType.ANTHROPIC
    if is_anthropic:
        headers = {
            "x-api-key": provider.get_api(),
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }
    else:
        headers = {
            "Authorization": f"Bearer {provider.get_api()}",
            "Content-Type": "application/json",
        }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(
                url=f"{provider.get_source()}/models",
                headers=headers,
            )
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise CheckModelError("Provider endpoint is incompatible (missing '/models').")
        raise CheckModelError(f"API Error: {e.response.status_code} - {e.response.text}")
    except httpx.RequestError as e:
        raise CheckModelError(f"Request failed: {str(e)}")

    # Both OpenAI-compatible and Anthropic responses wrap models in "data".
    index = {}
    for entry in data["data"]:
        mid = entry.get("id")
        if mid:
            index[mid] = entry

    if model not in index:
        return ModelInfo.create_empty()

    m = index[model]
    return _parse_anthropic(m) if is_anthropic else _parse_openai_compatible(m)
