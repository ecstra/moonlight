import httpx
from dataclasses import dataclass

from typing import List, Optional
from .main import Provider

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
        
async def CheckModel(
    provider: Provider,            
    model: str
) -> ModelInfo:
    """
    Validate model existence and retrieve its capabilities from the provider.

    Queries the provider's model registry to determine if the specified model exists
    and extracts key configuration details including context limits, supported modalities,
    and reasoning capabilities.

    Args:
        provider: LLM provider instance with get_source() and get_api() methods
        model (str): Model identifier (e.g., "openai/gpt-4-0314")

    Returns:
        model_info (ModelInfo): ModelInfo class with information.

    Raises:
        CheckModelError: If provider or model parameter is missing
        httpx.HTTPStatusError: If the API request fails

    Example:
        >>> info = await CheckModel(provider, "openai/gpt-4")
        >>> if info["model_exists"]:
        ...     print(f"Context: {info['context_length']} tokens")
    """
    
    # OpenRouter structure
    # {
    #     "id": "openai/gpt-4-0314",
    #     "canonical_slug": "openai/gpt-4-0314",
    #     "hugging_face_id": null,
    #     "name": "OpenAI: GPT-4 (older v0314)",
    #     "created": 1685232000,
    #     "description": "GPT-4-0314 is the first version of GPT-4 released, with a context length of 8,192 tokens, and was supported until June 14. Training data: up to Sep 2021.",
    #     "context_length": 8191,
    #     "architecture": {
    #         "modality": "text->text",
    #         "input_modalities": [
    #             "text"
    #         ],
    #         "output_modalities": [
    #             "text"
    #         ],
    #         "tokenizer": "GPT",
    #         "instruct_type": null
    #     },
    #     "pricing": {
    #         "prompt": "0.00003",
    #         "completion": "0.00006",
    #         "request": "0",
    #         "image": "0",
    #         "web_search": "0",
    #         "internal_reasoning": "0"
    #     },
    #     "top_provider": {
    #         "context_length": 8191,
    #         "max_completion_tokens": 4096,
    #         "is_moderated": true
    #     },
    #     "per_request_limits": null,
    #     "supported_parameters": [
    #         "frequency_penalty",
    #         "logit_bias",
    #         "logprobs",
    #         "max_tokens",
    #         "presence_penalty",
    #         "response_format",
    #         "seed",
    #         "stop",
    #         "structured_outputs",
    #         "temperature",
    #         "tool_choice",
    #         "tools",
    #         "top_logprobs",
    #         "top_p"
    #     ],
    #     "default_parameters": {}
    # }
    
    if not provider:
        raise CheckModelError("LLM Provider must be given")

    if not model:
        raise CheckModelError("Model must be provided")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(
                url=f"{provider.get_source()}/models",
                headers={
                    "Authorization": f"Bearer {provider.get_api()}",
                    "Content-Type": "application/json",
                },
            )
            r.raise_for_status()
            data = r.json()
            print(data)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise CheckModelError("Provider endpoint is incompatible (missing '/models').")
        raise CheckModelError(f"API Error: {e.response.status_code} - {e.response.text}")
    except httpx.RequestError as e:
        raise CheckModelError(f"Request failed: {str(e)}")

    # Build index: id -> model_info
    index = {}
    for m in data["data"]:
        mid = m.get("id")
        if mid:
            index[mid] = m

    if model not in index:
        return ModelInfo.create_empty()

    m = index[model]

    arch = m.get("architecture", {}) or {}

    input_modalities = arch.get("input_modalities", []) or []
    output_modalities = arch.get("output_modalities", []) or []

    # Prefer top_provider limits (actual routed limits)
    top = m.get("top_provider") or {}

    context_length = top.get("context_length") or m.get("context_length")
    max_completion_tokens = top.get("max_completion_tokens")

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
        input_modalities=input_modalities,
        output_modalities=output_modalities
    )