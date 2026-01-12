import httpx, json, re
from dataclasses import dataclass

from typing import Dict, List, Optional, Any
from .main import Provider

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

async def CheckModel(provider, model: str) -> Dict[str, Any]:
    """
    Validate model existence and retrieve its capabilities from the provider.

    Queries the provider's model registry to determine if the specified model exists
    and extracts key configuration details including context limits, supported modalities,
    and reasoning capabilities.

    Args:
        provider: LLM provider instance with get_source() and get_api() methods
        model (str): Model identifier (e.g., "openai/gpt-4-0314")

    Returns:
        - Dict[str, Any]: Model information dictionary containing:
            - model_exists (bool): Whether the model is available
            - context_length (int | None): Maximum input context tokens
            - max_completion_tokens (int | None): Maximum output tokens
            - reasoning (bool): Whether model supports reasoning/chain-of-thought
            - input_modalities (List[str]): Supported input types (e.g., ["text", "image"])
            - output_modalities (List[str]): Supported output types (e.g., ["text"])

    Raises:
        GetCompletionError: If provider or model parameter is missing
        httpx.HTTPStatusError: If the API request fails

    Example:
        >>> info = await CheckModel(provider, "openai/gpt-4")
        >>> if info["model_exists"]:
        ...     print(f"Context: {info['context_length']} tokens")
    """
    
    # Example structure
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
        raise GetCompletionError("LLM Provider must be given")

    if not model:
        raise GetCompletionError("Model must be provided")

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
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise GetCompletionError("Provider endpoint is incompatible (missing '/models').")
        raise GetCompletionError(f"API Error: {e.response.status_code} - {e.response.text}")
    except httpx.RequestError as e:
        raise GetCompletionError(f"Request failed: {str(e)}")

    # Build index: id -> model_info
    index = {}
    for m in data["data"]:
        mid = m.get("id")
        if mid:
            index[mid] = m

    if model not in index:
        return {
            "model_exists": False,
            "context_length": None,
            "max_completion_tokens": None,
            "reasoning": False,
            "input_modalities": [],
            "output_modalities": [],
        }

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

    return {
        "model_exists": True,
        "context_length": context_length,
        "max_completion_tokens": max_completion_tokens,
        "reasoning": reasoning,
        "input_modalities": input_modalities,
        "output_modalities": output_modalities,
    }

def _strip_tags(text: str, tags: List[str]) -> str:
    for tag in tags:
        # remove opening tags
        text = re.sub(rf"<{tag}[^>]*?>", "", text, flags=re.IGNORECASE)

        # remove closing tags
        text = re.sub(rf"</{tag}>", "", text, flags=re.IGNORECASE)

    return text

def _check_for_errors(request) -> str:
    error_msg = f"Unknown Error ({request.status_code})"
    
    try:
        error_response = request.json()
        if "error" in error_response and isinstance(error_response["error"], dict):
            api_error = error_response["error"]
            if "message" in api_error:
                error_msg = api_error["message"]
            
            if "metadata" in api_error:
                metadata = api_error["metadata"]
                
                # Handle moderation errors (reasons and flagged content)
                if request.status_code == 403 and "reasons" in metadata:
                    reasons = ", ".join(metadata["reasons"])
                    error_msg = f"Content flagged for: {reasons}"
                
                # Handle provider specifc raw errors (often nested JSON strings)
                elif "raw" in metadata:
                    try:
                        raw_data = metadata["raw"]
                        if isinstance(raw_data, str):
                            # Parse the raw error string (e.g. '{"details": "...", "error": "..."}')
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
                    except Exception:
                        pass # Keep the original or basic error message if parsing fails

    except Exception:
        pass # Failed to parse JSON or unexpected structure


    if error_msg.startswith("Unknown Error"):
        match request.status_code:
            case 400: error_msg = "Bad Request (invalid or missing params, CORS)"
            case 401: error_msg = "Invalid credentials (expired OAuth, disabled/invalid API key)"
            case 402: error_msg = "Insufficient credits - add more credits and retry"
            case 403: error_msg = "Input flagged by moderation system"
            case 404: raise GetCompletionError("Provider endpoint is incompatible (missing '/chat/completions').")
            case 408: error_msg = "Request timed out"
            case 429: error_msg = "Rate limited - too many requests"
            case 502: error_msg = "Model is down or returned invalid response"
            case 503: error_msg = "No available model provider meets your routing requirements"
    
    return error_msg

async def GetCompletion(
    provider: Provider,
    model: str,
    messages: List[Dict[str, str]],
    **kwargs
) -> Completion:
    """
    Asynchronously retrieves a chat completion from the specified LLM provider.

    This function validates inputs, sends a request to the provider's API, handles various
    HTTP error codes (including moderation flags and rate limits), and processes the
    response to extract content, reasoning, and usage statistics.

    Args:
        provider (Provider): The LLM provider instance containing API credentials and base URL.
        model (str): The specific model identifier to use for generation.
        messages (List[Dict[str, str]]): A list of message dictionaries representing the conversation history.
        **kwargs: Optional configuration parameters allowed by the API (e.g., temperature, top_p, tools).

    Returns:
        Completion: A dataclass containing the generated content, stripped reasoning traces,
                    generated images, error messages (if any), and total token usage.

    Raises:
        GetCompletionError: If required arguments are missing, messages are empty, or invalid kwargs are passed.
    """
    
    # Example Structure
    # {
    #     "id": "gen-123456789-a1b2c3D4z5C7S4",
    #     "provider": "Seed",
    #     "model": "bytedance-seed/seedream-4.5",
    #     "object": "chat.completion",
    #     "created": 1768040219,
    #     "choices": [
    #         {
    #             "logprobs": null,
    #             "finish_reason": "stop",
    #             "native_finish_reason": null,
    #             "index": 0,
    #             "message": {
    #                 "role": "assistant",
    #                 "content": "",
    #                 "refusal": null,
    #                 "reasoning": null,
    #                 "images": [
    #                     {
    #                         "index": 0,
    #                         "type": "image_url",
    #                         "image_url": {
    #                             "url": "data:image/jpeg;base64,..."
    #                         }
    #                     }
    #                 ]
    #             }
    #         }
    #     ],
    #     "usage": {
    #         "prompt_tokens": 23,
    #         "completion_tokens": 4175,
    #         "total_tokens": 4198,
    #         "cost": 0.040000675,
    #         "is_byok": false,
    #         "prompt_tokens_details": {
    #             "cached_tokens": 0,
    #             "audio_tokens": 0,
    #             "video_tokens": 0
    #         },
    #         "cost_details": {
    #             "upstream_inference_cost": null,
    #             "upstream_inference_prompt_cost": 0,
    #             "upstream_inference_completions_cost": 0.040000675
    #         },
    #         "completion_tokens_details": {
    #             "reasoning_tokens": 0,
    #             "image_tokens": 4175
    #         }
    #     }
    # }
    if not provider:
        raise GetCompletionError("LLM Provider must be given")
    
    if not model or model == "":
        raise GetCompletionError("Model must be provided")
    
    if not isinstance(messages, list):
        raise GetCompletionError("Messages must be a list of objects")
    
    if not messages or len(messages) == 0:
        raise GetCompletionError("Messages is empty")
    
    allowed = { 
        "tools", "temperature", "top_p", "top_k", "plugins",
        "tool_choice", "text", "reasoning", "max_output_tokens",
        "frequency_penalty", "presence_penalty", "repetition_penalty",
        "response_format", "verbosity", "modalities", "max_completion_tokens"
    }
    
    unknown = set(kwargs) - allowed
    
    if unknown: raise GetCompletionError(f"Unknown properties: {unknown}")
          
    async with httpx.AsyncClient(timeout=300.0) as client:
        request = await client.post(
            url=f"{provider.get_source()}/chat/completions",
            headers={
                "Authorization": f"Bearer {provider.get_api()}",
                "Content-Type": "application/json"
            },
            data=json.dumps({
                "model": model,
                "messages": messages,
                **kwargs,
            })
        )
        
        # request.raise_for_status()
        
        if request.status_code != 200:
            return Completion(error=_check_for_errors(request))
        
        try:
            response = request.json()
        except json.JSONDecodeError:
            return Completion(error=f"Failed to decode JSON response from provider. Raw output: {request.text[:200]}")

        usage = response.get('usage', {})
        total_tokens = usage.get('total_tokens') if usage else None

        if not response.get('choices'):
            return Completion(error="Response contained no choices")

        message = response['choices'][0]['message']
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