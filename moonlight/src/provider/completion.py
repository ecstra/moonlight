import httpx, json, re, random, asyncio
from dataclasses import dataclass

from typing import Dict, List, Optional
from .main import Provider

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

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
            
            # Fallback to top-level message if no specific error was extracted
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
            case 404: raise GetCompletionError("Provider endpoint is incompatible (missing '/chat/completions').")
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
    
    # OpenAI Structure
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
        # Retry Mechanism
        for attempt in range(max_retries + 1):
            try:
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