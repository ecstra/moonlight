import httpx, json, re
from dataclasses import dataclass

from typing import Dict, List, Optional
from .main import Provider

class GetCompletionError(Exception): pass

@dataclass
class Completion:
    content: Optional[str] = None
    reasoning: Optional[str] = None
    images: Optional[List[str]] = None
    error: Optional[str] = None
    total_tokens: Optional[int] = None
    
    def __str__(self) -> str:
        if self.error:
            return f"CompletionError({self.error})"
        return self.content if self.content is not None else ""

    def __repr__(self) -> str:
        content_preview = (self.content[:50] + "...") if self.content and len(self.content) > 50 else self.content
        return (
            f"Completion("
            f"content={content_preview!r}, "
            f"images={len(self.images) if self.images else 0}, "
            f"error={self.error!r}, "
            f"total_tokens={self.total_tokens})"
        )

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
            case 408: error_msg = "Request timed out"
            case 429: error_msg = "Rate limited - too many requests"
            case 502: error_msg = "Model is down or returned invalid response"
            case 503: error_msg = "No available model provider meets your routing requirements"
    
    return error_msg

async def GetCompletion(
    provider: Provider,
    model: str,
    messages: List[Dict[str, str]],
    json_mode: Optional[bool] = False,
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
    
    optional = {}

    if json_mode:
        optional["response_format"] = { "type": "json_object" }
        
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
                **optional
            })
        )
        
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