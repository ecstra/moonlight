from enum import Enum
from urllib.parse import urlparse

class ProviderError(Exception): pass

# Anthropic authenticates with x-api-key + a dated version header (rather than
# Bearer auth) and uses different request/response shapes than OpenAI-compatible
# providers. See structures.txt.
ANTHROPIC_VERSION = "2023-06-01"

class EndpointType(Enum):
    OPENAI    = "openai"
    ANTHROPIC = "anthropic"

# Some mapped endpoints for 
# easy access to commonly used providers.
# Extend as needed
ENDPOINTS = {
    "openai":     "https://api.openai.com/v1",
    "together":   "https://api.together.xyz/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek":   "https://api.deepseek.com",
    "groq":       "https://api.groq.com/openai/v1",
    "anthropic":  "https://api.anthropic.com/v1"
}

class Provider:
    def __init__(
        self,
        source: str,
        api: str,
        endpoint_type: EndpointType = EndpointType.OPENAI
    ):
        # Source of the Provider
        # can be the endpoint or a common provider that's already mapped
        self._source = source

        # API key for that provider
        self._api = api

        # Wire format used by the provider (OpenAI-compatible vs Anthropic).
        # Controls auth headers, request body and response parsing downstream.
        self._endpoint_type = endpoint_type

        # Validate
        self._validate_params()

    def get_source(self)        -> str:          return self._source
    def get_api(self)           -> str:          return self._api
    def get_endpoint_type(self) -> EndpointType: return self._endpoint_type
    
    def _validate_params(self):
        if self._source == "" or not self._source:
            raise ProviderError("Source is empty")
        
        if not isinstance(self._source, str):
            raise ProviderError("Source must be a string")
        
        if ENDPOINTS.get(self._source, None):
            self._source = ENDPOINTS[self._source]
        elif not self._is_valid_endpoint(self._source):
            raise ProviderError(f"Invalid provider endpoint/source provided. Please check the URL or try one of the defaults: {ENDPOINTS.keys()}")
        
        if self._api and not isinstance(self._api, str):
            raise ProviderError("API must be a string")
        
        if self._api == "" or not self._api:
            raise ProviderError("API key is empty")

        if not isinstance(self._endpoint_type, EndpointType):
            raise ProviderError("endpoint_type must be an EndpointType")

        # Auto-select the Anthropic wire format when the source points at
        # Anthropic, so callers don't have to also pass endpoint_type. Runs on
        # the resolved source, so it catches both the "anthropic" shorthand and
        # a direct api.anthropic.com URL.
        if "anthropic" in self._source.lower():
            self._endpoint_type = EndpointType.ANTHROPIC

    def _is_valid_endpoint(self, source: str):
        try:
            def is_valid(url: str):
                r = urlparse(url)
                if not (r.scheme and r.netloc): return False
                # Reject "plain" strings acting as domains unless it's localhost
                # e.g. "openai" (valid hostname technically) -> False
                # e.g. "openai.com" -> True
                # e.g. "localhost:3000" -> True
                return "." in r.netloc or "localhost" in r.netloc

            # already has scheme
            if "://" in source:
                return is_valid(source) and urlparse(source).scheme in ("http", "https")

            # try http and https schemas
            return is_valid("http://" + source) or is_valid("https://" + source)

        except Exception:
            return False

if __name__ == "__main__":
    # Example
    provider = Provider(
        api="...",
        source="openai"
    )