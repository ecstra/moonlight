from urllib.parse import urlparse

class ProviderError(Exception): pass

# Some mapped endpoints for 
# easy access to commonly used providers.
# Extend as needed
ENDPOINTS = {
    "openai": "https://api.openai.com/v1",
    "together": "https://api.together.xyz/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com",
    "groq": "https://api.groq.com/openai/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "hf_together": "https://huggingface.co/api/inference-proxy/together",
    "hf": "https://router.huggingface.co/v1",
}

class Provider:
    """
    A class for LLM providers.
    """
    def __init__(
        self,
        source: str,
        api: str
    ):
        # Source of the Provider
        # can be the endpoint or a common provider that's already mapped
        self._source = source
        
        # API key for that provider
        self._api = api
    
        # Validate
        self._validate_params()
    
    def get_source(self): return self._source
    def get_api(self): return self._api
    
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