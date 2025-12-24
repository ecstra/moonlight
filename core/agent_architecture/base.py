from rich.console import Console
import logging

logger = logging.getLogger(__name__)

console  = Console()

class MoonlightProviderException(Exception): pass

class MoonlightProvider:
    def __init__(
            self,
            provider_name: str = None,
            provider_url: str = None,
            api_key: str = None,
            thinking: bool = True
        ):
        
        self.provider_name = provider_name
        self.provider_url = provider_url
        self.api_key = api_key
        self.thinking = thinking
        
        if not self.provider_name and not self.provider_url:
            raise MoonlightProviderException("Provider Name/URL is not set.")
        
            
        if not self.api_key:
            raise ValueError("API key is not set. Set it in the Agent.")
        
        self._validate_provider()
        
    def _validate_provider(self):
        allowed_providers = [
            'openai', 
            'together', 
            'openrouter', 
            'deepseek', 
            'groq', 
            'google', 
            'hf_together', 
            'hf', 
            'github'
        ]
        
        if (self.provider_name not in allowed_providers) and (not self.provider_url):
            raise ValueError(f"Provider '{self.provider}' is not supported. Supported providers are: {', '.join(allowed_providers)}")
        
        if not self.provider_url:
            self._map_provider_to_api()
            
    def _map_provider_to_api(self):
        if self.provider_name == 'openai':
            self.provider_url = ""
        elif self.provider_name == 'together':
            self.provider_url = "https://api.together.xyz/v1"
        elif self.provider_name == 'openrouter':
            self.provider_url = "https://openrouter.ai/api/v1"
        elif self.provider_name == 'deepseek':
            self.provider_url = "https://api.deepseek.com"
        elif self.provider_name == 'groq':
            self.provider_url = "https://api.groq.com/openai/v1"
        elif self.provider_name == 'google':
            self.provider_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        elif self.provider_name == 'hf_together':
            self.provider_url = "https://huggingface.co/api/inference-proxy/together"
        elif self.provider_name == 'hf':
            self.provider_url = "https://api-inference.huggingface.co/v1/"
        elif self.provider_name == 'github':
            self.provider_url = "https://models.inference.ai.azure.com"
    
    def __repr__(self):
        console.print(f"[bold green]Provider Name:[/bold green] {self.provider_name}")
        console.print(f"[bold green]Provider URL:[/bold green] {self.provider_url}")
        console.print(f"[bold green]Provider initialized successfully.[/bold green]")
        return ""