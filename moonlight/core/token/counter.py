import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRANSFORMERS_NO_PYTORCH"] = "1"

import tiktoken, re, math
from typing import Dict, Any, Optional
from transformers import AutoTokenizer

class TokenCounterErrror(Exception):
    pass

class TokenCounter:
    """
    A class for counting tokens across different LLM models
    """
    
    def __init__(
            self, 
            cache_dir: str = ".token_cache",
            use_offline_only: bool = False,
            use_fast: bool = False,
            model: str = "cl100k_base"
        ):
        """
        Initialize the token counter
        
        Args:
            cache_dir: Directory to store tokenizer data
            use_offline_only: If True, will only use cached tokenizers and won't download new ones
            use_fast: If True, will use fast tokenization methods
            model: Model name or ID for which to count tokens
        """
        if not os.path.isabs(cache_dir):
            # Get the directory where this file is located
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            self.cache_dir = os.path.join(current_file_dir, cache_dir)
        else:
            self.cache_dir = cache_dir
            
        self.use_offline_only = use_offline_only
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
        # Variables for tokenizer
        self.use_fast = use_fast
        self.model = model
        
        # Load the tokenizer
        if self.model and not self.use_fast:
            self.tokenizer = self._get_tokenizer_for_model()
            
        # Overhead and Multiplier
        self.overhead = 5     # 1 for role, 4 for history syntax
        self.multiplier = 1.1 # Multiplier for overhead of provider tokens

    def _detect_model_family(self) -> Optional[str]:
        """Detect the model family from the model name"""
        model_name_lower = self.model.lower()
        
        patterns = {
            "gpt": r"(gpt-3\.5|gpt-4)",
            "llama": r"(llama|meta-llama)",
            "deepseek": r"(deepseek)",
            "gemini": r"(gemini|gemma|google|palm)"
        }
        
        for family, pattern in patterns.items():
            if re.search(pattern, model_name_lower):
                return family
                
        return None
    
    def _get_tiktoken_tokenizer(self, tokenizer_model: str):
        """Get a tiktoken tokenizer"""
        try:
            # For exact model matches
            return tiktoken.encoding_for_model(tokenizer_model)
        except:
            # Fallback to cl100k_base (used by GPT-4, Claude, etc.)
            try:
                return tiktoken.get_encoding("cl100k_base")
            except:
                return None
    
    def _map_model_to_tokenizer(self) -> str:
        """Map model name to an appropriate tokenizer model ID"""
        family = self._detect_model_family()
        
        # Use default mappings for common model families
        if family == "llama":
            return "meta-llama/Llama-3.2-1B-Instruct"
        elif family == "gpt":
            return "cl100k_base"  # OpenAI's tokenizer encoding
        elif family == "deepseek":
            return "deepseek-ai/DeepSeek-V3-0324"
        elif family == "gemini":
            return "google/gemma-7b"
        
        # Default to OpenAI's tokenizer for unknown models
        return "cl100k_base"
    
    def _get_tokenizer_for_model(self):
        """
        Get the appropriate tokenizer for a given model
        """

        # Get the appropriate tokenizer model ID
        tokenizer_model = self._map_model_to_tokenizer()
        
        # For OpenAI models, use tiktoken
        if tokenizer_model == "cl100k_base" or "gpt" in self.model.lower():
            tokenizer = self._get_tiktoken_tokenizer(self.model)
        else:
            if self.use_offline_only:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        tokenizer_model,
                        cache_dir=self.cache_dir,
                        local_files_only=True
                    )
                except Exception as e:
                    raise TokenCounterErrror(f"Failed to load tokenizer from cache: {e}")
            
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_model, 
                    cache_dir=self.cache_dir,
                )

        return tokenizer
    
    def _apply_overhead_and_multiplier(
            self, 
            token_count: int
        ) -> int:
        """
        Apply overhead and multiplier to the token count
        """
        # Apply overhead and multiplier
        adjusted_count =  int((token_count + self.overhead) * self.multiplier)
        
        return adjusted_count
    
    def _fast_count_tokens(
            self, 
            text: str
        ) -> int:
        """
        Count tokens using a fast method (regex)
        May not be accurate for all models (Overperforms for english, underperforms for other languages and code)
        """
        words = len(re.findall(r'\w+', text))
        spaces = len(re.findall(r'\s+', text))
        punctuation = len(re.findall(r'[^\w\s]', text))
        token_count = words + spaces + punctuation
        
        return token_count
        
    def count_tokens(
            self, 
            text: str,
        ) -> Dict[str, Any]:
        
        """
        Count the number of tokens in a given text for a specific model
        """
        token_count = 0
        
        # Get accurate count
        if self.use_fast:
            token_count = self._fast_count_tokens(text)    
        else:
            if hasattr(self.tokenizer, 'encode'):  # tiktoken style
                token_count = len(self.tokenizer.encode(text))
            elif hasattr(self.tokenizer, '__call__'):  # transformers style
                token_count = len(self.tokenizer(text)["input_ids"])
        
        # Apply overhead and multiplier
        token_count = self._apply_overhead_and_multiplier(token_count)
        
        # Return the token count
        return token_count

            
    def _count_image_tokens(
            self, 
            width: int, 
            height: int
        ):
        """
        Calculate the number of tokens for an image based on its dimensions.
        Formula found somewhere on the internet (GPT forum).
        """
        if width > 1024 or height > 1024:
            if width > height:
                height = int(height * 1024 / width)
                width = 1024
            else:
                width = int(width * 1024 / height)
                height = 1024
    
        h = math.ceil(height / 512)
        w = math.ceil(width / 512)
        total = 85 + 170 * h * w
        
        return total
    
if __name__ == "__main__":
    import time
    
    # Initialize the TokenCounter
    token_counter = TokenCounter(
        model="gpt-3.5-turbo"
    )
    
    # Example text
    text = "Hello, how are you doing today? This is a test for counting tokens."
    
    # Count tokens
    start_time = time.time()
    token_count = token_counter.count_tokens(text)
    end_time = time.time()
    
    print(f"Token count: {token_count}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    print(f"Token count for '{text}' using gpt-3.5-turbo: {token_count}")