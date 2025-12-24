import logging, re
from typing import List, Dict
from openai import OpenAI
from ..agent_architecture.base import MoonlightProvider

logger = logging.getLogger("hive")

class LLMProviderException(Exception):
    pass

class CompletionInput:
    def __init__(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        max_context: int,
        max_output_length: int,
        temperature: float,
    ):
        self.model_name = model_name
        self.messages = messages
        self.max_context = max_context
        self.max_output_length = max_output_length
        self.temperature = temperature
        
class OpenAIProvider:
    def __init__(
            self, 
            provider: MoonlightProvider,
        ):            
        
        self.provider = provider
        self.llm_sdk = None

        self._construct_openai_sdk()
    
    def _construct_openai_sdk(self):        
        if self.provider.provider_name == 'openai':
            self.llm_sdk = OpenAI(
                api_key = self.provider.api_key,
            )
        else:
            self.llm_sdk = OpenAI(
                base_url = self.provider.provider_url, 
                api_key = self.provider.api_key,
            )

    def _clean_completion(self, completion):
        thought = None
        answer = None
        
        try:
            response = completion.choices[0].message.content
        except Exception as e:
            raise LLMProviderException(f"Error Cleaning Completion: {str(e)}")
        
        if not response:
            return "<thought>None</thought><answer>None</answer>"
        
        thought_search = re.search(r'<thought>(.*?)</thought>', response, re.DOTALL)
        think_search = re.search(r'<think>(.*?)</think>', response, re.DOTALL)

        if hasattr(completion.choices[0].message, "reasoning"):
            thought = getattr(completion.choices[0].message, "reasoning", None)
        elif hasattr(completion.choices[0].message, "reasoning_content"):
            thought = getattr(completion.choices[0].message, "reasoning_content", None)
        elif thought_search:
            thought = thought_search.group(1).strip()
            response = re.sub(r'<thought>.*?</thought>', '', response, flags=re.DOTALL)
        elif think_search:
            thought = think_search.group(1).strip()
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        elif '</think>' in response and '<think>' not in response:
            unbalanced = re.search(r'^(.*?)</think>', response, re.DOTALL)
            if unbalanced:
                thought = unbalanced.group(1).strip()
                response = response.replace(unbalanced.group(0), '')
                    
        answer_search = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        answer = answer_search.group(1).strip() if answer_search else response

        # Fix: Convert None values to "None" string
        thought_str = "None" if thought is None else thought
        answer_str = "None" if answer is None else answer
        
        raw_message = f'<thought>{thought_str}</thought><answer>{answer_str}</answer>'
        return raw_message
    
    def get_completion(
            self,
            completion_input: CompletionInput
        ):
        
        """
        Get completion from the LLM based on the input parameters.
        """
        
        params = {
            "model": completion_input.model_name,
            "messages": completion_input.messages,
        }
        
        max_context_unsupported_providers = ["github"]
        
        if self.provider not in max_context_unsupported_providers:
            if not completion_input.max_output_length == -1:                
                params["max_tokens"] = completion_input.max_output_length
            params["temperature"] = completion_input.temperature

        if self.provider in ["openrouter", "deepseek"]:
            params["extra_body"] = {"include_reasoning": True}
            params["extra_query"] = {"include_reasoning": True}
        
        if not self.provider.thinking:
            if "reasoning_effort" in params:
                del params["reasoning_effort"]
        try:
            completion = self.llm_sdk.chat.completions.create(**params)
        except Exception as e:
            raise LLMProviderException(f"Error running OpenAI SDK: {str(e)}")
        
        raw_message = self._clean_completion(completion)        
        return raw_message