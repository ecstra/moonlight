import asyncio
from typing import Type, Union, Optional, Any
from textwrap import dedent
from pydantic import BaseModel
from dataclasses import is_dataclass, dataclass

from .base import Content
from .history import AgentHistory
from ..helpers import ModelConverter
from ..provider import Provider, GetCompletion, Completion, CheckModel

class AgentError(Exception):
    pass

class Agent:
    """
    The Agent class acts as a high-level interface for interacting with various AI providers.
    It manages the conversation history, system role, and optional output schema validation.
    
    Attributes:
        _model (str): The identifier of the AI model to be used.
        _provider (Provider): The provider instance handling the API calls.
        _output_schema (Optional[Union[Type[BaseModel], Type[dataclass]]]): Optional schema for structured output.
        _params (dict): Additional parameters for the model configuration.
        _total_tokens (int): Total tokens consumed during the agent's interaction.
        _history (AgentHistory): Manages the conversation history.
    """
    def __init__(
        self,
        provider: Provider,
        model: str,
        output_schema: Optional[Union[Type[BaseModel], Type[dataclass]]] = None,
        system_role: str = "",
        image_gen: bool = False,
        **kwargs
    ):
        """
        Initializes the Agent with a provider, model, and optional configurations.

        Args:
            provider (Provider): The AI provider to use (e.g., OpenAI, Anthropic).
            model (str): The specific model identifier (e.g., 'gpt-4', 'claude-3').
            output_schema (Optional[Union[Type[BaseModel], Type[dataclass]]]): 
                A Pydantic BaseModel or Python dataclass to define the expected structured output.
            system_role (str): The initial system instruction for the agent. Defaults to an empty string.
            **kwargs: Additional model parameters such as 'temperature', 'top_p', etc.
        """
        # Agent params
        self._model = model
        self._provider = provider
        self._output_schema = output_schema
        self._params = kwargs
        self._image_gen = image_gen
        
        # total tokens
        self._total_tokens = 0
        
        # History
        self._history = None
        
        # Validate params
        self._model_data = None
        self._validate()
        
        # construct system role
        self._construct_sys_role(system_role)
        
    def get_total_tokens(self) -> int: 
        """
        Returns the total number of tokens used by the agent during its operation.

        Returns:
            int: The total count of tokens.
        """
        return self._total_tokens
    
    def get_history(self):
        """
        Retrieves the current conversation history.

        Returns:
            list: A list of message objects representing the conversation history.
        """
        return self._history.get_history()
    
    def _validate(self):
        """
        Validates the initialization parameters.

        Raises:
            AgentError: If the model name is empty, if unknown parameters are provided, 
                        or if the output schema is invalid.
        """
        if self._image_gen and self._output_schema:
            raise AgentError("Image Generation does not support Custom Output Schemas")
        
        if not self._model or self._model == "":
            raise AgentError("Model cannot be empty")
         
        allowed = { 
            "tools", "temperature", "top_p", "top_k", "plugins",
            "tool_choice", "text", "reasoning", "max_output_tokens",
            "frequency_penalty", "presence_penalty", "repetition_penalty",
            "response_format", "verbosity", "modalities", "max_completion_tokens"
        }
        
        unknown = set(self._params) - allowed
        
        if unknown: raise AgentError(f"Unknown properties: {unknown}")
        
        if self._output_schema and (not is_dataclass(self._output_schema) and not hasattr(self._output_schema, 'model_fields')):
            raise AgentError("Output Schema must be either a DataClass or BaseModel.")
        
        # Validate the model and it's params
        self._model_data = asyncio.run(CheckModel(
            provider=self._provider,
            model=self._model
        ))
        
        if not self._model_data["model_exists"]:
            raise AgentError(f"Model {self._model} does not exist in the given provider.")
        
        if self._image_gen and ("image" not in self._model_data["output_modalities"]):
            raise AgentError("This model does not support image generation")
        
        if self._params.get("modalities"):
            if self._model_data["output_modalities"]:
                unsupported_modalities = set(self._params.get("modalities")) - set(self._model_data["output_modalities"])
                if unsupported_modalities:
                    raise AgentError(f"The following modalities are not supported by the model: {unsupported_modalities}")
                
        max_allowed_tokens =  self._model_data["max_completion_tokens"]
        if max_allowed_tokens:
            if self._params.get("max_completion_tokens"):
                if self._params['max_completion_tokens'] > max_allowed_tokens:
                    raise AgentError(f"Max Completion tokens {self._params['max_completion_tokens']} exceeds model limits of {max_allowed_tokens}")
            
            if self._params.get("max_tokens"):
                if self._params["max_tokens"] > max_allowed_tokens:
                    raise AgentError(f"Max Completion tokens {self._params['max_tokens']} exceeds model limits of {max_allowed_tokens}")
            
    def _construct_sys_role(self, sys_role: str):
        """
        Constructs and updates the system role message, appending schema instructions if applicable.

        Args:
           sys_role (str): The base system role string.
        """
        sys_role = sys_role.strip()
        if sys_role == "": sys_role = "You are a helpful assistant."
            
        if self._output_schema:
            sys_role += dedent(f"""

            ---
            
            ## Important
            
            Ensure to return your responses only in JSON format.
            
            Here is the JSON schema of your output:
            ```json
            {ModelConverter.model_to_schema(self._output_schema)}
            ```
            
            and here is the Model Schema for reference:
            ```
            {ModelConverter.model_to_string(self._output_schema)}
            ```
            
            """)
        
        if self._history:
            self._history.update_system_role(sys_role)
        else:
            self._history = AgentHistory(sys_role)
    
    def update_system_role(
        self, 
        system_role: str,
    ):
        """
        Updates the system role for the agent.

        Args:
            system_role (str): The new system role string.
        """
        self._construct_sys_role(system_role)
    
    def clear(self):
        """
        Clears the conversation history (except the system role).
        """
        self._history.clear_history()
        
    async def run(self, prompt: Content) -> Union[Completion, Any]:
        """
        Executes the agent with a given prompt, handling history and optional schema parsing.

        Args:
            prompt (Content): The user input content.

        Returns:
            Union[Completion, Any]: A Completion object containing the response, 
                                    or a parsed object if an output schema is defined.
        """
        
        if prompt.images and ("image" not in self._model_data["input_modalities"]):
            raise AgentError("This model does not support image inputs.")
        
        custom_output = True if self._output_schema else False
        
        await self._history.add(
            role="user",
            content=prompt
        )
        
        optional = {}

        if custom_output:
            optional["response_format"] = { "type": "json_object" }
        
        if self._image_gen:
            optional["modalities"] = ["text", "image"]
            
        response = await GetCompletion(
            provider=self._provider,
            model=self._model,
            messages=self.get_history(),
            **self._params,
            **optional
        )

        await self._history.add(
            role="assistant",
            content=Content(
                text=response.content,
                images=response.images
            )
        )
        
        self._total_tokens = response.total_tokens

        # if response.error:
        #     raise AgentError(f"Agent failed to generate response: {response.error}")
        
        if custom_output and not response.error:
            try:
                response = ModelConverter.json_to_model(self._output_schema, response.content)
            except: 
                pass
        
        return response

    def __str__(self):
        """Returns a string description of the Agent."""
        return f"Agent(model='{self._model}', provider={self._provider.__class__.__name__})"

    def __repr__(self):
        """Returns a stable string representation of the Agent."""
        return (
            f"Agent(provider={self._provider!r}, model='{self._model}', "
            f"output_schema={self._output_schema!r})"
        )