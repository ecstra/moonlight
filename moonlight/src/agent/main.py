from typing import Type, Union, Optional, Any, Dict
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
    A high-level interface for interacting with various AI language models through different providers.
    
    The Agent class provides a unified API for communicating with AI models while handling
    conversation history management, system role configuration, structured output validation,
    token tracking, and multi-modal support (text and images).
    
    Key Features:
        - Multi-provider support: Works with various AI providers (OpenAI, Anthropic, etc.)
        - Conversation history: Automatically manages message history for context-aware conversations
        - Structured output: Validates and parses responses using Pydantic models or dataclasses
        - Token tracking: Monitors both contextual and consumed tokens across interactions
        - Image generation: Supports models with image generation capabilities
        - Persistance: Optional for single-turn interactions without history persistence
        - Flexible configuration: Customizable parameters like temperature, top_p, top_k, etc.
    
    Attributes:
        _model (str): The identifier of the AI model to be used (e.g., 'gpt-4', 'claude-3').
        _provider (Provider): The provider instance handling the API calls.
        _output_schema (Optional[Union[Type[BaseModel], Type[dataclass]]]): 
            Optional schema for structured output validation and parsing.
        _params (dict): Additional parameters for model configuration (temperature, top_p, etc.).
        _image_gen (bool): Whether the agent is configured for image generation.
        _persistence (bool): If False, conversation history is cleared after each interaction.
        _contextual_tokens (int): Number of tokens currently in the conversation context.
        _consumed_tokens (int): Total tokens consumed across all interactions.
        _history (AgentHistory): Manages the conversation history and system role.
        _model_data (dict): Validated model metadata including capabilities and limitations.
    
    Example:
        ```python
        from moonlight import Agent, Provider
        from pydantic import BaseModel
        
        # Basic usage
        agent = Agent(
            provider=Provider.OPENAI,
            model="gpt-4",
            system_role="You are a helpful coding assistant."
        )
        response = await agent.run(Content(text="Explain recursion"))
        
        # Structured output
        class Response(BaseModel):
            answer: str
            confidence: float
        
        agent = Agent(
            provider=Provider.OPENAI,
            model="gpt-4",
            output_schema=Response,
            temperature=0.7
        )
        result = await agent.run(Content(text="What is Python?"))
        print(result.answer, result.confidence)
        ```
    
    Raises:
        AgentError: If invalid configuration is provided or model constraints are violated.
    """
    def __init__(
        self,
        provider: Provider,
        model: str,
        output_schema: Optional[Union[Type[BaseModel], Type[dataclass]]] = None,
        persistence: bool = True,
        system_role: str = "",
        image_gen: bool = False,
        **kwargs
    ):
        # Agent params
        self._model = model
        self._provider = provider
        self._output_schema = output_schema
        self._params = kwargs
        self._image_gen = image_gen
        self._persistence = persistence
        
        # total tokens
        self._contextual_tokens = 0 # to count how many tokens are there in "history" right now
        self._consumed_tokens = 0   # to count total consumption of tokens across all messages
        
        # History
        self._history = None
        
        # Validate params
        self._model_data = None
        self._validate()
        
        # construct system role
        self._construct_sys_role(system_role)
        
    def get_total_tokens(self) -> Dict[str, int]: 
        """
        Returns the total number of tokens used by the agent during its operation.

        Returns:
            Dict[str, int]: The total count of tokens in current context and consumed.
        """
        return {
            "context": self._contextual_tokens,
            "consumed": self._consumed_tokens
        }
    
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
        
        # Mutually exclusive. If Image Gen is on
        # then response must be Completion type
        # since it's hard to figure out where to put the images in 
        # a custom model
        if self._image_gen and self._output_schema:
            raise AgentError("Image Generation does not support Custom Output Schemas")
        
        # Model cannot be empty...
        # How else would this work?
        if not self._model or self._model == "":
            raise AgentError("Model cannot be empty")
         
        # All the allowed params 
        # that can be passed through kwargs
        # and provider accepts
        # (removed response_format & modalities since those are controlled automatically)
        allowed = { 
            "tools", "temperature", "top_p", "top_k", "plugins",
            "tool_choice", "text", "reasoning", "max_output_tokens",
            "frequency_penalty", "presence_penalty", "repetition_penalty",
            "verbosity", "max_completion_tokens"
        }
        
        unknown = set(self._params) - allowed
        
        if unknown: raise AgentError(f"Unknown properties: {unknown}")
        
        # Output Schema check. 
        # Must be dataclass or BaseModel
        # Nothing else allowed since ModelConverter can handle only those
        if self._output_schema and (not is_dataclass(self._output_schema) and not hasattr(self._output_schema, 'model_fields')):
            raise AgentError("Output Schema must be either a DataClass or BaseModel.")
        
    async def _ensure_initialized(self):
        # Lazy init...
        # Just check before running
        # Run is async anyways, so no need for a model factory
        if self._model_data is not None: return
        
        # Validate the model and it's params
        # using the actual data gotten from the provider
        self._model_data = await CheckModel(
            provider=self._provider,
            model=self._model
        )
        
        if not self._model_data["model_exists"]:
            raise AgentError(f"Model {self._model} does not exist in the given provider.")
        
        # Modalities Check
        # Can the model handle image input/output?
        if self._image_gen and ("image" not in self._model_data["output_modalities"]):
            raise AgentError("This model does not support image generation")
        
        # Max allowed tokens check (sometimes it's null)                
        max_allowed_tokens =  self._model_data["max_completion_tokens"]
        if max_allowed_tokens:
            if self._params.get("max_completion_tokens"):
                if self._params['max_completion_tokens'] > max_allowed_tokens:
                    raise AgentError(f"Max Completion tokens {self._params['max_completion_tokens']} exceeds model limits of {max_allowed_tokens}")
            
            if self._params.get("max_tokens"):
                if self._params["max_tokens"] > max_allowed_tokens:
                    raise AgentError(f"Max Completion tokens {self._params['max_tokens']} exceeds model limits of {max_allowed_tokens}")
        
        # TODO: Get the max context length and pass it to history if persistance is enabled
        # Then in history auto cleanup the old messages if it exceeds context length * 0.8 
        # or summarize all messages using the same agent and add it to system role
        
    def _construct_sys_role(self, sys_role: str):
        """
        Constructs and updates the system role message, appending schema instructions if applicable.

        Args:
           sys_role (str): The base system role string.
        """
        # default role if nothing is provided
        sys_role = sys_role.strip()
        if sys_role == "": sys_role = "You are a helpful assistant."
        
        # Tell model about the model schema and return type
        # Ask it nicely to give it in that format
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
        
        # Update the system role if it already exists.
        # But if it's the first time initializing agent, then 
        # create the history
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
        Clears the agent. This includes context token count and history.
        """
        self._contextual_tokens = 0
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
        
        # Lazy init. Fill up _model_data if not already filled and perform a check
        await self._ensure_initialized()
        
        # TODO: Evaluate whether this "run" will grow. 
        # If yes, then perhaps put it as a function in it's own file
        # or folder. Then figure out data passing between agents.
        # And then expand to sequential and parallel running.
        
        # Check if prompt has images, but images should not be provided.
        if prompt.images and ("image" not in self._model_data["input_modalities"]):
            raise AgentError("This model does not support image inputs.")
        
        # Custom Output is set to true if output schema is provider
        custom_output = True if self._output_schema else False
        
        # Add the user message to history of the agent
        await self._history.add(
            role="user",
            content=prompt
        )
        
        # Construct optional parameters that must
        # be passed to the provider. This is based on 
        # modality or response format
        
        optional = {}

        if custom_output:
            optional["response_format"] = { "type": "json_object" }
        
        if self._image_gen:
            optional["modalities"] = ["text", "image"]
        
        # Get the response from the provider...
        response = await GetCompletion(
            provider=self._provider,
            model=self._model,
            messages=self.get_history(),
            **self._params,
            **optional
        )
        
        # Check if persistance is enabled
        # if yes, add assistant message to history
        # If not, then clear the history of the added user message 
        if self._persistence:
            await self._history.add(
                role="assistant",
                content=Content(
                    text=response.content,
                    images=response.images
                )
            )
        else:
            # This will clear the user message
            # but will keep the system role as is.
            self._history.clear_history()
        
        # Count the tokens. Logic:
        # a. For context, the history grows. Each time the user response and assistant
        #    message is added, it's added to history and sent to the provider. This is
        #    counted as "input" tokens and output is what the assistant generates next.
        #    So context tokens is whatever is in history, which is what the provider sends
        #    on every response.
        # b. For consumption, it grows quadratically (ignoring caching). So token's consumed by 
        #    the provider is the history provided to the model. And since history grows each time
        #    so does the token. This is ofc ignoring caching but oh well.
        #    eg: 1024 + (1024 + 128) + (1024 + 128 + 2048) ... so on
        #    where 1024 is the tokens for first response, 
        #    128 for second (+1024 input), 2048 for the third (+1024 + 128 input)
        # Note: Can't add None to int D:
        
        if response.total_tokens:
            self._contextual_tokens = response.total_tokens
            self._consumed_tokens  += response.total_tokens
        
        # Check if agent has an output to follow
        # If we can't follow that output, throw an error
        # or if there is an error, throw that
        # But, if we are using Completion output, then just output
        # the Completion object since that has an exclusive "error" property.
        # TODO: Might need to rethink this logic. Would consistency of throwing be beter?
        if custom_output:
            if response.error:
                raise AgentError(f"Agent failed to generate response: {response.error}")
            try:
                return ModelConverter.json_to_model(self._output_schema, response.content)
            except:
                pass
    
        return response

    def __str__(self):
        """
        Returns a human-readable string description of the Agent.
        
        Displays the model, provider, and key configuration details like persistence mode,
        image generation capability, and output schema status.
        """
        schema_name = self._output_schema.__name__ if self._output_schema else "None"
        mode = "persistent" if self._persistence else "stateless"
        
        parts = [
            f"model={self._model}",
            f"provider={self._provider.__class__.__name__}",
            f"mode={mode}"
        ]
        
        if self._output_schema:
            parts.append(f"schema={schema_name}")
        
        if self._image_gen:
            parts.append("image_gen=True")
        
        if self._params:
            key_params = {k: v for k, v in self._params.items() if k in ['temperature', 'top_p', 'max_completion_tokens']}
            if key_params:
                parts.append(f"params={key_params}")
        
        return f"Agent({', '.join(parts)})"

    def __repr__(self):
        """
        Returns a detailed string representation of the Agent for debugging.
        
        Includes all initialization parameters needed to recreate the agent instance.
        """
        parts = [
            f"provider={self._provider!r}",
            f"model={self._model!r}",
        ]
        
        if self._output_schema:
            parts.append(f"output_schema={self._output_schema!r}")
        
        if not self._persistence:
            parts.append(f"persistence={self._persistence!r}")
        
        if self._image_gen:
            parts.append(f"image_gen={self._image_gen!r}")
        
        if self._params:
            for key, value in sorted(self._params.items()):
                parts.append(f"{key}={value!r}")
        
        return f"Agent({', '.join(parts)})"