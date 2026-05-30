from typing import Type, Union, Optional, Any, Dict
from textwrap import dedent
from pydantic import BaseModel
from dataclasses import is_dataclass, dataclass

from .base import Content
from .history import AgentHistory, TypeAgentHistory
from ..helpers import ModelConverter
from ..provider import (
    Provider,
    GetCompletion, Completion,
    CheckModel
)

class AgentError(Exception):
    pass

class Agent:
    """
    A high-level, async interface for talking to any supported model behind a Provider.

    Args:
        provider (Provider): Configured LLM provider (endpoint, API key, wire format).
        model (str): Model identifier to use (e.g. "gpt-4o", "claude-opus-4-6").
        output_schema (Optional[BaseModel | dataclass]): If set, run() returns a validated
            instance of this schema instead of a Completion. Mutually exclusive with image_gen.
        persistence (bool): If True (default), conversation history is kept across run() calls;
            if False, history is cleared after each run (stateless / single-turn).
        system_role (str): System prompt for the agent. Defaults to a generic assistant.
        image_gen (bool): If True, request image output from an image-capable model.
            Mutually exclusive with output_schema.
        schema_retries (int): When output_schema parsing fails, how many times to show the model
            its error and ask it to fix the response before falling back (default 2).
        summarize_threshold (float): Fraction (0-1) of the model's context length at which older
            turns are auto-summarized into the system role. Set to 0 to disable (default 0.85).
        keep_recent (int): Most recent messages kept verbatim when summarizing (default 2).
        **kwargs: Sampling/config params forwarded to the provider (temperature, top_p, top_k,
            max_completion_tokens, frequency_penalty, presence_penalty, etc.).

    Automatically handles:
        - Model validation on first run (existence, modalities, token limits).
        - Conversation history and system-role management.
        - Structured output: JSON mode, schema injection, validation, and self-correction retries.
        - Multimodal image input and image generation.
        - Auto-summarization of old turns as the context window fills.
        - Token tracking (current context and cumulative consumed).
        - Transient-failure retries and provider error reporting.

    Example:
        ```python
        from moonlight import Agent, Provider, Content
        from pydantic import BaseModel

        provider = Provider(source="anthropic", api="your-api-key")

        # Basic usage
        agent = Agent(
            provider=provider,
            model="claude-opus-4.8",
            system_role="You are a helpful coding assistant."
        )
        response = await agent.run(Content(text="Explain recursion"))

        # Structured output
        class Response(BaseModel):
            answer: str
            confidence: float

        agent = Agent(
            provider=provider,
            model="claude-opus-4.8",
            output_schema=Response,
            temperature=0.7
        )
        result = await agent.run(Content(text="What is Python?"))
        print(result.answer, result.confidence)
        ```
    """
    def __init__(
        self,
        provider: Provider,
        model: str,
        output_schema: Optional[Union[Type[BaseModel], Type[dataclass]]] = None,
        persistence: bool = True,
        system_role: str = "",
        image_gen: bool = False,
        schema_retries: int = 2,
        summarize_threshold: float = 0.85,
        keep_recent: int = 2,
        **kwargs
    ):
        # Agent params
        self._model = model
        self._provider = provider
        self._output_schema = output_schema
        self._params = kwargs
        self._image_gen = image_gen
        self._persistence = persistence

        # Self-correction attempts when a structured-output response fails schema
        # validation (the model is shown the error and asked to fix it) before
        # falling back to the raw response.
        self._schema_retries = schema_retries

        # Auto-summarization: once the conversation reaches summarize_threshold of
        # the model's context length, the oldest turns are folded into a running
        # summary (kept in the system role) and dropped, while keep_recent of the
        # most recent messages stay verbatim. Only engages when the provider
        # reports a context length; set summarize_threshold to 0 to disable.
        self._summarize_threshold = summarize_threshold
        self._keep_recent = keep_recent
        self._summary = ""            # running summary of folded-out turns
        self._base_system_role = ""   # system role without the summary section

        # total tokens
        self._contextual_tokens = 0 # to count how many tokens are there in "history" right now
        self._consumed_tokens = 0   # to count total consumption of tokens across all messages

        # History
        self._history = None

        # Validate params
        self._model_info = None
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

        # Self-correction retry count for structured output
        if not isinstance(self._schema_retries, int) or self._schema_retries < 0:
            raise AgentError("schema_retries must be a non-negative integer")

        # Auto-summarization config
        if not isinstance(self._summarize_threshold, (int, float)) or not (0 <= self._summarize_threshold <= 1):
            raise AgentError("summarize_threshold must be a number between 0 and 1")

        if not isinstance(self._keep_recent, int) or self._keep_recent < 0:
            raise AgentError("keep_recent must be a non-negative integer")

    async def _ensure_initialized(self):
        # Lazy init...
        # Just check before running
        # Run is async anyways, so no need for a model factory
        if self._model_info is not None: return

        # Validate the model and it's params
        # using the actual data gotten from the provider
        self._model_info = await CheckModel(
            provider=self._provider,
            model=self._model
        )

        if not self._model_info.model_exists:
            raise AgentError(f"Model {self._model} does not exist in the given provider.")

        # Modalities Check
        # Can the model handle image input/output?
        # Empty modalities = provider didn't report them (minimal /models
        # endpoints like OpenAI/DeepSeek) -> unknown, so don't block.
        if self._image_gen and self._model_info.output_modalities and ("image" not in self._model_info.output_modalities):
            raise AgentError("This model does not support image generation")

        # Max allowed tokens check (sometimes it's null)
        max_allowed_tokens =  self._model_info.max_completion_tokens
        if max_allowed_tokens:
            if (max_allowed_tokens > 0):
                if self._params.get("max_completion_tokens"):
                    if self._params['max_completion_tokens'] > max_allowed_tokens:
                        raise AgentError(f"Max Completion tokens {self._params['max_completion_tokens']} exceeds model limits of {max_allowed_tokens}")

                if self._params.get("max_tokens"):
                    if self._params["max_tokens"] > max_allowed_tokens:
                        raise AgentError(f"Max Completion tokens {self._params['max_tokens']} exceeds model limits of {max_allowed_tokens}")

    def _construct_sys_role(
        self,
        sys_role: str
    ):
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

        # Store the base system role (without any summary section) and apply it.
        self._base_system_role = sys_role
        self._apply_system_role()

    def _apply_system_role(self):
        # Effective system role = base instructions + the running summary (if any).
        effective = self._base_system_role
        if self._summary:
            effective += dedent(f"""

            ---

            ## Summary of earlier conversation

            {self._summary}
            """)

        # Update the system role if history exists, otherwise create it (first init).
        if self._history:
            self._history.update_system_role(effective)
        else:
            self._history = AgentHistory(effective)

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

    def _account_tokens(
        self, 
        response: Completion
    ):
        # Context = tokens in the current history (the last call)
        # consumed = cumulative total across every call this agent makes, retries included.
        if response.total_tokens:
            self._contextual_tokens = response.total_tokens
            self._consumed_tokens  += response.total_tokens

    async def _complete(
        self,
        optional: dict
    ) -> Completion:
        # One provider call over the current history, with token accounting.
        response = await GetCompletion(
            provider=self._provider,
            model=self._model,
            messages=self.get_history(),
            **self._params,
            **optional
        )
        self._account_tokens(response)
        return response

    def _serialize_messages(
        self, 
        messages: TypeAgentHistory
    ) -> str:
        """
        Flatten conversation messages into a plain-text transcript for the summarizer.

        Multimodal messages (content is a list of parts) keep their text and note
        each image as "[image]"; plain-text messages are used as-is.

        Args:
            messages (TypeAgentHistory): The messages to serialize.

        Returns:
            str: One "role: content" line per message, joined by newlines.
        """
        lines = []
        for m in messages:
            role = m.get("role", "?")
            content = m.get("content", "")
            if isinstance(content, list):
                parts = []
                for p in content:
                    if not isinstance(p, dict):
                        continue
                    if p.get("type") == "text":
                        parts.append(p.get("text", ""))
                    elif p.get("type") == "image_url":
                        parts.append("[image]")
                content = " ".join(parts)
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    async def _try_summarize(self):
        """
        Compact the conversation if it is nearing the model's context limit.

        Delegates to _compact_history once contextual token usage reaches
        summarize_threshold of the model's context length. No-op when
        summarization is disabled (threshold of 0) or the model is not
        compactable (no known context length), leaving history untouched
        (fail open).
        """
        
        if not self._summarize_threshold or not self._model_info.compactable:
            return
        
        limit = self._model_info.context_length
        
        if self._contextual_tokens < limit * self._summarize_threshold:
            return
        
        await self._compact_history()

    async def _compact_history(self):
        """
        Fold the oldest turns into the running summary and drop them.

        Summarizes everything except the system message and the most recent
        keep_recent messages (the boundary is nudged back so the kept slice
        starts on a user turn, which Anthropic requires). The summary is produced
        by a plain side-call to the same model, stored in self._summary, and
        injected into the system role; the folded turns are then removed while the
        recent turns are kept verbatim.

        Best effort: if the summary call errors or returns nothing, the history is
        left unchanged. The side-call's tokens count toward consumed totals but
        not the live context size.
        """
        history = self._history.get_history()

        # history[0] is the system message
        # only later turns are foldable.
        convo = history[1:]
        if len(convo) <= self._keep_recent:
            return

        # Keep the most recent messages verbatim. Nudge the boundary back so the
        # kept slice starts on a user turn (Anthropic needs user-first / strict
        # alternation; harmless for OpenAI-compatible providers).
        cut = len(convo) - self._keep_recent
        while 0 < cut < len(convo) and convo[cut].get("role") != "user":
            cut -= 1
        if cut <= 0:
            return  # no clean boundary -> skip this round

        old, recent = convo[:cut], convo[cut:]

        transcript = self._serialize_messages(old)
        if self._summary:
            # dedent the template (placeholders only), then fill it in -- dedent
            # on an f-string would mis-handle the multi-line interpolated values.
            instruction = dedent("""
                Existing summary:
                {summary}

                New messages to fold in:
                {transcript}

                Return a single updated summary.
            """).strip().format(summary=self._summary, transcript=transcript)
        else:
            instruction = f"Summarize this conversation:\n\n{transcript}"

        # Plain text side-call: no schema, no response_format.
        summary = await GetCompletion(
            provider=self._provider,
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (dedent(f"""
                        You compress conversations. Produce a concise summary that
                        preserves key facts, decisions, names, numbers, and open
                        questions. Be faithful and terse; output only the summary.
                    """).strip()),
                },
                {"role": "user", "content": instruction},
            ],
        )

        # Best effort: leave history untouched if the summary call fails.
        if summary.error or not summary.content:
            return

        # The side-call consumes tokens (counted) but isn't the live context size.
        if summary.total_tokens:
            self._consumed_tokens += summary.total_tokens

        self._summary = summary.content.strip()
        self._apply_system_role()              # refresh system role with the new summary
        self._history.replace_history(recent)  # drop folded turns, keep recent verbatim

    def _coerce_output(
        self, 
        content: str
    ):
        """
        Parse content into the output schema.
        Returns (instance, None) on success or (None, error_message) on failure.
        """
        try:
            return ModelConverter.json_to_model(self._output_schema, content), None
        except Exception as e:
            return None, str(e)

    async def run(
        self, 
        prompt: Content
    ) -> Union[Completion, Any]:
        """
        Executes the agent with a given prompt, handling history and optional schema parsing.

        When an output schema is set and the response fails validation, the agent shows the
        model its own output plus the validation error and asks it to fix it, retrying up to
        `schema_retries` times before falling back to the raw response.

        Args:
            prompt (Content): The user input content.

        Returns:
            Union[Completion, Any]: A Completion object containing the response, or a parsed object if an output schema is defined.
        """

        # Lazy init. Fill up _model_info if not already filled and perform a check
        await self._ensure_initialized()

        # Compact history into a running summary if near the context limit.
        await self._try_summarize()

        # Check if prompt has images, but images should not be provided.
        # Skip when modalities are unknown (empty) -> let the provider decide.
        if prompt.images and self._model_info.input_modalities and ("image" not in self._model_info.input_modalities):
            raise AgentError("This model does not support image inputs.")

        # Custom Output is set to true if output schema is provided
        custom_output = True if self._output_schema else False

        # Add the user message to history of the agent
        await self._history.add(
            role="user",
            content=prompt
        )

        # Construct optional parameters that must be passed to the provider,
        # based on modality or response format.
        optional = {}

        if custom_output:
            optional["response_format"] = { "type": "json_object" }

        if self._image_gen:
            optional["modalities"] = ["text", "image"]

        # Initial completion, then self-correct if the structured output fails
        # schema validation: hand the model its own invalid output and the
        # error, and retry up to self._schema_retries times.
        response = await self._complete(optional)

        parsed = None
        if custom_output and not response.error:
            parsed, err = self._coerce_output(response.content)

            attempt = 0
            while parsed is None and attempt < self._schema_retries:
                attempt += 1

                await self._history.add(
                    role="assistant",
                    content=Content(text=response.content or "")
                )
                await self._history.add(
                    role="user",
                    content=Content(
                        text=dedent(f"""
                        Your previous response could not be parsed into the required schema.

                        Error:
                        {err}

                        Generate a proper response that fixes all issues and strictly matches the
                        required JSON schema from the system instructions. Output only valid JSON,
                        with no extra commentary or code fences.
                        """).strip()
                    )
                )

                response = await self._complete(optional)
                if response.error:
                    break  # provider error mid-correction -> fall back below

                parsed, err = self._coerce_output(response.content)

        # Persist the final assistant turn, or wipe the working history (which
        # also discards the user message and any correction turns).
        if self._persistence:
            await self._history.add(
                role="assistant",
                content=Content(
                    text=response.content,
                    images=response.images
                )
            )
        else:
            self._history.clear_history()

        # Resolve structured output: return the parsed instance on success,
        # otherwise fall back to the current behavior (raise on a hard provider
        # error, else return the raw Completion below).
        if custom_output:
            if parsed is not None:
                return parsed
            if response.error:
                raise AgentError(f"Agent failed to generate response: {response.error}")

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