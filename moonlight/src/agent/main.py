from typing import Type, Union, Optional, Any, Dict
from textwrap import dedent
from datetime import date
from pydantic import BaseModel
from dataclasses import is_dataclass, dataclass

from .base import Content
from .history import AgentHistory, TypeAgentHistory
from ..helpers import ModelConverter
from ..helpers.web_search import web_search
from ..provider import (
    Provider,
    GetCompletion, 
    Completion,
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
        web_search (bool): If True, the agent may search the web (DuckDuckGo + Scrapy) when it
            needs external info, folding the results into the prompt before answering. It only
            searches when needed and reuses results already gathered. Works alongside output_schema.
        max_search_iterations (int): Max searches per run before answering (default 3).
        max_verify_iterations (int): After a grounded answer, max fact-check passes that drop
            claims the search results don't support (default 2). Set to 0 to disable.
        **kwargs: Sampling/config params forwarded to the provider (temperature, top_p, top_k,
            max_completion_tokens, frequency_penalty, presence_penalty, etc.).

    Automatically handles:
        - Model validation on first run (existence, modalities, token limits).
        - Conversation history and system-role management.
        - Structured output: JSON mode, schema injection, validation, and self-correction retries.
        - Multimodal image input and image generation.
        - Auto-summarization of old turns as the context window fills.
        - Optional web-search grounding before answering (DuckDuckGo + Scrapy).
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
        web_search: bool = False,
        max_search_iterations: int = 3,
        max_verify_iterations: int = 2,
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

        # Web-search grounding: when enabled, run() drives a search/answer loop
        # (up to max_search_iterations) instead of a single completion.
        self._web_search = web_search
        self._max_search_iterations = max_search_iterations

        # After a grounded answer, a fact-check loop (up to max_verify_iterations)
        # re-checks each claim against the search results and drops anything the
        # results do not actually support. Set to 0 to disable.
        self._max_verify_iterations = max_verify_iterations

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

        # Web search gathers grounding before the normal completion, so it works
        # with output_schema. It can't pair with image generation, though.
        if self._web_search and self._image_gen:
            raise AgentError("web_search cannot be combined with image_gen")

        if not isinstance(self._max_search_iterations, int) or self._max_search_iterations < 1:
            raise AgentError("max_search_iterations must be a positive integer")

        if not isinstance(self._max_verify_iterations, int) or self._max_verify_iterations < 0:
            raise AgentError("max_verify_iterations must be a non-negative integer")

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
        
        sys_role += dedent(f"""
        
        ---
        
        Today's date is {date.today().isoformat()}
        """)

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
        custom_params: dict = {},
        override_messages: Optional[TypeAgentHistory] = None
    ) -> Completion:
        # One provider call over the current history, with token accounting.
        response = await GetCompletion(
            provider=self._provider,
            model=self._model,
            messages=override_messages if override_messages else self.get_history(),
            **self._params,
            **custom_params
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
        summary = await self._complete(
            override_messages=[
                {
                    "role": "system",
                    "content": (dedent(f"""
                        You compress conversations. Produce a concise summary that
                        preserves key facts, decisions, names, numbers, and open
                        questions. Be faithful and terse; output only the summary.
                    """).strip()),
                },
                {"role": "user", "content": instruction},
            ]
        )
        
        # Best effort: leave history untouched if the summary call fails.
        if summary.error or not summary.content:
            return

        self._summary = summary.content.strip()
        self._apply_system_role()              # refresh system role with the new summary
        self._history._history.append(recent)
        
        # Account for the recent tokens
        # Rough estimation, will be accurate upon next request
        self._contextual_tokens += len(recent) // 4

    async def _search_context(
        self,
        prompt: Content
    ) -> str:
        """
        Gather web-search grounding for the prompt.

        Runs a search-only loop with a dedicated 'web research' system role over a
        copy of the conversation: the model proposes queries, web_search runs them,
        and the formatted results are accumulated until the model stops (or
        max_search_iterations is reached). run() folds the returned text into the
        prompt before the normal completion.
        """
        
        class _SearchStep(BaseModel):
            # One decision per search step: the next query, or null when done searching.
            search: Optional[str] = None

        # Copy the conversation but swap in the research system role.
        working = list(self.get_history())
        working[0] = {
            "role": "system",
            "content": dedent(f"""
                Today's date is {date.today().isoformat()}. Use it to judge what is current, and write
                time-aware queries (search the actual current year, not an old one).

                You decide whether a web search is needed to answer the user's request,
                and if so, what to search for.

                Reply ONLY with JSON matching this schema:
                {ModelConverter.model_to_schema(_SearchStep)}

                Set "search" to a focused query ONLY when the request needs current,
                external, or factual information that you do not already know or that has
                not already been gathered earlier in this conversation. If you can answer
                from your own knowledge or from results already gathered, set "search" to
                null. Do not search for things you already have. Before you stop, make sure
                the gathered results cover everything the request asks for, not just the main
                fact. If a needed part is still missing and likely findable, search for it.

                When you do search, write a clear, specific query about the actual subject,
                using concrete names and terms, not vague wording copied from the request.
                When the request asks for the latest or current information, query for that
                directly and prefer the most authoritative, up-to-date source.
            """).strip()
        }
        working.append({"role": "user", "content": prompt.text})

        chunks: list[str] = []
        for _ in range(self._max_search_iterations):
            response = await self._complete(
                override_messages=working
            )
            if response.error:
                break

            content = (response.content or "").strip()
            print(content)
            working.append({"role": "assistant", "content": content})
            
            try:
                step: _SearchStep = ModelConverter.json_to_model(_SearchStep, content)
            except:
                break
            
            if not step.search:
                break

            results = await web_search(step.search, max_results=5)
            chunks.append(results)
            working.append({
                "role": "user", 
                "content": dedent(f"""
                    ## Results for: {step.search}

                    ```text
                    {results}
                    ```
                    
                    ---
                    
                    If these results answer the request, set "search" to null. If they are
                    off-topic or incomplete, set "search" to a better, more specific query.
                """).strip()
            })

        return "\n\n".join(chunk for chunk in chunks if chunk)

    async def _verify_grounding(
        self,
        grounding: str,
        response: Completion
    ) -> Completion:
        """
        Fact-check a grounded answer against its sources, proving or dropping each claim.

        Runs a verify-and-revise loop: a strict checker compares the drafted answer to the
        web search results. For a plausible but unproven claim it may issue its own follow-up
        search, whose results are folded into the grounding and re-checked; a claim still
        unproven is removed. Repeats up to max_verify_iterations or until the answer is fully
        supported. Returns the (possibly revised) Completion, keeping the original if a
        verification step errors.
        """

        class _Verdict(BaseModel):
            # ok=True when every claim is backed by the sources. search holds a follow-up
            # query when a plausible claim needs proof; answer holds the current best answer
            # with unprovable claims removed.
            ok: bool
            answer: str
            search: Optional[str] = None

        answer = response.content or ""
        for _ in range(self._max_verify_iterations):
            messages = [
                {
                    "role": "system",
                    "content": dedent(f"""
                        You are a strict fact-checker. Check the drafted answer against the
                        search results only. A claim is allowed only when the results contain
                        hard proof of it. A loose or partial match (a similar name, a shared
                        username, a related but different topic) is not proof.

                        Reply ONLY with JSON matching this schema:
                        {ModelConverter.model_to_schema(_Verdict)}

                        If a claim is plausible but the results do not prove it, set "search"
                        to a query that would find that proof and keep the claim in "answer"
                        for now. If every claim is supported, set "ok" to true and "answer" to
                        the drafted answer unchanged. If a claim still has no proof after
                        searching, set "ok" to false and "answer" to a corrected version in
                        the same format with that claim removed. Do not add new claims.
                    """).strip()
                },
                {
                    "role": "user",
                    "content": dedent(f"""
                        ## Search results
                        ```text
                        {grounding}
                        ```

                        ## Drafted answer
                        ```text
                        {answer}
                        ```
                    """).strip()
                },
            ]
            
            check = await self._complete(
                override_messages=messages
            )
            
            if check.error:
                break

            try:
                verdict: _Verdict = ModelConverter.json_to_model(_Verdict, check.content)
            except:
                break

            answer = verdict.answer or answer

            # A plausible-but-unproven claim: gather proof and re-check, instead of
            # dropping it. The extra results never reach history (see run()'s trim).
            if verdict.search:
                results = await web_search(verdict.search, max_results=5)
                if results:
                    grounding = f"{grounding}\n\n{results}"
                continue

            if verdict.ok:
                break

        response.content = answer
        return response

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

        # Web-search grounding: gather results first, then fold them into the
        # prompt so the normal flow (schema, persistence, etc.) produces the answer.
        original_prompt = prompt
        grounded = False
        grounding = ""
        
        # Baseline token count for history before this turn's grounding, used to restore an
        # accurate context count after the bulky grounding is cropped out below.
        g_tokens = self._contextual_tokens
        
        if self._web_search:
            grounding = await self._search_context(prompt)
            if grounding:
                grounded = True
                today = date.today().isoformat()
                prompt = Content(
                    text=dedent(f"""
                        {prompt.text} 
                        
                        ---
                        
                        Today's date is {today}. Base your answer on the web search
                        results below. When sources disagree, trust the most recent and
                        most official one, and if several values appear (for example
                        multiple version numbers or dates) treat the newest as current
                        rather than reporting an older value as the latest. Share a claim
                        only when the results contain hard proof of it. No hard proof
                        means do not share it. A loose or partial match (a similar name, a
                        shared username, a related but different topic) is not proof, so do
                        not present it as fact. When the results do not actually answer the
                        request, reply with one brief sentence that you could not confirm the
                        information, and nothing more. Do not narrate your searches, list the
                        sources, or mention unrelated results you came across.
                        
                        ## Grounding:
                        ```text
                        {grounding}
                        ```
                        """),
                    images=prompt.images,
                )

        # Custom Output is set to true if output schema is provided
        custom_output = True if self._output_schema else False

        # Add the user message to history of the agent
        await self._history.add(
            role="user",
            content=prompt
        )

        # Construct optional parameters that must be passed to the provider,
        # based on modality or response format.
        custom_params = {}

        if custom_output:
            custom_params["response_format"] = { "type": "json_object" }

        if self._image_gen:
            custom_params["modalities"] = ["text", "image"]
        
        # Initial completion, then self-correct if the structured output fails
        # schema validation: hand the model its own invalid output and the
        # error, and retry up to self._schema_retries times.
        response = await self._complete(custom_params)

        # Fact-check the grounded answer against its sources, dropping unsupported claims.
        if grounded and self._max_verify_iterations and not response.error:
            response = await self._verify_grounding(grounding, response)

        # For a grounded run, drop the bulky search results from the stored user message and
        # keep only the original question plus a marker, to save context on later turns.
        if grounded:
            note = dedent(f"""
                {original_prompt.text}

                ---
                [Web research was used to answer this. Raw results omitted from history to save context.]
            """).strip()

            # content is a list of parts when the prompt had images (text part first),
            # otherwise a plain string.
            last = self._history._history[-1]["content"]
            if isinstance(last, list):
                self._history._history[-1]["content"][0]["text"] = note
            else:
                self._history._history[-1]["content"] = note

            # The token count was set from the call that still had the full grounding,
            # which we just cropped out of history. Rebuild it from the pre-grounding
            # baseline (g_tokens, the provider's accurate count for prior history) plus a
            # rough estimate of just the small cropped note (~4 chars/token). This keeps
            # the estimate scoped to the note instead of re-guessing the whole history,
            # and the next turn's call recomputes it exactly.
            self._contextual_tokens = g_tokens + (len(note) // 4)

        parsed = None
        if custom_output and not response.error:
            parsed = None
            err = None
            
            try:
                parsed = ModelConverter.json_to_model(self._output_schema, response.content)
            except Exception as e:
                err = e

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

                response = await self._complete(custom_params)
                if response.error:
                    break  # provider error mid-correction -> fall back below

                parsed = ModelConverter.json_to_model(self._output_schema, response.content)

        # Persist the final assistant turn, or wipe the working history (which also
        # discards the user message and any correction turns)
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