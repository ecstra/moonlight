##############################################################################################
# THIS IS A STRIPPED DOWN VERSION SPECIFICALLY FOR TEXT-BASED AGENTS
#
# CODE EXECUTION IS STRIPPED DOWN
# MCP CLIENTS ARE STRIPPED DOWN
# IMAGES ARE STRIPPED DOWN
# AGENT CALLING ANOTHER AGENT IS STRIPPED DOWN
# DYNAMIC MULTI AGENT DISCOVERY IS STRIPPED DOWN
# DYNAMIC MULTI AGENT ORCHESTRATION IS STRIPPED DOWN
# DEEP RESEARCH IS STRIPPED DOWN
# WORKFLOW IS STRIPPED DOWN
#
# THIS IS NOT THE FULL VERSION OF THE HIVE AGENT ARCHITECTURE
# DOWNLOAD THE FULL VERSION FOR CODE EXECUTION
##############################################################################################

import logging

from rich.console import Console
from rich.markdown import Markdown

from ...src.json_parser import parse_json
from ...core.token import TokenCounter

from .base import MoonlightProvider

console = Console()
logger = logging.getLogger(__name__)

class AgentException(Exception):
    pass

token_counter = TokenCounter(model="deepseek")

class AgentHistory:
    global token_counter
    
    def __init__(
            self,
            max_length: int = 4096,
            system_role: str = "",
        ):
        # History
        self.history = []

        # Max Length
        self.max_length = max_length
        
        # Initialize the token counter and total tokens
        self.token_counter = token_counter
        
        # Total tokens in the convo
        self.conversation_tokens = 0
        
        # Tokens of the system role in case of updation
        self.system_role_tokens = 0
        
        # Tokens of output
        self.output_tokens = 0
        
        # Set the system role
        self._set_system_role(system_role)

    def _set_system_role(
            self,
            system_role: str = "",
        ):
        self.system_role = system_role
        
        # Calculate the token count of the system role
        self.system_role_tokens = self.token_counter.count_tokens(system_role)
        self.conversation_tokens += self.system_role_tokens

        if len(self.history) > 1:
            self.history.insert(0, {"role": "system", "content": system_role})
        else:
            self.history = [
                {"role": "system", "content": system_role}
            ]
    
    def update_system_role(
            self, 
            system_role: str = "",
        ):
        self.system_role = system_role
        
        # Calculate the token count of the new system role
        new_system_role_tokens = self.token_counter.count_tokens(system_role)
        self.conversation_tokens += new_system_role_tokens - self.system_role_tokens
        self.system_role_tokens = new_system_role_tokens
        
        # Update the system role in the history
        self.history.pop(0)
        self.history.insert(0, {"role": "system", "content": system_role})
        
    def append_message(
            self, 
            role: str = "", 
            message: str = "",
        ):

        # Append Images
        # if self.multimodal and images:
        #     message = [{"type": "text", "text": message}]
        #     for i, image_b64 in enumerate(images):
        #         message.append({"type": "text", "text": f"Image #{i}"})
        #         message.append({"type": "image_url", "image_url": {"url": image_b64}})
        
        # Add the count normally to total_token_pool
        token_count = self.token_counter.count_tokens(message)
        
        self.conversation_tokens += token_count
        
        if role == "assistant":
            self.output_tokens = token_count
            
        # Append text message
        self.history.append({"role": role, "content": message})
        
    def clear_history(self):
        self.history = []
        self._set_system_role(self.system_role)
    
    def __repr__(self):
        return f"AgentHistory(max_length={self.max_length}, check_length={self.check_length})"
    
    def __str__(self):
        out_str = "=" * 20 + "\n"
        out_str += f"Conversation History: \n{self.history}"
        out_str += "\n" + "=" * 20
        return out_str

class AgentResponse:
    def __init__(
            self, 
            response: dict = {}
        ):
        self.raw_message = response.get('raw_message', '')
        self.assistant_message = response.get('assistant_message', '')
        self.reason = response.get('reason', '')
       
    def __repr__(self):
        console.rule("[bold green]Agent Response Details[/bold green]")
        
        # Render Reason with Markdown and print label separately
        console.print("[bold yellow]Reason:[/bold yellow]")
        md_reason = Markdown(self.reason)
        console.print(md_reason)
        console.print("")  # for spacing
        
        # Render Assistant Message with Markdown and print label separately
        console.print("[bold yellow]Assistant Message:[/bold yellow]")
        md_assistant = Markdown(str(self.assistant_message))
        console.print(md_assistant)
        console.print("")  # for spacing
        
        console.rule()
        return ""
    
class Agent:
    def __init__(
            self,
            
            # About the Agent
            name: str = "Agent",
            description: str = "",

            # Model Parameters
            model_name: str = "",
            max_context: int = 64_000,
            max_output_length: int = 8192,
            temperature: float = 0.7,

            # Instructions
            instruction: str = "",
            
            # Agent Provider/URL
            provider: MoonlightProvider = None,

            # Agent Parameters
            enable_history: bool = True,

            # JSON Mode
            json_mode: bool = False,
            json_schema: str = ""
        ):
        
        # About the Agent
        self.name                   =      name
        self.description            =      description

        # Model Parameters
        self.model_name             =      model_name
        self.max_context            =      max_context
        self.max_output_length      =      max_output_length
        self.temperature            =      temperature

        # Instructions
        self.instruction            =      instruction
        self.role                   =      ""

        # Agent Provider
        self.provider               =      provider
        
        # Agent Parameters
        self.enable_history         =      enable_history
        
        # JSON Mode
        self.json_mode              =      json_mode
        self.json_schema            =      json_schema
        
        # Token Counter
        self.total_tokens           =      0
        
        # Initialize Agent
        self._init_agent()
    
    def _init_agent(self):
        # Check the parameters of the agent
        self._check_parameters()

        # Build the role of the agent
        self.role = self.instruction
        
        # Initialize the Agent History
        self.history = AgentHistory(
            system_role=self.role, 
            max_length=self.max_context, 
        )
            
    def _check_parameters(self):  

        if not isinstance(self.provider, MoonlightProvider):
            raise AgentException("Provider must be an instance of MoonlightProvider.")
        
        if len(self.instruction) == 0:
            logger.warning("System role is not set. Please set the system role. Proceeding with default role.")
            self.instruction = "You are an Agent with a secret mission."
        
        if len(self.model_name) == 0:
           raise AgentException("Found no model. Please pass one!")
        
        if self.json_schema and not self.json_mode:
            raise AgentException("Please enable json_mode to use json_schema.")
        
        if self.json_mode and not self.json_schema:
            raise AgentException("Please provide a json schema to use json_mode.")
        
        if self.json_mode and self.json_schema:            
            try:
                parse_json(
                    self.json_schema,
                    fix_mode=False
                )
            except Exception as e:
                raise AgentException(f"Invalid JSON schema. {e}")
            
            self.instruction += f"You are only allowed to provide JSON data. Anything else will be ignored. Here is the schema:\n \n```\njson\n{self.json_schema}\n```\n. Each json object must be seperated by a comma. Make sure it follows proper JSON format. You are not allowed to return anything but JSON data within the schema and in ```json ... ``` format (markdown json format). You can return either top level object or array of objects. Your wish depending on the schema. Do not return anything else."

    def _update_token_count(self):
        """
        Add total tokens to agents tokens
        """
        self.total_tokens += self.history.conversation_tokens
    
    def refresh_role(self):
        self.role = self.instruction
        self.history.update_system_role(self.role)       
    
    def get_history(self):
        return self.history.history
    
    def get_tokens(self):
        """
        Get the total tokens used in the conversation history.
        """
        return self.total_tokens
    
    def clear(self):
        self.history.clear_history()
        self.total_tokens = 0
        
    def __repr__(self):
        return f"Agent(name={self.name}, description={self.description})"
    
    def __str__(self):
        out_str = "=" * 20 + "\n"
        out_str += f"Agent: {self.name}\n\nModel: {self.model_name}\nMax Context: {self.max_context}, \nMax Output Length: {self.max_output_length} \nProvider: {self.provider}\nTemperature: {self.temperature}\n\nDescription: {self.description},\n\nSystem Role: {self.role}"
        out_str += "\n" + "=" * 20
        return out_str