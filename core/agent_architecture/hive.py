##############################################################################################
# THIS IS A STRIPPED DOWN VERSION SPECIFICALLY FOR TEXT-BASED AGENTS
#
# CODE EXECUTION IS STRIPPED DOWN
# MCP CLIENTS ARE STRIPPED DOWN
# IMAGES ARE STRIPPED DOWN
# AGENT CALLING ANOTHER AGENT IS STRIPPED DOWN
#
# THIS IS NOT THE FULL VERSION OF THE HIVE AGENT ARCHITECTURE
# DOWNLOAD THE FULL VERSION FOR CODE EXECUTION
##############################################################################################

import logging, re
from textwrap import dedent

from .agent import Agent, AgentResponse

from ...core.providers.llm_provider import CompletionInput, OpenAIProvider
from ...src.json_parser import parse_json

logger = logging.getLogger("hive")

# Hive is the main class that runs the agent architecture.
class Hive:
    def __init__(
            self,
            agent: Agent,
        ):
        
        # Agent
        self.agent = agent
        
        # LLM Provider
        self.llm = OpenAIProvider(
            provider=agent.provider,
        )
        
    def _extract_reason(
            self, 
            text: str
        ):
        """
        Extract the reason from the text.
        """
        reason_match = re.search(r'<(thought|think)>(.*?)</(thought|think)>', text, re.DOTALL)
        if reason_match:
            full_reason = reason_match.group(2).strip()  # Changed from group(1) to group(2)
            # Remove the plan section from reason if it exists
            reason_without_plan = re.sub(r'<plan>.*?</plan>', '', full_reason, flags=re.DOTALL)
            return reason_without_plan.strip()
        return ""

    def _extract_answer(
            self, 
            text: str
        ):
        """
        Extract the answer from the text.
        """
        answer_match = re.search(r"<answer>(.*?)(</answer>|$)", text, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        return ""
                  
    def _normal_run(
            self, 
            message: str = ""
        ):
        """
        The main function that runs the agents.
        
        Args:
            message: The message to send to the agent.
            images: A list of images to send to the agent.
        
        Returns:
            AgentResponse: The response from the agent.
        """
        if not self.agent.enable_history:
            self.agent.history.clear_history()

        # DISABLE IMAGES
        # if len(images) > 1 and self.agent.provider == "together":
        #     log_error("Together provider does not support multiple images. Only the first image will be used.")
        #     images = images[:1]
        
        # if message and not images:
        #     self.agent.history.append_message("user", message)
        # elif images and message:
        #     self.agent.history.append_message("user", message=message, images=images)
        # elif images and not message:
        #     self.agent.history.append_message("user", message="images", images=images)
        
        self.agent.history.append_message(
            role="user", 
            message=message
        )

        raw_message = self.llm.get_completion(
            CompletionInput(
                model_name = self.agent.model_name,
                messages = self.agent.get_history(),
                max_context = self.agent.max_context,
                max_output_length = self.agent.max_output_length,
                temperature = self.agent.temperature
            )
        )
            
        # Extract response content with better error handling
        assistant_message = self._extract_answer(raw_message)

        # Append the message to agents history
        self.agent.history.append_message(
            role="assistant", 
            message=assistant_message
        )

        # update agents total tokens
        self.agent._update_token_count()
        
        return AgentResponse({
            "raw_message": raw_message,
            "assistant_message": assistant_message,
            "reason": self._extract_reason(raw_message),
        })
    
    def _json_run(
            self, 
            message: str = "", 
    ):
        """
        Run the agent. Then convert the response of assistant to JSON format using custom JSON Parser.
        """
        message += "\n Return the output in the JSON format only."
        
        logger.debug("=" * 100)
        logger.debug("Running in JSON mode")
        logger.debug("=" * 100)
        
        response = self._normal_run(message)
        
        if response == "<out_of_tokens>": 
            return [{"error": "<out_of_tokens>"}]
        
        # Convert the response to JSON format
        response.assistant_message = re.sub(r'```json\s*(.*?)\s*```', r'\1', response.assistant_message, flags=re.DOTALL)
        response.assistant_message = re.sub(r'```', '', response.assistant_message, flags=re.DOTALL)
        response.assistant_message = response.assistant_message.strip()
        response.assistant_message, _ = parse_json(dedent(response.assistant_message).strip())
        
        return response
    
    def run(
            self, 
            message: str = ""
        ):
        
        """
        Run the agent with the given message and images.
        
        Args:
            message: The message to send to the agent.
            images: A list of images to send to the agent.
        
        Returns:
            AgentResponse: The response from the agent.
        """
        try:             
            if self.agent.json_mode:
                return self._json_run(message)
            else:
                return self._normal_run(message)                
        except Exception as e:
            message = f"Output could not be generated due to an error. {e}"
            
            if self.agent.json_mode:
                message = [{"error": message}]
            
            return AgentResponse({
                "raw_message": "",
                "assistant_message": f"Output could not be generated due to an error. {e}",
                "reason": "",
            })