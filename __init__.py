##############################################################################################
# Moonlight SDK - Agents SDK
#
# WHATS STRIPPED DOWN FROM V0:
# - Agent calling another agent is stripped down
# - Deep Research is stripped down
# - Workflow is stripped down
# - Code execution is stripped down
# - Shell execution is stripped down
# - Orhcestration is stripped down
#
# WHAT IS NOT STRIPPED DOWN:
# - Agent Architecture
# - LLM Provider
# - Token Counter
# - JSON Parser
# - Agent History
# - Agent Response
#
# WHATS ADDED:
# - Async RAG System (Needs custom hosting)
#
# This is a stripped down version of the Moonlight SDK.
# It is designed to be lightweight and easy to use for basic agent orchestration tasks.
# Shifting this project towards being an AgentsSDK used for building systems as compared to Agents as a Platform.
##############################################################################################

__version__ = "internal"
__author__ = "ecstra"
__description__ = "Advanced Multi-Agent AI Orchestration Framework"

import os
os.environ['TOKENIZERS_PARALLELISM'] = "false"

# Core Agent Architecture
from .core.agent_architecture import MoonlightProvider, Agent, Hive
from .core.functionality import WebSearch
from .core.processors import FileProcessor
from .utils import AsyncTimer

__all__ = [
    # Core Components
    "MoonlightProvider",
    "Agent", 
    "Hive",
    
    # RAG and Web Search
    "RAGSystem",
    "WebSearch",
    
    # Processors
    "FileProcessor",

    # Metadata
    "__version__",
    "__author__",
    "__description__",
]