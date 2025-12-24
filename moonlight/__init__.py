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
# WHATS ADDED:
# - Async RAG System (Needs custom hosting)
#
# WHY:
# It is designed to be lightweight and easy to use for basic agent orchestration tasks.
# Shifting this project towards being an AgentsSDK used for building systems as compared to Agents as a Platform.
# This makes it easier to build custom pipelines as per the use-case.
#
# FEATURES:
# - Agent Model + Runner
# - Aggregated Provider (Multiple Options, including local)
# - In-Build Agent history with role updation/editing
# - Custom Response structure (Json Mode)
# - Custom RAG System (Must provide hosted QDRANT URL)
# - Async Web search
##############################################################################################

__version__ = "0.1.10"
__author__ = "ecstra"
__description__ = "Advanced Multi-Agent AI Orchestration Framework"

import os
os.environ['TOKENIZERS_PARALLELISM'] = "false"

# Core Agent Architecture
from .core.agent_architecture import MoonlightProvider, Agent, Runner
from .core.functionality import WebSearch
from .core.processors import FileProcessor

__all__ = [
    # Core Components
    "MoonlightProvider",
    "Agent", 
    "Runner",
    
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