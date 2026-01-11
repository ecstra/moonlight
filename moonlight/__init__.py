__version__ = "0.2.4"
__author__ = "ecstra"
__description__ = "Lightweight AI Agents SDK for building intelligent automation systems"

# Core Agent Architecture
from .src.agent import Agent, Content
from .src.provider import Provider

__all__ = [
    # Core Components
    "Agent",
    "Content",
    "Provider",

    # Metadata
    "__version__",
    "__author__",
    "__description__",
]