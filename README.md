<p align="left">
  <img src="https://github.com/user-attachments/assets/e3ccd296-65d6-4774-90cb-8ecda6714763" width="500" alt="MoonLight">
</p>

> **Lightweight AI Agents SDK for building intelligent automation systems**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PyPI version](https://img.shields.io/pypi/v/moonlight-ai.svg)](https://pypi.org/project/moonlight-ai/)

A minimal SDK for building agent-based systems and RAG pipelines. Moonlight provides composable primitives without prescribing orchestration or execution models. It is intentionally small, designed to be understood, extended, and integrated into larger systems rather than serving as a framework.

## Installation

```bash
uv pip install moonlight-ai
```

---

## Design Philosophy

Moonlight is built on minimalism and composability. Each component does one thing well: agents manage conversation and execution, runners handle invocation, RAG provides retrieval, and processors handle data. There are no hidden abstractions or magic workflows. You build your own coordination logic using standard Python patterns.

## Core Concepts

**Agent**: Stateful conversation manager with configurable LLM backend, history tracking, and structured output support.

**Runner**: Simple executor that invokes an agent with a prompt and returns structured responses.

**RAG System**: Async document storage and retrieval using Qdrant for vector search with configurable chunking and reranking.

**File Processors**: Extract text from documents (PDF, DOCX, XLSX, CSV, TXT, PPTX, SQLite).

**Web Search**: Async web scraping and content extraction for research tasks.

## Minimal Example

```python
from moonlight import Agent, Runner, MoonlightProvider, RAGSystem

# Basic agent
provider = MoonlightProvider(
	provider_name="openrouter",
	api_key="OPENROUTER_API_KEY"
)
agent = Agent(
    name="analyst",
    instruction="You are a research analyst.",
    provider=provider,
    model_name="mistralai/devstral-2512:free"
)

response = Runner(agent).run("Summarize recent AI trends")
print(response.assistant_message)

# With RAG
rag = RAGSystem(namespace="research", qdrant_url="http://localhost:6333")
rag.add_document("report.pdf")
results = rag.query("What are the key findings?")

# With structured output
schema = """
{
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "confidence": {"type": "number"}
    }
}
"""

structured_agent = Agent(
    name="analyzer",
    instruction="Extract structured insights.",
    provider=provider,
    model_name="gpt-4",
    json_mode=True,
    json_schema=schema
)

result = Runner(structured_agent).run("Analyze this data...")
```

## Architecture

```
moonlight/
├── core/
│   ├── agent_architecture/    # Agent, Runner, MoonlightProvider
│   ├── functionality/         # RAGSystem, WebSearch
│   ├── processors/            # FileProcessor, ChunkProcessor
│   ├── providers/             # LLM and RAG provider interfaces
│   └── token/                 # Token counting utilities
└── src/
    └── json_parser/           # JSON parsing for structured outputs
```

## RAG Setup

Moonlight uses Qdrant for vector storage. Start a local instance:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Configure RAG with namespace isolation:

```python
rag = RAGSystem(
    namespace="my_project",
    qdrant_url="http://localhost:6333",
    chunk_min_size=800,
    chunk_max_size=1200
)

rag.add_document("document.pdf")
rag.add_text("Custom knowledge...")

results = rag.query("Your question here")
```

## Provider Support

Moonlight supports OpenAI-compatible providers through a unified interface:

```python
# OpenAI
provider = MoonlightProvider(provider_name="openai", api_key="...")

# DeepSeek
provider = MoonlightProvider(provider_name="deepseek", api_key="...")

# Custom endpoint
provider = MoonlightProvider(provider_url="https://api.example.com/v1", api_key="...")
```

Supported providers: OpenAI, DeepSeek, Together, Groq, Google AI, HuggingFace, GitHub Models, OpenRouter.

## Building Locally

Before publishing, build and test the package:

```bash
# Build distribution
pip install build twine
python -m build

# Install locally
pip install dist/moonlight-0.2.0-py3-none-any.whl

# Test
python -c "from moonlight import Agent, Runner; print('OK')"

# Publish (after testing)
python -m twine upload --repository testpypi dist/*
python -m twine upload dist/*
```

## Limitations

Moonlight does not provide:

- Multi-agent orchestration or workflows
- Built-in code execution or sandboxing
- Prompt optimization or caching
- Model fine-tuning or training
- Production monitoring or observability
- Authentication or rate limiting

These are intentionally excluded to keep the scope minimal. Build your own orchestration using standard async patterns and task queues.

## Status

Personal project, actively maintained. Designed for learning and prototyping agent systems with production-aware patterns. Suitable for integration into larger systems but not a complete application framework.

## License

MIT License - see LICENSE file for details.
