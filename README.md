<p align="left">
  <img src="https://github.com/user-attachments/assets/e3ccd296-65d6-4774-90cb-8ecda6714763" width="500" alt="MoonLight">
</p>

> **Minimal async AI agent framework with zero bloat**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PyPI version](https://img.shields.io/pypi/v/moonlight-ai.svg)](https://pypi.org/project/moonlight-ai/)

Moonlight is a lightweight SDK for building AI agents with full control. It provides async stateful agents, multimodal input/output (text, images, vision), image generation, structured responses via Pydantic/dataclass, and works with any OpenAI-compatible provider. No dependencies on OpenAI libraries, no hidden abstractions, no framework bloat.

## Installation

```bash
uv pip install moonlight-ai
```

## Quick Start

```python
import asyncio
from moonlight import Provider, Agent, Content

# Configure provider
provider = Provider(
    source="openrouter",  # or "openai", "deepseek", custom URL
    api="your-api-key"
)

# Create agent
analysis_agent = Agent(
    provider=provider,
    model="qwen/qwen3-4b:free",
    system_role="You are a data analyst"
)

# Analyze some data
data = """
Q1 Sales: $125k, Q2 Sales: $157k, Q3 Sales: $198k, Q4 Sales: $223k
Top product: Widget A (45% revenue), Customer satisfaction: 4.2/5
"""

prompt = Content(f"Analyze this business data and provide key insights:\n{data}")

# Run async
response = asyncio.run(analysis_agent.run(prompt))
print(response.content)
# Output: The business shows strong growth momentum with 78% increase from Q1 to Q4...
```

## Core Features

### Structured Output

Return type-safe responses using Pydantic models or dataclasses:

```python
from pydantic import BaseModel
from typing import List
from enum import Enum

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class Entity(BaseModel):
    name: str
    type: str  # person, organization, location, etc.
    mentions: int

class Analysis(BaseModel):
    sentiment: Sentiment
    confidence: float
    key_topics: List[str]
    entities: List[Entity]
    summary: str

sentiment_agent = Agent(
    provider=provider,
    model="google/gemini-3-flash-preview",
    output_schema=Analysis  # Automatic JSON mode + validation
)

text = """
Apple Inc. announced record quarterly earnings today, with CEO Tim Cook 
praising the team's innovation. The iPhone 15 sales exceeded expectations 
in Asian markets, particularly China and India.
"""

result: Analysis = asyncio.run(sentiment_agent.run(Content(f"Analyze this text:\n{text}")))
print(result.sentiment)
# Output: Sentiment.POSITIVE

print(result.confidence)
# Output: 0.92

print(result.entities[0].name)
# Output: Apple Inc.

print(result.summary)
# Output: Apple reports strong earnings driven by iPhone 15 success in Asia
```

The SDK automatically:

- Enables JSON mode on the provider
- Injects schema into system prompt
- Validates and parses response into your model
- Handles nested structures and optional fields

### Multimodal Input

Send images alongside text (URLs, local files, or base64):

```python
response = asyncio.run(agent.run(
    Content(
        text="What's in these images?",
        images=[
            "https://example.com/image.jpg",  # URL
            "/path/to/local/image.png",        # Local file
            "data:image/jpeg;base64,..."       # Base64
        ]
    )
))
print(response.content)
# Output: The first image shows a sunset over mountains...
```

Images are automatically:

- Downloaded from URLs (async)
- Read from disk with proper MIME types
- Converted to base64 data URIs
- Validated and filtered

### Image Generation

Generate images directly from text prompts using multimodal models:

```python
import asyncio
import base64

image_agent = Agent(
    provider=provider,
    model="google/gemini-3-flash-preview",  # or other image-capable models
    image_gen=True  # Enable image generation mode
)

# Run the image generation prompt
response = asyncio.run(image_agent.run(
    Content("Create a serene mountain landscape at sunset with a lake reflection")
))

# This prints the text description or caption returned by the model (if any)
print(response.content)
# Example output: "A calm mountain lake at sunset with orange and purple skies..."

# If the model returned images, they will be in response.images as base64 data URLs
if response.images:
    for i, img_url in enumerate(response.images):

        # This prints the raw base64 data URL (useful for debugging or logging)
        print(f"Generated image (base64 data URL): {img_url}")

        # Image is in "data:image/...;base64,XXXX" format.
        # strip the header and keep only the base64 part
        img_base64 = img_url.split(",", 1)[1]

        # Decode base64 into raw image bytes
        img_bytes = base64.b64decode(img_base64)

        # Save the decoded image bytes to a PNG file on disk
        out_path = f"generated_{i}.png"
        with open(out_path, "wb") as f:
            f.write(img_bytes)

        # This prints where the image was saved on disk
        print(f"Saved image to {out_path}")

```

The SDK automatically:

- Validates model supports image output
- Enables multimodal mode (`["text", "image"]`)
- Returns base64-encoded images in response
- Prevents using `image_gen` with `output_schema` (incompatible)

**Note**: Image generation and structured output schemas are mutually exclusive.

### Conversation History

Agents maintain stateful conversation history:

```python
agent = Agent(provider=provider, model="gpt-5.2")

# First turn
asyncio.run(agent.run(Content("My name is Alice")))

# Second turn (agent remembers context)
response = asyncio.run(agent.run(Content("What's my name?")))
print(response.content)
# Output: Your name is Alice

# Clear history
agent.clear()

# Update system role mid-conversation
agent.update_system_role("You are now a pirate")
```

### Provider Support

Works with any OpenAI-compatible API:

```python
# Built-in providers
Provider(source="openai", api="sk-...")
Provider(source="deepseek", api="sk-...")
Provider(source="openrouter", api="sk-...")
Provider(source="together", api="...")
Provider(source="groq", api="gsk-...")
Provider(source="google", api="...")

# Custom endpoints
Provider(source="http://localhost:11434/v1", api="ollama")
Provider(source="https://api.custom.com/v1", api="key")
```

Supported providers: OpenAI, DeepSeek, Together, Groq, Google AI, HuggingFace, OpenRouter, or any custom OpenAI-compatible endpoint.

### Model Validation

Agents automatically validate model capabilities on initialization:

```python
agent = Agent(
    provider=provider,
    model="gpt-4o",              # Checks if model exists in given provider
    max_completion_tokens=8192,  # Validates against model limits
    image_gen=True               # Validates against model limits
)

# Automatically checks:
# - Endpoint compatibility
# - Model exists in provider
# - Context length and max_completion_tokens
# - Input modalities (text, image, audio, video)
# - Output modalities (text, image)
# - Reasoning capability support
```

Validation prevents runtime errors by checking:

- **Endpoint Compatibility:** Checks whether the Provider source has the necessary routes to be compatible
- **Model existence**: Ensures the model is available from the provider
- **Token limits**: Validates `max_completion_tokens` doesn't exceed model capacity
- **Modality support**: Verifies model supports requested input/output types (images, video, etc.)
- **Image generation**: Confirms model can generate images when `image_gen=True`

Errors are raised immediately during agent initialization with clear messages:

```python
try:
    agent = Agent(
        provider=provider,
        model="qwen-4b",
        image_gen=True
    )
except AgentError as e:
    print(e)
    # Output: This model does not support image generation
```

### Token Tracking

Agents track token usage automatically:

```python
agent = Agent(provider=provider, model="gpt-4o")
asyncio.run(agent.run(Content("Hello")))

print(agent.get_total_tokens())
# Output: 156 (total tokens used)
```

### Error Handling

Detailed error messages from providers with intelligent parsing:

```python
response = asyncio.run(agent.run(Content("...")))

if response.error:
    print(f"Error: {response.error}")
    # Output: Error: Rate limited - too many requests
else:
    print(response.content)
    # Output: [normal response content]
```

Handles:

- **Invalid credentials (401)**: Expired OAuth tokens, invalid API keys
- **Rate limits (429)**: Too many requests, retry guidance
- **Content moderation (403)**: Specific reasons for flagged content
- **Parameter errors (400)**: Detailed validation messages (e.g., "max_tokens exceeds limit")
- **Insufficient credits (402)**: Clear payment/billing errors
- **Timeout errors (408)**: Request took too long
- **Provider errors (502)**: Model down or invalid response
- **Routing errors (503)**: No provider meets requirements
- **Provider-specific errors**: Parses nested JSON error structures

Error messages include context from provider metadata when available, making debugging easier.

## Architecture

```
moonlight/
└── src/
    ├── agent/
    │   ├── base.py             # Content dataclass
    │   ├── history.py          # AgentHistory (conversation + image processing)
    │   └── main.py             # Agent class
    ├── provider/
    │   ├── main.py             # Provider class
    │   └── completion.py       # GetCompletion (async API calls)
    └── helpers/
        └── model_converter.py  # Schema/model conversion utilities
```

## Design Philosophy

Moonlight is intentionally minimal:

- **No framework lock-in**: Standard Python async, bring your own orchestration
- **No hidden magic**: Direct API calls, explicit control flow
- **No bloat**: Zero dependencies on OpenAI SDK or heavy frameworks
- **Full control**: Access raw responses, customize at any level
- **Provider agnostic**: Works with any OpenAI-compatible API

## What Moonlight Doesn't Do

To stay lightweight, Moonlight does not include:

- Multi-agent orchestration (build your own with asyncio)
- RAG systems or vector databases
- Web scraping or search
- Tool calling (in development)
- Streaming responses
- Built-in retry logic (in development)
- Observability or logging (in development)
- Audio/video output (in development)
- MCP (Model Context Protocol) integration (in development)

These are left to you or future extensions to keep the core minimal.

## Advanced Configuration

```python
agent = Agent(
    provider=provider,
    model="gpt-4o",
    system_role="You are an expert analyst",
    output_schema=MyModel,   # Optional structured output
    image_gen=False,         # Enable image generation (conflicts with output_schema)
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    max_completion_tokens=2048,
    frequency_penalty=0.5,
    presence_penalty=0.5,
    repetition_penalty=1.1
)

# Access history
messages = agent.get_history()

# Token usage
tokens = agent.get_total_tokens()
```

**Supported Parameters:**

- `temperature`, `top_p`, `top_k`: Sampling parameters
- `max_completion_tokens`, `max_output_tokens`: Token limits
- `frequency_penalty`, `presence_penalty`, `repetition_penalty`: Repetition control
- `tools`, `tool_choice`: Tool calling configuration (planned)
- `plugins`: Provider-specific plugins
- `reasoning`, `verbosity`: Control reasoning traces

## Building From Source

```bash
# Clone repo
git clone https://github.com/ecstra/moonlight.git
cd moonlight

# Build distribution
pip install build twine
python -m build

# Install locally
pip install dist/moonlight_ai-0.2.0-py3-none-any.whl

# Test
python -c "from moonlight import Agent; print('OK')"
```

## Roadmap

- [ ] Retry logic for API calls
- [ ] Audio and video output support
- [ ] Sequential and parallel agent execution engines with data sharing
- [ ] Tool calling support (evaluating usefulness)
- [ ] MCP (Model Context Protocol) integration
- [ ] Logging and observability framework
- [ ] RAG and web search (evaluating necessity)

## License

MIT License - use freely in personal and commercial projects.
