
<p align="left">
  <img src="https://github.com/user-attachments/assets/e3ccd296-65d6-4774-90cb-8ecda6714763" width="500" alt="MoonLight">
</p>

> **A powerful Multi-Agent AI Framework for intelligent automation and orchestration**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![MCP Servers](https://img.shields.io/badge/MCP%20Servers-3000%2B-green.svg)](https://github.com/modelcontextprotocol/servers)

Moonlight transforms how you build AI applications by providing a comprehensive framework that orchestrates multiple AI agents, automates complex workflows, and integrates with thousands of tools seamlessly.

---

## 🚀 What Makes Moonlight Special?

**🏗️ Multiple Execution Modes**

- **Hive** - Single agent with advanced tooling
- **Orchestra** - Multi-agent collaboration
- **DeepResearch** - Recursive web research
- **Workflow** - Trigger-based automation

**🔌 Massive Tool Ecosystem**

- 3,000+ MCP servers for databases, APIs, and services
- Built-in web search, code execution, and file operations
- Support for all major AI providers

**⚡ Production Ready**

- Token optimization and context management
- Multi-modal support (text + images)
- Comprehensive error handling and logging

---

## ⚡ Quick Start

### Installation

```bash
# Install
uv pip install moonlight-ai

# Install Playwright (NEEDED)
playwright install
```

> Note: Install NPX and UVX on your machine to get the most out of MCP Compatibility.

---

## 💡 Usage Examples

### 🏠 Single Agent (Hive) - Content Creation Assistant

```python
import os
from moonlight_ai import Agent, Hive, MoonlightProvider
from dotenv import load_dotenv

load_dotenv()

# Create a content creation provider
provider = MoonlightProvider(
    provider_name="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# Create a specialized content agent
content_agent = Agent(
    name="content_creator",
    instruction="""You are an expert content creator specializing in technical blog posts. 
    Create engaging, well-structured content with clear explanations and practical examples.""",
    provider=provider,
    model_name="deepseek-chat"
)

# Generate a technical blog post
response = Hive(content_agent).run("""
Write a 500-word blog post about the benefits of microservices architecture, 
including real-world examples and potential challenges.
""")

print(response)
```

### 📊 Structured Data Analysis with JSON Mode

```python
# Schema for financial analysis
financial_schema = {
    "type": "object", 
    "properties": {
        "risk_level": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "Investment risk assessment"
        },
        "recommendation": {
            "type": "string",
            "enum": ["buy", "hold", "sell"],
            "description": "Trading recommendation"
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Confidence score (0-1)"
        },
        "key_factors": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key factors influencing the decision"
        }
    },
    "required": ["risk_level", "recommendation", "confidence", "key_factors"]
}

# Create financial analysis agent
analyst_agent = Agent(
    name="financial_analyst",
    instruction="""You are a senior financial analyst. Analyze stocks and provide 
    structured recommendations based on market data, company fundamentals, and trends.""",
    provider=provider,
    model_name="deepseek-chat",
    json_mode=True,
    json_schema=financial_schema
)

# Analyze a stock
analysis = Hive(analyst_agent).run("""
Analyze Tesla (TSLA) stock based on recent performance, EV market trends, 
and competition. Consider Q4 2024 earnings and 2025 outlook.
""")

print(analysis)
# Returns structured JSON with risk_level, recommendation, confidence, and key_factors
```

### 🔌 MCP Integration - Time & Weather Assistant

```python
from moonlight_ai import MCPConfig, MCPRegistry

# Create MCP configurations for time and weather
time_config = MCPConfig(
    name="time",
    command="uvx",
    args=["mcp-server-time", "--local-timezone=Asia/Kolkata"],
    description="Provides current time in IST timezone",
    tags=["time", "utility", "timezone"]
)

weather_config = MCPConfig(
    name="weather",
    command="uvx", 
    args=["mcp-server-weather", "--api-key=${WEATHER_API_KEY}"],
    description="Real-time weather data and forecasts",
    tags=["weather", "forecast", "api"]
)

# Create a personal assistant with multiple MCP servers
assistant_agent = Agent(
    name="personal_assistant",
    instruction="""You are a helpful personal assistant with access to time and weather data. 
    Provide contextual responses that combine multiple data sources for better insights.""",
    provider=provider,
    model_name="deepseek-chat",
    mcp_servers=[time_config, weather_config]
)

# Get contextual information
response = Hive(assistant_agent).run("""
What's the current time in Mumbai? Also check the weather and let me know 
if it's a good time for an outdoor meeting this afternoon.
""")

print(response.assistant_message)
```

### Registry

```python
registry = MCPRegistry()

# Register a server
reistry.register(time_config)

# load a config
registry.get_by_name("time") # or search by tags based on the sdk
```

### 🎼 Multi-Agent Orchestra - Market Research Team

```python
from moonlight_ai import Orchestra, Agent

# Create specialized research agents
market_researcher_agent = Agent(
    name="market_researcher", 
    instruction="""You are a market research specialist. Focus on industry trends, 
    market size, growth rates, and competitive landscape analysis.""",
    provider=provider,
    model_name="deepseek-chat",
)

competitor_analyst_agent = Agent(
    name="competitor_analyst",
    instruction="""You are a competitive intelligence expert. Analyze competitor 
    strategies, pricing, product features, and market positioning.""", 
    provider=provider,
    model_name="deepseek-chat",
)

financial_analyst_agent = Agent(
    name="financial_analyst",
    instruction="""You are a financial analyst specializing in startup and tech company 
    valuations, funding rounds, and financial performance metrics.""",
    provider=provider,
    model_name="deepseek-chat",
)

# Set up the research orchestra
research_team = {
    "market_researcher_agent": market_researcher_agent,
    "competitor_analyst_agent": competitor_analyst_agent,
    "financial_analyst_agent": financial_analyst_agent
}

orchestrator = Orchestra(
    orchestra_model="deepseek-chat",
    provider=provider
)

orchestrator.add_agents(research_team)
orchestrator.add_task("""
Conduct a comprehensive analysis of the AI code assistant market in 2025. 
I need insights on:
1. Market size and growth projections
2. Key competitors and their positioning
3. Recent funding rounds and valuations
4. Emerging trends and opportunities

Provide a structured report with actionable insights for a startup entering this space.
""")

# Execute the research
results = orchestrator.activate()
print(results)
```

### 🔍 DeepResearch - Comprehensive Industry Analysis

```python
from moonlight_ai import DeepResearcher

# Create a deep researcher for emerging technologies
tech_researcher = DeepResearcher(
    model="deepseek-chat",
    provider=provider,
    research_task="""
    Analyze the current state and future prospects of quantum computing in 2025.
    Focus on: commercial applications, major players, recent breakthroughs, 
    investment trends, and timeline for mainstream adoption.
    """,
    max_depth=3,        # Deep recursive research
    breadth=4,          # Multiple angles per level
    links_per_query=8,  # Comprehensive source coverage
    max_content_length=2000  # Detailed content analysis
)

# Execute comprehensive research
print("🔍 Starting deep research on quantum computing...")
results = tech_researcher.run()

# Get the detailed research report
print("📄 Research Report:")
print("=" * 80)
print(results)
```

### ⚡ Workflow - Automated Trading Bot

#### Example Schema
```json
{
  "blocks": {
    "block_1": {
      "type": "agent",
      "agent_name": "market_scanner",
      "task": "Scan market data for stocks crossing above 20-day moving average using a general market data API and filter results to only include stocks with volume exceeding 150% of 30-day average",
      "connected_to": [
        {
          "next_block": "block_2",
          "condition": "None"
        }
      ]
    },
    "block_2": {
      "type": "agent",
      "agent_name": "risk_manager",
      "task": "Calculate risk/reward ratio for each qualifying stock and reject any stock with risk/reward ≤ 2:1",
      "connected_to": [
        {
          "next_block": "block_3",
          "condition": "Some random condition"
        },
        {
          "next_block": "block_4",
          "condition": "Another random condition"
        }
      ]
    },
    "block_3": {
      "type": "agent",
      "agent_name": "risk_manager",
      "task": "Determine position size based on account balance (placeholder $10,000) and 2% stop-loss",
      "connected_to": [
        {
          "next_block": "block_4",
          "condition": "None"
        }
      ]
    },
    "block_4": {
      "type": "agent",
      "agent_name": "trade_executor",
      "task": "For approved trades, set stop-loss at 2% below entry price and set take-profit at 4% above entry price",
      "connected_to": [
        {
          "next_block": "block_5",
          "condition": "None"
        }
      ]
    },
    "block_5": {
      "type": "agent",
      "agent_name": "trade_executor",
      "task": "Execute trades through a generic brokerage API (authentication details required)",
      "connected_to": []
    }
  },
  "connections": {
    "block_1": ["block_2"],
    "block_2": ["block_3", "block_4"],
    "block_3": ["block_4"],
    "block_4": ["block_5"]
  },
  "trigger_block": {
    "type": "agent",
    "agent_name": "market_scanner",
    "task": "Scan market data for stocks crossing above 20-day moving average",
    "condition": "Is it between 9:30 AM - 4:00 PM EST?",
    "repeat": "15m"
  }
}
```

#### Trading Bot Example
```python
from moonlight_ai import WorkflowGenerator, Workflow, Agent

# Create specialized trading agents
market_scanner = Agent(
    name="market_scanner",
    instruction="""You are a market scanning specialist. Monitor stock prices, 
    volume, and technical indicators. Identify potential trading opportunities.""",
    provider=provider,
    model_name="deepseek-chat"
)

risk_manager = Agent(
    name="risk_manager", 
    instruction="""You are a risk management expert. Evaluate trading opportunities 
    for risk/reward ratios, position sizing, and portfolio impact.""",
    provider=provider,
    model_name="deepseek-chat"
)

trade_executor = Agent(
    name="trade_executor",
    instruction="""You are a trade execution specialist. Execute trades based on 
    approved signals while managing order timing and execution quality.""",
    provider=provider,
    model_name="deepseek-chat"
)

available_agents = {
    "market_scanner": market_scanner,
    "risk_manager": risk_manager, 
    "trade_executor": trade_executor
}

# Generate automated trading workflow
wfg = WorkflowGenerator(
    available_agents=available_agents,
    model_name="deepseek-chat",
    provider=provider,
)

# Create trading workflow
trading_prompt = """
Create a trading workflow that:
1. Scans for stocks breaking above 20-day moving average with high volume
2. Analyzes risk/reward ratio and position sizing
3. Executes trades only if risk/reward > 2:1
4. Sets stop-loss at 2% and take-profit at 4%
5. Runs every 15 minutes during market hours
"""

print('🤖 Generating automated trading workflow...')

# Generate and execute the workflow
# Step 1: Generate first level workflow
print('\n1. Generating first level workflow...')
first_level_result = wfg.generate_first_level(trading_prompt)
print(f'First level result: {first_level_result}\n\n')

# Process first level result
previous_level_details = first_level_result["first_level_workflow"]
critique = first_level_result["critique"]

# Answer the requirements
answered_requirements = []
for requirement in first_level_result["requirements"]:
    requirement_answer = input(f"Answer the requirement: {requirement}\n")
    answered_requirements.append(f"Q. {requirement}; A. {requirement_answer}\n\n")

# Step 2: Generate final workflow
print('\n3. Generating final workflow...')
final_workflow = wfg.generate_final_workflow(
    critique=critique,
	answered_requirements=answered_requirements,
	previous_level_details=first_level_result
)

# Final Workflow can be a json or generated.
print(f'Final generated workflow: {final_workflow}')
print('\n' + '=' * 60)

# Create and run the trading workflow
trading_workflow = Workflow(
    workflow=final_workflow,
    available_agents=available_agents,
    provider=provider,
    model_name="deepseek-chat"
)

print('📈 Starting automated trading workflow...')
trading_workflow.run()
```

#### Simple TIme Workflow
```python
from moonlight_ai import Workflow, Agent

# Define a simple agent
simple_agent = Agent(
    name="minute_checker",
    instruction="Check the current minute.",
    provider=provider,
    model_name="deepseek-chat"
)

# Agents registry
available_agents = {
    "minute_checker": simple_agent
}

# Simple manual workflow (not using WorkflowGenerator)
simple_workflow_json = {
    "blocks": {
        "block_1": {
            "type": "agent",
            "agent_name": "minute_checker",
            "task": "Check current minute. If odd, print 'odd'. Else, print 'even'.",
            "connected_to": []
        }
    },
    "connections": {
        "block_1": []
    },
    "trigger_block": {
        "type": "agent",
        "agent_name": "minute_checker",
        "task": "Check current system time.",
        "condition": "Current minute is odd number.",
        "repeat": "1m"
    }
}

# Create and run the workflow
workflow = Workflow(
    workflow=simple_workflow_json,
    available_agents=available_agents,
    provider=provider,
    model_name="deepseek-chat"
)

print('⏱️ Running simple minute-checker workflow...')
workflow.run()
```

> Note: This mode will run the workflow if trigger is met and the trigger is reset at ever "N" repeat interval set indefinitely. (e.g., 15m -> Every 15 minutes)

---

## 🏗️ Architecture Overview

### 🏠 **Hive - Single Agent Engine**

Perfect for focused tasks requiring deep expertise:

- **Conversation Management** - Maintains context across interactions
- **Code Execution** - Python and shell command support
- **Multi-modal Processing** - Handle text, images, and structured data
- **Token Optimization** - Efficient context window management

**Best For:** Content creation, data analysis, specialized consulting

### 🎼 **Orchestra - Multi-Agent Coordination**

Ideal for complex tasks requiring diverse expertise:

- **Intelligent Task Decomposition** - Break complex problems into subtasks
- **Agent Specialization** - Each agent focuses on their domain expertise
- **Dynamic Coordination** - Agents collaborate and share information
- **Result Synthesis** - Combine outputs into cohesive final results

**Best For:** Research projects, business analysis, multi-step processes

### 🔍 **DeepResearch - Recursive Investigation**

Specialized for comprehensive research workflows:

- **Recursive Query Generation** - Questions lead to deeper questions
- **Multi-source Analysis** - Aggregate information from diverse sources
- **Progressive Refinement** - Each level builds on previous insights
- **Comprehensive Reporting** - Detailed, structured research outputs

**Best For:** Market research, academic investigation, competitive analysis

### ⚡ **Workflow - Automation Engine**

Advanced automation for complex business processes:

- **Trigger-based Execution** - Time, event, or condition-based activation
- **Conditional Logic** - Complex branching and decision trees
- **Multiple Block Types** - Combine different execution modes
- **Scalable Architecture** - Support for up to 100 workflow blocks

**Best For:** Business process automation, monitoring systems, trading bots

---

## 🔌 MCP Integration

### **3,000+ Pre-configured Servers** (To be done)

Moonlight includes validated configurations for thousands of MCP servers across categories:

- **🤖 AI/LLM Tools** - Model integrations, prompt management
- **🗄️ Databases** - PostgreSQL, MongoDB, Redis, Elasticsearch
- **🛠️ Developer Tools** - Git, Docker, CI/CD, monitoring
- **🔬 Research & Science** - Academic databases, calculation engines
- **📊 Business Tools** - CRM, analytics, project management
- **🌐 Web Services** - APIs, webhooks, cloud platforms

### **MCPRegistry - Configuration Management**

```python
from moonlight_ai import MCPRegistry, MCPConfig

# Create and register configurations
registry = MCPRegistry()

# Database configuration
db_config = MCPConfig(
    name="postgres",
    command="uvx",
    args=["mcp-server-postgres", "--connection-string=${DB_URL}"],
    description="PostgreSQL database integration",
    tags=["database", "sql", "postgres"]
)

registry.register(db_config)

# Retrieve and use saved configurations
saved_config = registry.get_by_name("postgres")
database_agent = Agent(
    name="db_analyst",
    instruction="You are a database analyst with PostgreSQL access.",
    provider=provider,
    model_name="deepseek-chat",
    mcp_servers=[saved_config]
)
```

---

## 🎯 Provider Support

Moonlight supports all major AI providers with unified interfaces:

- **OpenAI** - GPT-4, GPT-3.5-turbo
- **DeepSeek** - DeepSeek-Chat, DeepSeek-Coder
- **Google** - Gemini Pro, Gemini Flash
- **Anthropic** - Claude 3 (Opus, Sonnet, Haiku)
- **Groq** - Llama, Mixtral (ultra-fast inference)
- **Together AI** - Open source models
- **Hugging Face** - Transformers ecosystem

### **Easy Provider Switching**

```python
# Switch between providers seamlessly
openai_provider = MoonlightProvider(
	provider_name="openai", 
	api_key=openai_key
)

deepseek_provider = MoonlightProvider(
	"deepseek", 
	api_key=deepseek_key
)
groq_provider = MoonlightProvider(
	"groq", 
	api_key=groq_key
)

custom_provider = MooonlightProvider(
	provider_url="https://my_custom_server/v1", 
	api_key=custom_key
)

# Same agent, different backends
agent_gpt4 = Agent(
	"analyst", 
	instruction="...", 
	provider=openai_provider, 
	model_name="gpt-4"
)

agent_deepseek = Agent(
	"analyst",
	 instruction="...", 
	 provider=deepseek_provider, 
	 model_name="deepseek-chat"
)
```

---


## ToDo

- [ ] Code Cleanup
- [ ] Add console.logs to workflow
- [ ] Async Running for Hive Architecture
- [ ] Unit tests and full workflow/deepresearch test

---

<div align="center">

Made with 🌙 by the Moonlight Team

</div>
