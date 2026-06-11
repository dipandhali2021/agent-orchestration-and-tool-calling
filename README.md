# Agent Orchestration & Tool Calling

A dependency-free Python framework for agent orchestration and tool calling. No external dependencies — just the standard library.

Stacks that need to route tasks to the right agent and manage tool execution can use this instead of re-implementing the pattern.

## Install

```bash
pip install agent-orchestration-and-tool-calling
```

Or copy the single `core.py` into your project.

Requires Python >= 3.10.

## Usage

### Define tools

```python
from agent_orchestration_and_tool_calling import Tool

def get_weather(city: str) -> str:
    return f"Sunny in {city}"

weather_tool = Tool(
    name="get_weather",
    description="Get the weather for a city",
    fn=get_weather,
)
```

Parameter schemas are inferred from type hints automatically. You can also pass an explicit `parameters` dict.

### Create agents

```python
from agent_orchestration_and_tool_calling import Agent

weather_agent = Agent(
    name="weather_agent",
    instructions="Handle weather-related queries",
    tools=[weather_tool],
)
```

### Route and execute

```python
from agent_orchestration_and_tool_calling import Orchestrator, RouteRule

orch = Orchestrator(
    agents=[weather_agent],
    rules=[RouteRule(patterns=["weather", "forecast"], target="weather_agent")],
    default_agent="weather_agent",
)

# Route a task to the best agent
agent = orch.route("What's the weather in London?")

# Execute a tool
result = orch.call_tool("weather_agent", "get_weather", city="London")
print(result.output)   # "Sunny in London"
print(result.success)  # True
```

## API Overview

| Class | Purpose |
|-------|---------|
| `Tool` | A callable with a name, description, and JSON Schema parameters |
| `ToolCall` | A request to invoke a tool with specific arguments |
| `ToolResult` | The outcome of a tool invocation (output or error) |
| `Agent` | A named entity with instructions and a set of tools |
| `RouteRule` | Keyword-based patterns that map tasks to agents |
| `Orchestrator` | Routes tasks to agents and dispatches tool calls |

## License

MIT
