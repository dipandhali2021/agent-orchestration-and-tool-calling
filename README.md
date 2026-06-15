# Agent Orchestration & Tool Calling

A dependency-free Python framework for agent orchestration and tool calling. No external dependencies — just the standard library.

Stacks that need to route tasks to the right agent and manage tool execution can use this instead of re-implementing the pattern.

## Install

```bash
pip install agent-orchestration-and-tool-calling
```

Or install from source:

```bash
git clone <repo-url>
cd agent-orchestration-and-tool-calling
pip install .
```

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

### Register tools with the decorator

The `@tool()` decorator wraps a function into a `Tool` and registers it on the global registry:

```python
from agent_orchestration_and_tool_calling import tool, registry

@tool()
def fetch_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny in {city}"

@tool(name="current_time", description="Get the current system time")
def get_time() -> str:
    import datetime
    return str(datetime.datetime.now())

# Look up and invoke from the global registry
result = registry.invoke("fetch_weather", city="London")
print(result.output)  # "Sunny in London"
```

The tool name defaults to the function name. The description defaults to the function's docstring. You can override both with keyword arguments.

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

### Run the orchestration loop

The `AgentLoop` class manages state across multiple tool-calling steps. It executes tool calls, stores results in a shared state dict keyed by tool name, and makes those results available for subsequent steps.

```python
from agent_orchestration_and_tool_calling import (
    Tool, ToolCall, Agent, Orchestrator, AgentLoop,
)

def add(a: int, b: int) -> int:
    return a + b

def mul(a: int, b: int) -> int:
    return a * b

add_tool = Tool(name="add", description="Add two numbers", fn=add)
mul_tool = Tool(name="mul", description="Multiply two numbers", fn=mul)

calc_agent = Agent(
    name="calculator",
    instructions="Perform calculations",
    tools=[add_tool, mul_tool],
)

orch = Orchestrator(agents=[calc_agent])
loop = AgentLoop(orch)

# Step 1: add two numbers
results = loop.step([ToolCall("add", {"a": 2, "b": 3})])
print(results[0].output)          # 5
print(loop.state["add"])          # 5 — stored in state

# Step 2: use previous result in a multiplication
results = loop.step([ToolCall("mul", {"a": loop.state["add"], "b": 10})])
print(results[0].output)          # 50
print(loop.state["mul"])          # 50
print(loop.state)                 # {"add": 5, "mul": 50}
```

Key behaviours:

- **State accumulation**: results persist across `step()` calls, so later steps can reference earlier outputs.
- **Cross-agent dispatch**: when `agent_name` is omitted, all registered agents are searched for the named tool. The first agent that has the tool executes it.
- **Scoped dispatch**: pass `agent_name` to restrict execution to a specific agent.
- **Error handling**: if a tool raises an exception, the error message is stored in state and the loop continues.

## API Overview

| Class | Purpose |
|-------|---------|
| `Tool` | A callable with a name, description, and JSON Schema parameters |
| `ToolCall` | A request to invoke a tool with specific arguments |
| `ToolResult` | The outcome of a tool invocation (output or error) |
| `ToolRegistry` | A central registry for registering, looking up, and invoking tools by name |
| `Agent` | A named entity with instructions and a set of tools |
| `RouteRule` | Keyword-based patterns that map tasks to agents |
| `Orchestrator` | Routes tasks to agents and dispatches tool calls |
| `AgentLoop` | Orchestration loop with state management for multi-step tool calling |

### Utility

| Function | Purpose |
|----------|---------|
| `tool()` | Decorator that wraps a function into a `Tool` and registers it on the global registry |
| `registry` | Global `ToolRegistry` instance for tool registration and lookup |
| `validate_args(schema, kwargs)` | Validates keyword arguments against a JSON Schema; returns `None` on success or an error string |

## License

MIT
