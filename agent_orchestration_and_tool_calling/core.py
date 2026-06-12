"""Core types and orchestration for agent tool-calling systems."""

import inspect
from typing import (
    Any,
    Callable,
    Optional,
    Sequence,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

try:
    from typing import Literal  # Python 3.8+
except ImportError:
    Literal = None

__all__ = [
    "Tool",
    "ToolCall",
    "ToolResult",
    "Agent",
    "RouteRule",
    "Orchestrator",
    "ToolRegistry",
    "registry",
    "tool",
    "validate_args",
]


def _type_to_json_schema(tp: type) -> dict:
    """Convert a Python type annotation to a JSON Schema property."""
    origin = get_origin(tp)
    args = get_args(tp)

    # Optional[X] → Union[X, None]
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            schema = _type_to_json_schema(non_none[0])
            schema["nullable"] = True
            return schema
        return {"type": "string"}

    if origin is list:
        item = _type_to_json_schema(args[0]) if args else {}
        return {"type": "array", "items": item}

    if origin is dict:
        return {"type": "object"}

    if Literal is not None and origin is Literal:
        return {"type": "string", "enum": list(args)}

    # Primitive types
    if tp is str:
        return {"type": "string"}
    if tp is int:
        return {"type": "integer"}
    if tp is float:
        return {"type": "number"}
    if tp is bool:
        return {"type": "boolean"}

    return {"type": "string"}


def _infer_parameters(fn: Callable) -> dict:
    """Build a JSON Schema from a callable's signature."""
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    properties: dict[str, dict] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name == "return":
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        type_hint = hints.get(name, str)
        prop = _type_to_json_schema(type_hint)

        if param.default is not inspect.Parameter.empty:
            if param.default is not None:
                prop["default"] = param.default
            else:
                prop["nullable"] = True
        else:
            required.append(name)

        properties[name] = prop

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def validate_args(schema: dict, kwargs: dict) -> Optional[str]:
    """Validate keyword arguments against a JSON Schema parameter definition.

    Returns an error message if validation fails, or None if valid.
    """
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    for param in required:
        if param not in kwargs:
            return f"Missing required argument: {param!r}"

    allowed = set(properties.keys())
    for key in kwargs:
        if key not in allowed:
            return f"Unexpected argument: {key!r}"

    return None


class ToolCall:
    """Represents a request to invoke a specific tool."""

    def __init__(self, tool_name: str, arguments: dict[str, Any]) -> None:
        self.tool_name = tool_name
        self.arguments = arguments

    def __repr__(self) -> str:
        return (
            f"ToolCall(tool_name={self.tool_name!r}, "
            f"arguments={self.arguments!r})"
        )


class ToolResult:
    """The outcome of executing a Tool."""

    def __init__(
        self,
        tool_name: str,
        output: Any = None,
        error: Optional[str] = None,
    ) -> None:
        self.tool_name = tool_name
        self.output = output
        self.error = error

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "output": self.output,
            "error": self.error,
            "success": self.success,
        }

    def __repr__(self) -> str:
        return (
            f"ToolResult(tool_name={self.tool_name!r}, "
            f"success={self.success})"
        )


class Tool:
    """A callable tool with a name, description, and parameter schema."""

    def __init__(
        self,
        name: str,
        description: str,
        fn: Callable,
        parameters: Optional[dict] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.fn = fn
        self.parameters = (
            parameters if parameters is not None else _infer_parameters(fn)
        )

    def invoke(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given keyword arguments."""
        error = validate_args(self.parameters, kwargs)
        if error:
            return ToolResult(tool_name=self.name, error=error)
        try:
            output = self.fn(**kwargs)
            return ToolResult(tool_name=self.name, output=output)
        except Exception as exc:
            return ToolResult(tool_name=self.name, error=str(exc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r})"


class Agent:
    """An agent with a role, instructions, and a set of tools."""

    def __init__(
        self,
        name: str,
        instructions: str,
        tools: Optional[Sequence[Tool]] = None,
    ) -> None:
        self.name = name
        self.instructions = instructions
        self.tools: list[Tool] = list(tools) if tools else []

    def add_tool(self, tool: Tool) -> None:
        """Register a tool with this agent."""
        self.tools.append(tool)

    def get_tool(self, name: str) -> Optional[Tool]:
        """Look up a tool by name."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def has_tool(self, name: str) -> bool:
        return self.get_tool(name) is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "instructions": self.instructions,
            "tools": [t.to_dict() for t in self.tools],
        }

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, tools={len(self.tools)})"


class RouteRule:
    """A rule for routing tasks to agents by keyword matching."""

    def __init__(
        self,
        patterns: Sequence[str],
        target: str,
    ) -> None:
        self.patterns = [p.lower() for p in patterns]
        self.target = target

    def match(self, task: str) -> bool:
        """Return True if any pattern is found in the task (case-insensitive)."""
        task_lower = task.lower()
        return any(p in task_lower for p in self.patterns)

    def __repr__(self) -> str:
        return (
            f"RouteRule(patterns={self.patterns!r}, "
            f"target={self.target!r})"
        )


class Orchestrator:
    """Routes tasks to agents and manages tool execution."""

    def __init__(
        self,
        agents: Optional[Sequence[Agent]] = None,
        rules: Optional[Sequence[RouteRule]] = None,
        default_agent: Optional[str] = None,
    ) -> None:
        self._agents: dict[str, Agent] = {}
        self._rules: list[RouteRule] = []
        self.default_agent: Optional[str] = default_agent

        if agents:
            for agent in agents:
                self.register(agent)

        if rules:
            self._rules.extend(rules)

    def register(self, agent: Agent) -> None:
        """Register an agent with the orchestrator."""
        self._agents[agent.name] = agent

    def add_rule(self, rule: RouteRule) -> None:
        """Add a routing rule."""
        self._rules.append(rule)

    def get_agent(self, name: str) -> Optional[Agent]:
        """Retrieve a registered agent by name."""
        return self._agents.get(name)

    @property
    def agents(self) -> list[Agent]:
        return list(self._agents.values())

    def route(self, task: str) -> Optional[Agent]:
        """Route a task to the best-matching agent using rules, then default."""
        for rule in self._rules:
            if rule.match(task):
                agent = self._agents.get(rule.target)
                if agent:
                    return agent

        if self.default_agent:
            return self._agents.get(self.default_agent)

        return None

    def call_tool(
        self,
        agent_name: str,
        tool_name: str,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute a tool on a specific agent."""
        agent = self._agents.get(agent_name)
        if agent is None:
            raise ValueError(f"Unknown agent: {agent_name!r}")

        tool = agent.get_tool(tool_name)
        if tool is None:
            raise ValueError(
                f"Unknown tool {tool_name!r} on agent {agent_name!r}"
            )

        return tool.invoke(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agents": [a.to_dict() for a in self.agents],
            "rules": [
                {"patterns": r.patterns, "target": r.target}
                for r in self._rules
            ],
            "default_agent": self.default_agent,
        }

    def __repr__(self) -> str:
        return (
            f"Orchestrator(agents={len(self._agents)}, "
            f"rules={len(self._rules)})"
        )


class ToolRegistry:
    """A central registry for tools with registration, lookup, and invocation."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool by name."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Look up a registered tool by name."""
        return self._tools.get(name)

    def list(self) -> list[Tool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def invoke(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Look up a tool by name and invoke it with the given arguments."""
        tool = self.get(tool_name)
        if tool is None:
            return ToolResult(
                tool_name=tool_name,
                error=f"Unknown tool: {tool_name!r}",
            )
        return tool.invoke(**kwargs)

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={list(self._tools.keys())})"


# Global default registry.
registry = ToolRegistry()


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable[[Callable], Tool]:
    """Decorator that wraps a function into a Tool and registers it globally.

    Usage::

        @tool()
        def my_func(arg1: str, arg2: int) -> str:
            \"\"\"Description.\"\"\"
            ...

        @tool(name="custom_name", description="Custom description")
        def another() -> None:
            ...
    """
    def decorator(fn: Callable) -> Tool:
        t = Tool(
            name=name or fn.__name__,
            description=description or fn.__doc__ or "",
            fn=fn,
        )
        registry.register(t)
        return t

    return decorator
