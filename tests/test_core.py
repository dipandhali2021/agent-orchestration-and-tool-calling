"""Tests for agent_orchestration_and_tool_calling."""

import pytest
from agent_orchestration_and_tool_calling import (
    Tool,
    ToolCall,
    ToolResult,
    Agent,
    RouteRule,
    Orchestrator,
)


# --- Tool tests ---


def test_tool_creation_with_auto_params():
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    tool = Tool(name="greet", description="Greet someone", fn=greet)
    assert tool.name == "greet"
    assert tool.description == "Greet someone"
    assert "name" in tool.parameters["properties"]


def test_tool_creation_with_explicit_params():
    tool = Tool(
        name="add",
        description="Add two numbers",
        fn=lambda a, b: a + b,
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
    )
    assert tool.name == "add"
    assert tool.parameters["required"] == ["a", "b"]


def test_tool_invoke_success():
    def add(a: int, b: int) -> int:
        return a + b

    tool = Tool(name="add", description="Add", fn=add)
    result = tool.invoke(a=2, b=3)
    assert result.success
    assert result.output == 5


def test_tool_invoke_error():
    def fail(msg: str) -> None:
        raise ValueError(msg)

    tool = Tool(name="fail", description="Fail", fn=fail)
    result = tool.invoke(msg="broken")
    assert not result.success
    assert result.error == "broken"


def test_tool_invoke_empty_params():
    tool = Tool(name="hello", description="Say hello", fn=lambda: "hello")
    result = tool.invoke()
    assert result.success
    assert result.output == "hello"


def test_tool_to_dict():
    tool = Tool(name="t", description="d", fn=lambda: None)
    d = tool.to_dict()
    assert d["name"] == "t"
    assert d["description"] == "d"
    assert "parameters" in d


def test_tool_repr():
    tool = Tool(name="test_tool", description="A test", fn=lambda: None)
    r = repr(tool)
    assert "test_tool" in r


# --- ToolResult tests ---


def test_tool_result_success():
    r = ToolResult(tool_name="t", output=42)
    assert r.success
    assert r.output == 42
    assert r.error is None


def test_tool_result_failure():
    r = ToolResult(tool_name="t", error="fail")
    assert not r.success
    assert r.error == "fail"


def test_tool_result_to_dict():
    r = ToolResult(tool_name="t", output=42)
    d = r.to_dict()
    assert d["tool_name"] == "t"
    assert d["output"] == 42
    assert d["success"] is True


def test_tool_result_repr():
    r = ToolResult(tool_name="t", output=1)
    assert "ToolResult" in repr(r)
    assert "t" in repr(r)


# --- ToolCall tests ---


def test_tool_call_creation():
    tc = ToolCall(tool_name="greet", arguments={"name": "Alice"})
    assert tc.tool_name == "greet"
    assert tc.arguments == {"name": "Alice"}


def test_tool_call_repr():
    tc = ToolCall(tool_name="t", arguments={"k": "v"})
    r = repr(tc)
    assert "t" in r


# --- Agent tests ---


def test_agent_creation():
    agent = Agent(name="helper", instructions="You are helpful")
    assert agent.name == "helper"
    assert agent.tools == []


def test_agent_with_tools():
    tool = Tool(name="echo", description="Echo", fn=lambda x: x)
    agent = Agent(name="e", instructions="Echo", tools=[tool])
    assert len(agent.tools) == 1
    assert agent.has_tool("echo")


def test_agent_add_tool():
    agent = Agent(name="a", instructions="")
    t = Tool(name="t", description="", fn=lambda: None)
    agent.add_tool(t)
    assert agent.has_tool("t")


def test_agent_get_tool_nonexistent():
    agent = Agent(name="a", instructions="")
    assert agent.get_tool("nothing") is None


def test_agent_to_dict():
    tool = Tool(name="t", description="d", fn=lambda: None)
    agent = Agent(name="a", instructions="inst", tools=[tool])
    d = agent.to_dict()
    assert d["name"] == "a"
    assert d["instructions"] == "inst"
    assert len(d["tools"]) == 1


def test_agent_repr():
    agent = Agent(name="test", instructions="")
    assert "test" in repr(agent)


# --- RouteRule tests ---


def test_route_rule_match():
    rule = RouteRule(patterns=["weather", "temperature"], target="weather")
    assert rule.match("What's the weather?")
    assert rule.match("temperature today")
    assert not rule.match("tell me a joke")


def test_route_rule_case_insensitive():
    rule = RouteRule(patterns=["weather"], target="w")
    assert rule.match("WEATHER report")
    assert rule.match("weather report")


def test_route_rule_repr():
    rule = RouteRule(patterns=["hello"], target="greeter")
    r = repr(rule)
    assert "hello" in r
    assert "greeter" in r


# --- Orchestrator tests ---


def test_orchestrator_empty():
    orch = Orchestrator()
    assert len(orch.agents) == 0
    assert orch.route("anything") is None


def test_orchestrator_register():
    agent = Agent(name="a", instructions="")
    orch = Orchestrator()
    orch.register(agent)
    assert len(orch.agents) == 1


def test_orchestrator_route_by_rule():
    agent = Agent(name="weather", instructions="Weather")
    rule = RouteRule(patterns=["weather"], target="weather")
    orch = Orchestrator(agents=[agent], rules=[rule])
    matched = orch.route("What's the weather?")
    assert matched is not None
    assert matched.name == "weather"


def test_orchestrator_route_no_match():
    agent = Agent(name="general", instructions="General")
    orch = Orchestrator(agents=[agent])
    assert orch.route("anything") is None


def test_orchestrator_route_default():
    default = Agent(name="default", instructions="Fallback")
    orch = Orchestrator(agents=[default], default_agent="default")
    matched = orch.route("anything")
    assert matched is not None
    assert matched.name == "default"


def test_orchestrator_route_default_precedence():
    """Rules take precedence over default agent."""
    general = Agent(name="general", instructions="General")
    weather = Agent(name="weather", instructions="Weather")
    rule = RouteRule(patterns=["weather"], target="weather")
    orch = Orchestrator(
        agents=[general, weather],
        rules=[rule],
        default_agent="general",
    )
    assert orch.route("weather report").name == "weather"
    assert orch.route("something else").name == "general"


def test_orchestrator_call_tool():
    def greet(name: str) -> str:
        return f"Hi {name}"

    tool = Tool(name="greet", description="Greet", fn=greet)
    agent = Agent(name="greeter", instructions="Greet", tools=[tool])
    orch = Orchestrator(agents=[agent])
    result = orch.call_tool("greeter", "greet", name="Alice")
    assert result.success
    assert result.output == "Hi Alice"


def test_orchestrator_call_tool_unknown_agent():
    orch = Orchestrator()
    with pytest.raises(ValueError, match="Unknown agent"):
        orch.call_tool("unknown", "tool")


def test_orchestrator_call_tool_unknown_tool():
    agent = Agent(name="a", instructions="")
    orch = Orchestrator(agents=[agent])
    with pytest.raises(ValueError, match="Unknown tool"):
        orch.call_tool("a", "nonexistent")


def test_orchestrator_add_rule():
    agent = Agent(name="test", instructions="")
    orch = Orchestrator(agents=[agent])
    orch.add_rule(RouteRule(patterns=["hello"], target="test"))
    assert orch.route("hello").name == "test"


def test_orchestrator_get_agent():
    agent = Agent(name="a", instructions="")
    orch = Orchestrator(agents=[agent])
    assert orch.get_agent("a") is agent
    assert orch.get_agent("missing") is None


def test_orchestrator_to_dict():
    tool = Tool(name="echo", description="Echo", fn=lambda x: x)
    agent = Agent(name="test", instructions="Test", tools=[tool])
    rule = RouteRule(patterns=["test"], target="test")
    orch = Orchestrator(agents=[agent], rules=[rule])
    d = orch.to_dict()
    assert len(d["agents"]) == 1
    assert d["agents"][0]["name"] == "test"
    assert len(d["rules"]) == 1
    assert d["rules"][0]["target"] == "test"


def test_orchestrator_multiple_agents():
    a1 = Agent(name="a", instructions="A")
    a2 = Agent(name="b", instructions="B")
    orch = Orchestrator(agents=[a1, a2])
    assert len(orch.agents) == 2


def test_orchestrator_multiple_rules_first_match_wins():
    """First matching rule takes priority."""
    agent_a = Agent(name="agent_a", instructions="A")
    agent_b = Agent(name="agent_b", instructions="B")
    orch = Orchestrator(agents=[agent_a, agent_b])
    orch.add_rule(RouteRule(patterns=["alpha"], target="agent_a"))
    orch.add_rule(RouteRule(patterns=["alpha", "beta"], target="agent_b"))
    assert orch.route("alpha").name == "agent_a"
    assert orch.route("beta").name == "agent_b"


def test_orchestrator_repr():
    agent = Agent(name="test", instructions="")
    orch = Orchestrator(agents=[agent])
    r = repr(orch)
    assert "Orchestrator" in r
