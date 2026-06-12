"""Tests for agent_orchestration_and_tool_calling."""

from typing import Optional

import pytest
from agent_orchestration_and_tool_calling import (
    Tool,
    ToolCall,
    ToolResult,
    Agent,
    RouteRule,
    Orchestrator,
)
from agent_orchestration_and_tool_calling.core import AgentLoop


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


# --- Schema validation tests ---


def test_auto_schema_int_type():
    def add(a: int, b: int) -> int:
        return a + b

    tool = Tool(name="add", description="Add", fn=add)
    assert tool.parameters["properties"]["a"]["type"] == "integer"
    assert tool.parameters["properties"]["b"]["type"] == "integer"


def test_auto_schema_float_type():
    def div(a: float, b: float) -> float:
        return a / b

    tool = Tool(name="div", description="Div", fn=div)
    assert tool.parameters["properties"]["a"]["type"] == "number"


def test_auto_schema_bool_type():
    def check(flag: bool) -> bool:
        return flag

    tool = Tool(name="check", description="Check", fn=check)
    assert tool.parameters["properties"]["flag"]["type"] == "boolean"


def test_auto_schema_optional_param():
    def greet(name: str, greeting: Optional[str] = None) -> str:
        return f"{greeting or 'Hello'}, {name}"

    tool = Tool(name="greet", description="Greet", fn=greet)
    assert "greeting" not in tool.parameters["required"]
    assert tool.parameters["properties"]["greeting"].get("nullable") is True


def test_auto_schema_default_value():
    def repeat(msg: str, times: int = 1) -> str:
        return msg * times

    tool = Tool(name="repeat", description="Repeat", fn=repeat)
    assert tool.parameters["properties"]["times"]["default"] == 1
    assert "times" not in tool.parameters["required"]


def test_auto_schema_required_params():
    def cmd(name: str, action: str = "run") -> str:
        return f"{name}:{action}"

    tool = Tool(name="cmd", description="Cmd", fn=cmd)
    assert "name" in tool.parameters["required"]
    assert "action" not in tool.parameters["required"]


# --- Execution error handling tests ---


def test_tool_invoke_missing_argument():
    def greet(name: str) -> str:
        return f"Hello, {name}"

    tool = Tool(name="greet", description="Greet", fn=greet)
    result = tool.invoke()
    assert not result.success
    assert result.error is not None
    assert "missing" in result.error.lower()


def test_tool_invoke_unexpected_kwarg():
    def greet(name: str) -> str:
        return f"Hello, {name}"

    tool = Tool(name="greet", description="Greet", fn=greet)
    result = tool.invoke(name="Alice", extra="x")
    assert not result.success
    assert result.error is not None
    assert any(
        w in result.error.lower()
        for w in ["unexpected", "unexpected keyword argument"]
    )


def test_tool_invoke_type_error():
    def add(a: int, b: int) -> int:
        return a + b

    tool = Tool(name="add", description="Add", fn=add)
    result = tool.invoke(a="not_a_number", b=2)
    assert not result.success
    assert result.error is not None


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


# --- Tool registration tests ---


def test_agent_register_tool_twice():
    """Agent preserves both tools when the same name is added twice."""
    agent = Agent(name="test", instructions="")
    t1 = Tool(name="echo", description="First", fn=lambda: 1)
    t2 = Tool(name="echo", description="Second", fn=lambda: 2)
    agent.add_tool(t1)
    agent.add_tool(t2)
    assert len(agent.tools) == 2
    # get_tool returns first match via sequential search
    assert agent.get_tool("echo") is t1


def test_agent_register_multiple_tools():
    agent = Agent(name="test", instructions="")
    tools = [Tool(name=f"t{i}", description=str(i), fn=lambda i=i: i) for i in range(5)]
    for t in tools:
        agent.add_tool(t)
    assert len(agent.tools) == 5
    for t in tools:
        assert agent.has_tool(t.name)


def test_agent_register_no_tools():
    agent = Agent(name="empty", instructions="")
    assert agent.tools == []
    assert agent.get_tool("anything") is None


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


# --- AgentLoop orchestration loop tests ---


def test_agentloop_creation():
    """AgentLoop can be created with an orchestrator."""
    orch = Orchestrator()
    loop = AgentLoop(orch)
    assert loop.orchestrator is orch
    assert loop.state == {}


def test_agentloop_creation_with_initial_state():
    """AgentLoop accepts an initial state dict."""
    orch = Orchestrator()
    loop = AgentLoop(orch, initial_state={"foo": "bar"})
    assert loop.state == {"foo": "bar"}


def test_agentloop_state_snapshot():
    """state property returns a copy, not the internal dict."""
    orch = Orchestrator()
    loop = AgentLoop(orch)
    s = loop.state
    s["injected"] = "value"
    assert loop.state == {}

    # Multiple calls return independent copies
    s1 = loop.state
    s2 = loop.state
    assert s1 is not s2


def test_agentloop_step_single_call():
    """A single tool call is executed and its result returned."""
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    tool = Tool(name="greet", description="Greet", fn=greet)
    agent = Agent(name="helper", instructions="Help", tools=[tool])
    orch = Orchestrator(agents=[agent])
    loop = AgentLoop(orch)

    results = loop.step([ToolCall("greet", {"name": "Alice"})])
    assert len(results) == 1
    assert results[0].success
    assert results[0].output == "Hello, Alice!"


def test_agentloop_step_multiple_calls():
    """Multiple tool calls in a single step are all executed."""
    def add(a: int, b: int) -> int:
        return a + b

    def mul(a: int, b: int) -> int:
        return a * b

    add_tool = Tool(name="add", description="Add", fn=add)
    mul_tool = Tool(name="mul", description="Multiply", fn=mul)
    agent = Agent(name="calc", instructions="Calc", tools=[add_tool, mul_tool])
    orch = Orchestrator(agents=[agent])
    loop = AgentLoop(orch)

    results = loop.step([
        ToolCall("add", {"a": 2, "b": 3}),
        ToolCall("mul", {"a": 4, "b": 5}),
    ])
    assert len(results) == 2
    assert results[0].output == 5
    assert results[1].output == 20


def test_agentloop_state_transition():
    """After a step, results are stored in loop state keyed by tool name."""
    def greet(name: str) -> str:
        return f"Hi, {name}"

    tool = Tool(name="greet", description="Greet", fn=greet)
    agent = Agent(name="helper", instructions="Help", tools=[tool])
    orch = Orchestrator(agents=[agent])
    loop = AgentLoop(orch)

    loop.step([ToolCall("greet", {"name": "Bob"})])
    assert loop.state["greet"] == "Hi, Bob"


def test_agentloop_tool_result_injection():
    """Successful tool outputs and failed error messages are both stored in state."""
    def succeed(x: int) -> int:
        return x * 2

    success_tool = Tool(name="success", description="", fn=succeed)
    agent = Agent(name="worker", instructions="", tools=[success_tool])
    orch = Orchestrator(agents=[agent])
    loop = AgentLoop(orch)

    # Successful call
    loop.step([ToolCall("success", {"x": 7})])
    assert loop.state["success"] == 14

    # Failed call (no agent has this tool)
    loop.step([ToolCall("unknown", {})])
    assert isinstance(loop.state["unknown"], str)
    assert "unknown" in loop.state["unknown"]


def test_agentloop_multi_step_accumulation():
    """State accumulates across multiple steps (multi-step tool calling)."""
    def add(a: int, b: int) -> int:
        return a + b

    def mul(a: int, b: int) -> int:
        return a * b

    add_tool = Tool(name="add", description="Add", fn=add)
    mul_tool = Tool(name="mul", description="Multiply", fn=mul)
    agent = Agent(name="calc", instructions="Calc", tools=[add_tool, mul_tool])
    orch = Orchestrator(agents=[agent])
    loop = AgentLoop(orch)

    # Step 1: add
    loop.step([ToolCall("add", {"a": 2, "b": 3})])
    assert loop.state["add"] == 5

    # Step 2: multiply using the result from step 1
    loop.step([ToolCall("mul", {"a": loop.state["add"], "b": 10})])
    assert loop.state["mul"] == 50

    # Both keys present after accumulation
    assert set(loop.state.keys()) == {"add", "mul"}


def test_agentloop_step_unknown_tool():
    """When no agent has the requested tool, an error result is returned."""
    orch = Orchestrator()
    loop = AgentLoop(orch)
    results = loop.step([ToolCall("nonexistent", {})])
    assert len(results) == 1
    assert not results[0].success
    assert results[0].error is not None
    assert "nonexistent" in results[0].error


def test_agentloop_step_with_agent_name():
    """When agent_name is given, only that agent is searched for tools."""
    def greet(name: str) -> str:
        return f"Hello, {name}"

    tool = Tool(name="greet", description="Greet", fn=greet)
    agent = Agent(name="helper", instructions="Help", tools=[tool])
    orch = Orchestrator(agents=[agent])
    loop = AgentLoop(orch)

    results = loop.step(
        [ToolCall("greet", {"name": "Alice"})], agent_name="helper"
    )
    assert len(results) == 1
    assert results[0].success
    assert results[0].output == "Hello, Alice"


def test_agentloop_step_wrong_agent_name():
    """When agent_name doesn't match an agent, tool lookup fails."""
    def greet(name: str) -> str:
        return f"Hello, {name}"

    tool = Tool(name="greet", description="Greet", fn=greet)
    agent = Agent(name="helper", instructions="Help", tools=[tool])
    orch = Orchestrator(agents=[agent])
    loop = AgentLoop(orch)

    results = loop.step(
        [ToolCall("greet", {"name": "Alice"})], agent_name="other"
    )
    assert len(results) == 1
    assert not results[0].success


def test_agentloop_repr():
    """AgentLoop repr includes class name and state keys."""
    orch = Orchestrator()
    loop = AgentLoop(orch)
    r = repr(loop)
    assert "AgentLoop" in r
