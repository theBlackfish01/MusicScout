import os
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src import graph


def _state_for(query: str, **overrides) -> graph.AgentState:
    """Helper to generate the simplified AgentState for testing."""
    state: graph.AgentState = {
        "messages": [HumanMessage(content=query)],
        "next": "Supervisor",
        "sender": "User",
    }
    state.update(overrides)
    return state


# --- Mocks ---

class _FakeSupervisorChain:
    def __init__(self, decision: graph.SupervisorDecision):
        self._decision = decision

    def invoke(self, messages: list) -> graph.SupervisorDecision:
        return self._decision


class _FakeSupervisorLLM:
    def __init__(self, decision: graph.SupervisorDecision):
        self._decision = decision

    def with_structured_output(self, _) -> _FakeSupervisorChain:
        return _FakeSupervisorChain(self._decision)


class _FakeWorkerLLM:
    def __init__(self, message: AIMessage):
        self._message = message

    def bind_tools(self, _) -> Any:
        return self

    def invoke(self, _) -> AIMessage:
        return self._message


# --- Provider Resolution Tests ---

def test_resolve_provider_override():
    assert graph.resolve_provider("openai") == "openai"


def test_resolve_provider_env_openai(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert graph.resolve_provider() == "openai"


def test_resolve_provider_env_gemini(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    assert graph.resolve_provider() == "gemini"


def test_resolve_provider_raises_without_keys(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="No LLM provider configured"):
        graph.resolve_provider()


# --- Supervisor Node Tests ---

def test_supervisor_routes_to_research(monkeypatch):
    decision = graph.SupervisorDecision(next=graph.RouteChoice.RESEARCH)
    monkeypatch.setattr(graph, "_build_llm", lambda *_: ("mocked", _FakeSupervisorLLM(decision)))

    result = graph.supervisor_node(_state_for("Who founded Pink Floyd?"), config={})

    assert result == {"next": "ResearchAgent"}


def test_supervisor_routes_to_analysis(monkeypatch):
    decision = graph.SupervisorDecision(next=graph.RouteChoice.ANALYSIS)
    monkeypatch.setattr(graph, "_build_llm", lambda *_: ("mocked", _FakeSupervisorLLM(decision)))

    result = graph.supervisor_node(_state_for("Calculate 10 + 20 / 2"), config={})

    assert result == {"next": "AnalysisAgent"}


def test_supervisor_routes_to_finish(monkeypatch):
    decision = graph.SupervisorDecision(next=graph.RouteChoice.FINISH)
    monkeypatch.setattr(graph, "_build_llm", lambda *_: ("mocked", _FakeSupervisorLLM(decision)))

    result = graph.supervisor_node(_state_for("I have answered your question."), config={})

    assert result == {"next": "FINISH"}


# --- Agent Node Tests ---

def test_research_node_returns_message_and_sender(monkeypatch):
    ai_msg = AIMessage(content="Pink Floyd was formed in London.")
    monkeypatch.setattr(graph, "_build_llm", lambda *_: ("mocked", _FakeWorkerLLM(ai_msg)))

    result = graph.research_node(_state_for("Who founded Pink Floyd?"), config={})

    assert result["messages"] == [ai_msg]
    assert result["sender"] == "ResearchAgent"


def test_analysis_node_returns_message_and_sender(monkeypatch):
    ai_msg = AIMessage(content="The average age is 42.")
    monkeypatch.setattr(graph, "_build_llm", lambda *_: ("mocked", _FakeWorkerLLM(ai_msg)))

    result = graph.analysis_node(_state_for("Calculate the average age"), config={})

    assert result["messages"] == [ai_msg]
    assert result["sender"] == "AnalysisAgent"


# --- Conditional Routing Tests ---

def test_research_condition_routes_to_tools():
    msg_with_tools = AIMessage(
        content="",
        tool_calls=[{"name": "wikipedia", "args": {"query": "test"}, "id": "1", "type": "tool_call"}]
    )
    state = _state_for("test query", messages=[msg_with_tools])

    assert graph.research_condition(state) == "ResearchTools"


def test_research_condition_routes_to_supervisor():
    msg_plain_text = AIMessage(content="I found the information.")
    state = _state_for("test query", messages=[msg_plain_text])

    assert graph.research_condition(state) == "Supervisor"


def test_analysis_condition_routes_to_tools():
    msg_with_tools = AIMessage(
        content="",
        tool_calls=[{"name": "python_repl_tool", "args": {"code": "print(1)"}, "id": "1", "type": "tool_call"}]
    )
    state = _state_for("test query", messages=[msg_with_tools])

    assert graph.analysis_condition(state) == "AnalysisTools"


def test_analysis_condition_routes_to_supervisor():
    msg_plain_text = AIMessage(content="The calculation is complete.")
    state = _state_for("test query", messages=[msg_plain_text])

    assert graph.analysis_condition(state) == "Supervisor"


# --- Graph Invocation Tests ---

def test_invoke_graph_sends_only_human_message(monkeypatch):
    """Ensure invoke_graph relies on the checkpointer rather than overwriting state."""
    captured_state: dict = {}

    class _FakeCompiledGraph:
        def invoke(self, state, config):
            captured_state.update(state)
            return state

    monkeypatch.setattr(graph, "compiled_graph", _FakeCompiledGraph())

    graph.invoke_graph("My new question", thread_id="thread-123")

    assert len(captured_state["messages"]) == 1
    assert isinstance(captured_state["messages"][0], HumanMessage)
    assert captured_state["messages"][0].content == "My new question"