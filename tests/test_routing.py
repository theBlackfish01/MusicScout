from langchain_core.messages import HumanMessage

from src import graph


def _state_for(query: str, tool_traces: list[dict] | None = None) -> graph.AgentState:
    return {
        "messages": [HumanMessage(content=query)],
        "next": "Supervisor",
        "tool_traces": tool_traces or [],
    }


class _FakeSupervisorChain:
    def __init__(self, decisions):
        if isinstance(decisions, list):
            self._decisions = iter(decisions)
        else:
            self._decisions = decisions

    def invoke(self, _):
        if hasattr(self._decisions, "__next__"):
            return next(self._decisions)
        return self._decisions


class _FakeLLM:
    def __init__(self, decisions):
        self.chain = _FakeSupervisorChain(decisions)
    
    def with_structured_output(self, schema):
        return self.chain


def test_supervisor_routes_facts_query_to_research(monkeypatch):
    monkeypatch.setattr(
        graph,
        "_build_llm",
        lambda p, m: ("mocked", _FakeLLM(graph.SupervisorDecision(next=graph.RouteChoice.RESEARCH)))
    )

    result = graph.supervisor_node(_state_for("Who founded Pink Floyd?"), config={})

    assert result == {"next": "ResearchAgent"}


def test_supervisor_routes_compute_query_to_analysis(monkeypatch):
    monkeypatch.setattr(
        graph,
        "_build_llm",
        lambda p, m: ("mocked", _FakeLLM(graph.SupervisorDecision(next=graph.RouteChoice.ANALYSIS)))
    )

    result = graph.supervisor_node(_state_for("Calculate 10 + 20 / 2"), config={})

    assert result == {"next": "AnalysisAgent"}


def test_supervisor_multi_hop_progression(monkeypatch):
    decisions = [
        graph.SupervisorDecision(next=graph.RouteChoice.RESEARCH),
        graph.SupervisorDecision(next=graph.RouteChoice.ANALYSIS),
        graph.SupervisorDecision(next=graph.RouteChoice.FINISH),
    ]
    fake_llm = _FakeLLM(decisions)
    monkeypatch.setattr(
        graph,
        "_build_llm",
        lambda p, m: ("mocked", fake_llm)
    )

    state = _state_for("Find top 5 longest Pink Floyd songs and compute average length.")

    first = graph.supervisor_node(state, config={})
    state["tool_traces"].append(
        {
            "agent": "ResearchAgent",
            "tool": "DuckDuckGoSearchRun",
            "input": "Pink Floyd longest songs",
            "output": "mocked results",
        }
    )
    second = graph.supervisor_node(state, config={})
    state["tool_traces"].append(
        {
            "agent": "AnalysisAgent",
            "tool": "PythonREPL",
            "input": "average computation",
            "output": "420.3",
        }
    )
    third = graph.supervisor_node(state, config={})

    assert first == {"next": "ResearchAgent"}
    assert second == {"next": "AnalysisAgent"}
    assert third == {"next": "FINISH"}
