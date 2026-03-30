import os
import re
from enum import Enum
from typing import Any, TypedDict

from langchain_experimental.utilities import PythonREPL
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

try:
    from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper
except ImportError:  # pragma: no cover - compatibility fallback
    from langchain.tools import DuckDuckGoSearchRun, WikipediaQueryRun
    from langchain.utilities import WikipediaAPIWrapper

import dotenv
dotenv.load_dotenv()

class ToolTrace(TypedDict):
    agent: str
    tool: str
    input: str
    output: str


class AgentState(TypedDict):
    messages: list[BaseMessage]
    next: str
    tool_traces: list[ToolTrace]


class RouteChoice(str, Enum):
    RESEARCH = "ResearchAgent"
    ANALYSIS = "AnalysisAgent"
    FINISH = "FINISH"


class SupervisorDecision(BaseModel):
    next: RouteChoice


def _build_llm() -> tuple[str, BaseChatModel]:
    """Select and instantiate the LLM based on environment configuration.

    Priority:
      1. LLM_PROVIDER env var (explicit override: "openai" | "gemini")
      2. OPENAI_API_KEY present → openai
      3. GEMINI_API_KEY present → gemini
      4. Neither → EnvironmentError
    """
    provider = os.getenv("LLM_PROVIDER", "").lower()

    if not provider:
        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("GEMINI_API_KEY"):
            provider = "gemini"
        else:
            raise EnvironmentError(
                "No LLM provider configured. Set OPENAI_API_KEY or GEMINI_API_KEY."
            )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return "openai", ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=0,
        )
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return "gemini", ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-pro"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0,
        )
    raise EnvironmentError(
        f"Unknown LLM_PROVIDER: {provider!r}. Valid values are 'openai' or 'gemini'."
    )


duckduckgo_tool: DuckDuckGoSearchRun = DuckDuckGoSearchRun()
wikipedia_tool: WikipediaQueryRun = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
python_repl: PythonREPL = PythonREPL()
active_provider, llm = _build_llm()
supervisor_chain = llm.with_structured_output(SupervisorDecision)

SUPERVISOR_SYSTEM_PROMPT = """You are a routing supervisor. Based on the query and tool_traces, decide which agent to call next.
If the query requires searching for facts, names, or history, route to ResearchAgent.
If the query requires arithmetic, ranking, or data structuring, route to AnalysisAgent.
If mathematical calculation is required, you MUST route to AnalysisAgent before returning FINISH.
Return FINISH only when the final answer is complete and all required computations are done."""


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return str(content)


def _latest_message_text(state: AgentState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return ""
    return _message_text(messages[-1])


def _original_user_query(state: AgentState) -> str:
    for message in state.get("messages", []):
        if isinstance(message, HumanMessage):
            return _message_text(message)
    return _latest_message_text(state)


def _invoke_tool(tool: Any, tool_input: str) -> str:
    if hasattr(tool, "invoke"):
        return str(tool.invoke(tool_input))
    if hasattr(tool, "run"):
        return str(tool.run(tool_input))
    raise TypeError(f"Unsupported tool type: {type(tool)!r}")


def _should_use_wikipedia(query: str) -> bool:
    keywords = ("history", "origin", "biography", "born", "genre", "fact", "who is")
    normalized = query.lower()
    return any(keyword in normalized for keyword in keywords)


def _should_use_duckduckgo(query: str) -> bool:
    keywords = ("recent", "news", "trend", "latest", "today", "current")
    normalized = query.lower()
    return any(keyword in normalized for keyword in keywords)


def _extract_expression(query: str) -> str | None:
    candidates = re.findall(r"[0-9\.\+\-\*\/\(\)\s%]+", query)
    for candidate in candidates:
        expr = candidate.strip()
        if not expr:
            continue
        if not any(ch.isdigit() for ch in expr):
            continue
        if re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s%]+", expr):
            return expr
    return None


def research_node(state: AgentState) -> dict[str, Any]:
    query = _latest_message_text(state)
    traces = list(state.get("tool_traces", []))
    messages = list(state.get("messages", []))

    run_duckduckgo = _should_use_duckduckgo(query) or not _should_use_wikipedia(query)
    run_wikipedia = _should_use_wikipedia(query)

    outputs: list[str] = []

    if run_duckduckgo:
        ddg_output = _invoke_tool(duckduckgo_tool, query)
        traces.append(
            {
                "agent": "ResearchAgent",
                "tool": "DuckDuckGoSearchRun",
                "input": query,
                "output": ddg_output,
            }
        )
        outputs.append(f"DuckDuckGo:\n{ddg_output}")

    if run_wikipedia:
        wiki_output = _invoke_tool(wikipedia_tool, query)
        traces.append(
            {
                "agent": "ResearchAgent",
                "tool": "WikipediaQueryRun",
                "input": query,
                "output": wiki_output,
            }
        )
        outputs.append(f"Wikipedia:\n{wiki_output}")

    if not outputs:
        fallback_output = _invoke_tool(duckduckgo_tool, query)
        traces.append(
            {
                "agent": "ResearchAgent",
                "tool": "DuckDuckGoSearchRun",
                "input": query,
                "output": fallback_output,
            }
        )
        outputs.append(f"DuckDuckGo:\n{fallback_output}")

    messages.append(
        AIMessage(content="Research findings:\n\n" + "\n\n".join(outputs))
    )
    return {"messages": messages, "tool_traces": traces, "next": "Supervisor"}


def analysis_node(state: AgentState) -> dict[str, Any]:
    query = _latest_message_text(state)
    traces = list(state.get("tool_traces", []))
    messages = list(state.get("messages", []))
    expression = _extract_expression(query)

    if expression:
        code = (
            "from statistics import mean\n"
            f"expr = {expression!r}\n"
            "result = eval(expr, {'__builtins__': {}}, {'mean': mean})\n"
            "print(result)\n"
        )
    else:
        code = (
            f"query = {query!r}\n"
            "print('No direct arithmetic expression found. Provide explicit numbers or formula for computation.')\n"
        )

    repl_output = str(python_repl.run(code))
    traces.append(
        {
            "agent": "AnalysisAgent",
            "tool": "PythonREPL",
            "input": code,
            "output": repl_output,
        }
    )
    messages.append(AIMessage(content=f"Analysis result:\n{repl_output}"))
    return {"messages": messages, "tool_traces": traces, "next": "Supervisor"}


def supervisor_node(state: AgentState) -> dict[str, Any]:
    user_query = _original_user_query(state)
    tool_traces = state.get("tool_traces", [])
    decision_input = (
        f"User query:\n{user_query}\n\n"
        f"Current tool traces:\n{tool_traces}\n\n"
        "Return only the next route."
    )
    decision = supervisor_chain.invoke(
        [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT), HumanMessage(content=decision_input)]
    )
    return {"next": decision.next.value}


builder = StateGraph(AgentState)
builder.add_node("Supervisor", supervisor_node)
builder.add_node("ResearchAgent", research_node)
builder.add_node("AnalysisAgent", analysis_node)

builder.add_edge(START, "Supervisor")
builder.add_conditional_edges(
    "Supervisor",
    lambda state: state["next"],
    {
        RouteChoice.RESEARCH.value: "ResearchAgent",
        RouteChoice.ANALYSIS.value: "AnalysisAgent",
        RouteChoice.FINISH.value: END,
    },
)
builder.add_edge("ResearchAgent", "Supervisor")
builder.add_edge("AnalysisAgent", "Supervisor")

checkpointer = MemorySaver()
compiled_graph = builder.compile(checkpointer=checkpointer)


def invoke_graph(query: str, thread_id: str) -> dict[str, Any]:
    """Invoke the compiled graph using thread-scoped state persistence."""
    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "next": "Supervisor",
        "tool_traces": [],
    }
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 10,
    }
    return compiled_graph.invoke(initial_state, config=config)
