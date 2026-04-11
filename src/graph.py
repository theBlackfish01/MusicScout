import os
import sqlite3
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool as lc_tool
from langchain_experimental.utilities import PythonREPL
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

try:
    from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper
except ImportError:  # pragma: no cover - compatibility fallback
    from langchain.tools import DuckDuckGoSearchRun, WikipediaQueryRun
    from langchain.utilities import WikipediaAPIWrapper

import dotenv

dotenv.load_dotenv()


# --- State Management ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next: str
    sender: str


class RouteChoice(str, Enum):
    RESEARCH = "ResearchAgent"
    ANALYSIS = "AnalysisAgent"
    FINISH = "FINISH"


class SupervisorDecision(BaseModel):
    next: RouteChoice


# --- Provider & LLM Setup ---
def resolve_provider(provider_override: str | None = None) -> str:
    """Resolve the active provider, considering overrides and environment variables."""
    provider = provider_override or os.getenv("LLM_PROVIDER", "")
    provider = provider.lower() if provider else ""

    if not provider:
        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("GEMINI_API_KEY"):
            provider = "gemini"
        else:
            raise EnvironmentError(
                "No LLM provider configured. Set OPENAI_API_KEY or GEMINI_API_KEY."
            )
    return provider


@lru_cache(maxsize=8)
def _build_llm(provider_override: str | None = None, model_override: str | None = None) -> tuple[str, BaseChatModel]:
    """Select and instantiate the LLM based on environment configuration or request overrides.

    Cached by (provider_override, model_override) so the hot graph loop does not
    re-instantiate the LLM client (and its underlying httpx connection pool) on
    every node invocation. In tests, monkeypatching ``graph._build_llm`` replaces
    this attribute entirely, bypassing the cache.
    """
    provider = resolve_provider(provider_override)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return "openai", ChatOpenAI(
            model=model_override or os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=0,
        )
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return "gemini", ChatGoogleGenerativeAI(
            model=model_override or os.getenv("GEMINI_MODEL", "gemini-1.5-pro"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0,
        )
    raise EnvironmentError(
        f"Unknown LLM_PROVIDER: {provider!r}. Valid values are 'openai' or 'gemini'."
    )


# --- Tools ---
duckduckgo_tool = DuckDuckGoSearchRun()
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())


@lc_tool
def python_repl_tool(code: str) -> str:
    """Execute Python code for arithmetic, statistics, and data analysis.
    Input must be valid, executable Python code.
    IMPORTANT: You MUST use print() to output the final result.
    IMPORTANT: Use the standard Python library ONLY. Do NOT import pandas, numpy, or other external libraries.
    If you do not use print(), you will receive no output!"""
    # Instantiate a fresh PythonREPL per call. A shared module-level REPL would
    # leak variables across requests (and across users), which is both a
    # correctness and privacy bug.
    return PythonREPL().run(code)


research_tools = [duckduckgo_tool, wikipedia_tool]
analysis_tools = [python_repl_tool]


def _handle_tool_error(e: Exception) -> str:
    return f"Tool execution failed: {type(e).__name__}: {e}"


# --- Nodes ---

def supervisor_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """The Supervisor reads the full history and decides what to do next."""
    messages = state.get("messages", [])

    # DETERMINISTIC OVERRIDE: If the last message is a hard system error, force an exit.
    if messages:
        last_content = str(messages[-1].content)
        if "SYSTEM ERROR:" in last_content or "Calculation failed" in last_content:
            return {"next": RouteChoice.FINISH.value}

    provider = config.get("configurable", {}).get("provider")
    model = config.get("configurable", {}).get("model")
    _, llm = _build_llm(provider, model)
    supervisor_chain = llm.with_structured_output(SupervisorDecision)

    sys_msg = SystemMessage(
        content=(
            "You are a routing supervisor managing a ResearchAgent and an AnalysisAgent. "
            "Review the conversation history. "
            "\n- If the user's request is fully answered, route to FINISH."
            "\n- If new factual data needs to be gathered, route to ResearchAgent."
            "\n- If calculations or data analysis are needed, FIRST verify if the raw numbers/dates "
            "are already present in the conversation history. If the required data is missing, "
            "route to the ResearchAgent to gather it. If the data is present, route to the AnalysisAgent."
            "\n- CRITICAL: If an agent has reported they cannot find the information or cannot "
            "perform the calculation after trying, route to FINISH to gracefully exit. Do NOT "
            "trap the system in a loop."
        )
    )

    decision = supervisor_chain.invoke([sys_msg] + state["messages"])
    return {"next": decision.next.value}


def research_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    provider = config.get("configurable", {}).get("provider")
    model = config.get("configurable", {}).get("model")
    _, llm = _build_llm(provider, model)
    research_llm = llm.bind_tools(research_tools)

    sys_msg = SystemMessage(
        content=(
            "You are a music research expert. Use the available search tools to find "
            "accurate facts and raw data relevant to the user's query."
            "\n- IMPORTANT: Gather raw formats (e.g., MM:SS). Do NOT attempt to perform math."
            "\n- CRITICAL: Output ONLY the requested data. Do NOT ask follow-up questions."
            "\n- ANTI-RABBIT HOLE: You have a strict limit of 2-3 searches per missing fact. "
            "If the exact data is buried, messy, or unavailable after 3 attempts, you MUST stop "
            "searching. Use the best available estimate you found, or explicitly output "
            "'[Data Unavailable]' and move on. Do not get stuck endlessly tweaking search queries."
        )
    )

    ai_msg = research_llm.invoke([sys_msg] + state["messages"])
    return {"messages": [ai_msg], "sender": "ResearchAgent"}


def research_override_node(state: AgentState) -> dict[str, Any]:
    """Break out of a runaway research loop.

    When ``research_condition`` detects that too many consecutive tool
    interactions have occurred, it routes here instead of back to the tool
    node. This node:

    1. Synthesizes a ``ToolMessage`` for every pending ``tool_call`` on the
       last AI message, so the message history remains well-formed (the LLM
       provider SDK will error if any AI ``tool_call`` has no matching
       ``ToolMessage`` response).
    2. Appends an ``AIMessage`` halt notice so the Supervisor can see what
       happened and decide to FINISH or route to analysis with the data
       already gathered.

    Returning these via the normal node ``dict`` respects the ``add_messages``
    reducer and the checkpointer — unlike mutating the state inside a
    conditional edge.
    """
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    synthetic_responses: list[BaseMessage] = []
    for tc in tool_calls:
        tc_id = tc.get("id") if isinstance(tc, dict) else None
        if not tc_id:
            continue
        synthetic_responses.append(
            ToolMessage(
                content=(
                    "System override: research search budget exhausted. "
                    "No further tool calls permitted. Use the data already "
                    "gathered or report '[Data Unavailable]'."
                ),
                tool_call_id=tc_id,
                name=tc.get("name", "research_tool") if isinstance(tc, dict) else "research_tool",
            )
        )

    halt_notice = AIMessage(
        content=(
            "Research search limit reached. I have stopped searching and will "
            "return the best data gathered so far. If critical information is "
            "still missing, I will report '[Data Unavailable]'."
        )
    )

    return {
        "messages": synthetic_responses + [halt_notice],
        "sender": "ResearchAgent",
    }


def analysis_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    provider = config.get("configurable", {}).get("provider")
    model = config.get("configurable", {}).get("model")
    _, llm = _build_llm(provider, model)
    analysis_llm = llm.bind_tools(analysis_tools)

    sys_msg = SystemMessage(
        content=(
            "You are a data analysis expert. Use the python_repl_tool to calculate answers based "
            "on the conversation history. If the required data to perform the calculation is missing, "
            "state clearly what data is needed."
        )
    )

    # INJECTION: Force the LLM to realize it is its turn to act
    nudge_msg = HumanMessage(
        content="Please perform the necessary calculations using the python_repl_tool."
    )

    ai_msg = analysis_llm.invoke([sys_msg] + state["messages"] + [nudge_msg])

    # SAFETY NET: If the LLM still returns nothing, force a text response so the
    # Supervisor knows it failed, rather than causing an empty routing loop.
    if not ai_msg.tool_calls and not str(ai_msg.content).strip():
        ai_msg = AIMessage(content="SYSTEM ERROR: Calculation failed. I cannot process this request. Route to FINISH.")

    return {"messages": [ai_msg], "sender": "AnalysisAgent"}


# --- Edge Logic ---

#: Each search round produces two messages: an AIMessage with tool_calls and
#: a ToolMessage response. A threshold of 8 consecutive tool-related messages
#: therefore corresponds to 4 full search round-trips.
RESEARCH_SEARCH_LIMIT_MSGS = 8


def research_condition(state: AgentState) -> Literal["ResearchTools", "ResearchOverride", "Supervisor"]:
    """Route the ResearchAgent's output.

    PURE function — must not mutate ``state``. Conditional edges are expected
    to inspect state and return a node name; any state mutation here bypasses
    the ``add_messages`` reducer and the checkpointer.
    """
    messages = state["messages"]
    last_message = messages[-1]

    if not getattr(last_message, "tool_calls", None):
        return "Supervisor"

    # Count consecutive tool-related messages walking backwards from the end.
    consecutive = 0
    for msg in reversed(messages):
        if getattr(msg, "tool_calls", None) or getattr(msg, "type", "") == "tool":
            consecutive += 1
        else:
            break

    if consecutive >= RESEARCH_SEARCH_LIMIT_MSGS:
        return "ResearchOverride"

    return "ResearchTools"


def analysis_condition(state: AgentState) -> Literal["AnalysisTools", "Supervisor"]:
    """Route to tools if the AnalysisAgent made a tool call, otherwise back to Supervisor."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "AnalysisTools"
    return "Supervisor"


# --- Graph Construction ---
builder = StateGraph(AgentState)

# Add Nodes
builder.add_node("Supervisor", supervisor_node)
builder.add_node("ResearchAgent", research_node)
builder.add_node("ResearchTools", ToolNode(research_tools, handle_tool_errors=_handle_tool_error))
builder.add_node("ResearchOverride", research_override_node)
builder.add_node("AnalysisAgent", analysis_node)
builder.add_node("AnalysisTools", ToolNode(analysis_tools, handle_tool_errors=_handle_tool_error))

# Add Edges
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

# Agent -> Tool -> Agent loops. ResearchAgent has a third exit: the
# ResearchOverride node, which breaks runaway search loops by synthesizing
# well-formed tool responses and routing to the Supervisor for a graceful
# wind-down.
builder.add_conditional_edges(
    "ResearchAgent",
    research_condition,
    {
        "ResearchTools": "ResearchTools",
        "ResearchOverride": "ResearchOverride",
        "Supervisor": "Supervisor",
    },
)
builder.add_edge("ResearchTools", "ResearchAgent")
builder.add_edge("ResearchOverride", "Supervisor")

builder.add_conditional_edges("AnalysisAgent", analysis_condition)
builder.add_edge("AnalysisTools", "AnalysisAgent")

def _open_checkpoint_connection() -> sqlite3.Connection:
    """Open the sqlite connection used by ``SqliteSaver``.

    Configuration notes:

    - ``check_same_thread=False`` — FastAPI runs sync endpoints on a thread
      pool, so the connection must be usable from any thread. Python's
      sqlite3 module still serializes access internally per-connection.
    - ``isolation_level=None`` — autocommit mode. SqliteSaver issues its own
      transactions and the default DEFERRED isolation otherwise holds write
      locks longer than needed.
    - ``timeout=30.0`` + ``busy_timeout=30000`` — wait up to 30s on a locked
      database before raising ``OperationalError`` rather than failing
      immediately under contention.
    - ``journal_mode=WAL`` — Write-Ahead Logging allows readers and writers
      to proceed concurrently, which matters as soon as more than one
      request is in flight.
    - ``synchronous=NORMAL`` — safe with WAL and significantly faster than
      the default FULL.
    """
    db_path = Path(__file__).resolve().parent.parent / "data" / "checkpoints.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(db_path),
        check_same_thread=False,
        isolation_level=None,
        timeout=30.0,
    )
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


_checkpoint_conn = _open_checkpoint_connection()
checkpointer = SqliteSaver(_checkpoint_conn)
compiled_graph = builder.compile(checkpointer=checkpointer)


def close_checkpointer() -> None:
    """Close the shared sqlite connection.

    Wired into the FastAPI ``lifespan`` handler in ``main.py`` so the
    connection is released on application shutdown. Swallows errors because
    shutdown hooks must not raise.
    """
    try:
        _checkpoint_conn.close()
    except Exception:
        pass


# --- Invocation Method ---
def invoke_graph(
        query: str,
        thread_id: str,
        provider: str | None = None,
        model: str | None = None,
) -> dict[str, Any]:
    # We only inject the new message. LangGraph's checkpointer will correctly
    # append this to the history for an existing thread_id.
    initial_state = {
        "messages": [HumanMessage(content=query)],
    }

    config = {
        "configurable": {"thread_id": thread_id, "provider": provider, "model": model},
        "recursion_limit": 30,
    }

    # Snapshot the prior message count for this thread so callers can extract
    # ONLY the messages produced by this run (not the accumulated thread
    # history). Without this, trace_steps and token totals double-count every
    # message from earlier runs whenever a thread_id is reused.
    prior_len = 0
    try:
        prior_state = compiled_graph.get_state(config)
    except Exception:
        prior_state = None
    if prior_state is not None:
        prior_values = getattr(prior_state, "values", None) or {}
        prior_len = len(prior_values.get("messages", []))

    final_state = compiled_graph.invoke(initial_state, config=config)

    if isinstance(final_state, dict):
        all_messages = final_state.get("messages", [])
        final_state["new_messages"] = all_messages[prior_len:]

    return final_state