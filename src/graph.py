import os
from enum import Enum
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool as lc_tool
from langchain_experimental.utilities import PythonREPL
from langgraph.checkpoint.memory import MemorySaver
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
# Notice how lean this is now. We rely on the conversation history rather than custom counters.
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next: str
    sender: str  # Tracks which agent most recently executed


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


def _build_llm(provider_override: str | None = None, model_override: str | None = None) -> tuple[str, BaseChatModel]:
    """Select and instantiate the LLM based on environment configuration or request overrides."""
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
_python_repl = PythonREPL()


@lc_tool
def python_repl_tool(code: str) -> str:
    """Execute Python code for arithmetic, statistics, and data analysis.
    Input must be valid, executable Python code.
    IMPORTANT: You MUST use print() to output the final result.
    IMPORTANT: Use the standard Python library ONLY. Do NOT import pandas, numpy, or other external libraries.
    If you do not use print(), you will receive no output!"""
    return _python_repl.run(code)


research_tools = [duckduckgo_tool, wikipedia_tool]
analysis_tools = [python_repl_tool]


def _handle_tool_error(e: Exception) -> str:
    return f"Tool execution failed: {type(e).__name__}: {e}"


# --- Nodes ---

def supervisor_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """The Supervisor reads the full history and decides what to do next."""
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
            "\n- If calculations or data analysis are needed, route to AnalysisAgent."
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
            "accurate facts relevant to the user's query. If you cannot find the required "
            "information after searching, state clearly that the data is unavailable."
        )
    )

    ai_msg = research_llm.invoke([sys_msg] + state["messages"])
    return {"messages": [ai_msg], "sender": "ResearchAgent"}


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

    ai_msg = analysis_llm.invoke([sys_msg] + state["messages"])
    return {"messages": [ai_msg], "sender": "AnalysisAgent"}


# --- Edge Logic ---

def research_condition(state: AgentState) -> Literal["ResearchTools", "Supervisor"]:
    """Route to tools if the ResearchAgent made a tool call, otherwise back to Supervisor."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "ResearchTools"
    return "Supervisor"


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

# Agent -> Tool -> Agent loops
builder.add_conditional_edges("ResearchAgent", research_condition)
builder.add_edge("ResearchTools", "ResearchAgent")

builder.add_conditional_edges("AnalysisAgent", analysis_condition)
builder.add_edge("AnalysisTools", "AnalysisAgent")

checkpointer = MemorySaver()
compiled_graph = builder.compile(checkpointer=checkpointer)


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
        "recursion_limit": 30,  # Increased slightly to account for the flattened tool node hops
    }

    return compiled_graph.invoke(initial_state, config=config)