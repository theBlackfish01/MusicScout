import operator
import os
from enum import Enum
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
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


class ToolTrace(TypedDict):
    agent: str
    tool: str
    input: str
    output: str


class AgentState(TypedDict):
    # add_messages reducer: node updates are appended, never overwrite
    messages: Annotated[list[BaseMessage], add_messages]
    next: str
    # operator.add reducer: node updates are concatenated
    tool_traces: Annotated[list[ToolTrace], operator.add]


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


# --- Tools ---

duckduckgo_tool: DuckDuckGoSearchRun = DuckDuckGoSearchRun()
wikipedia_tool: WikipediaQueryRun = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
_python_repl = PythonREPL()


@lc_tool
def python_repl_tool(code: str) -> str:
    """Execute Python code for arithmetic, statistics, and data analysis.
    Input must be valid, executable Python code."""
    return _python_repl.run(code)


research_tools = [duckduckgo_tool, wikipedia_tool]
analysis_tools = [python_repl_tool]
_research_tool_node = ToolNode(research_tools)
_analysis_tool_node = ToolNode(analysis_tools)

# --- LLM and chains ---

active_provider, llm = _build_llm()
supervisor_chain = llm.with_structured_output(SupervisorDecision)
research_llm = llm.bind_tools(research_tools)
analysis_llm = llm.bind_tools(analysis_tools)

SUPERVISOR_SYSTEM_PROMPT = """You are a routing supervisor. Based on the query and tool_traces, decide which agent to call next.
If the query requires searching for facts, names, or history, route to ResearchAgent.
If the query requires arithmetic, ranking, or data structuring, route to AnalysisAgent.
If mathematical calculation is required, you MUST route to AnalysisAgent before returning FINISH.
Return FINISH only when the final answer is complete and all required computations are done."""


# --- Helpers ---

def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return str(content)


def _original_user_query(state: AgentState) -> str:
    for message in state.get("messages", []):
        if isinstance(message, HumanMessage):
            return _message_text(message)
    messages = state.get("messages", [])
    return _message_text(messages[-1]) if messages else ""


# --- Node functions ---

_RESEARCH_SYSTEM_MSG = SystemMessage(
    content="You are a music research expert. Use the available search tools to find "
            "accurate facts, biographies, history, and current information relevant to "
            "the user's query. Always call a tool — do not answer from memory alone."
)

_ANALYSIS_SYSTEM_MSG = SystemMessage(
    content="You are a data analysis expert. Use the python_repl_tool to write and execute "
            "Python code for all calculations, aggregations, and statistical analysis. "
            "Always call the tool — do not compute in your head."
)


def research_node(state: AgentState) -> dict[str, Any]:
    """LLM autonomously selects and invokes research tools, then synthesises findings."""
    ai_msg = research_llm.invoke([_RESEARCH_SYSTEM_MSG] + state["messages"])
    new_messages: list[BaseMessage] = [ai_msg]
    new_traces: list[ToolTrace] = []

    if ai_msg.tool_calls:
        tool_msgs = _research_tool_node.invoke({"messages": [ai_msg]})["messages"]
        new_messages.extend(tool_msgs)
        for tc, tm in zip(ai_msg.tool_calls, tool_msgs):
            new_traces.append({
                "agent": "ResearchAgent",
                "tool": tc["name"],
                "input": str(tc["args"]),
                "output": str(tm.content),
            })
        # Synthesise findings with full conversation context
        synthesis = llm.invoke(state["messages"] + new_messages)
        new_messages.append(synthesis)

    return {"messages": new_messages, "tool_traces": new_traces}


def analysis_node(state: AgentState) -> dict[str, Any]:
    """LLM autonomously generates and executes Python code via the REPL tool."""
    ai_msg = analysis_llm.invoke([_ANALYSIS_SYSTEM_MSG] + state["messages"])
    new_messages: list[BaseMessage] = [ai_msg]
    new_traces: list[ToolTrace] = []

    if ai_msg.tool_calls:
        tool_msgs = _analysis_tool_node.invoke({"messages": [ai_msg]})["messages"]
        new_messages.extend(tool_msgs)
        for tc, tm in zip(ai_msg.tool_calls, tool_msgs):
            new_traces.append({
                "agent": "AnalysisAgent",
                "tool": tc["name"],
                "input": str(tc["args"]),
                "output": str(tm.content),
            })
        # Synthesise results with full conversation context
        synthesis = llm.invoke(state["messages"] + new_messages)
        new_messages.append(synthesis)

    return {"messages": new_messages, "tool_traces": new_traces}


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


# --- Graph assembly ---

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
