Technical Product Requirements Document (PRD)
Product: Music Discovery & Analysis Assistant (Multi-Agent System)
Author: Lead Systems Architect / Product Manager
Status: Ready for Implementation
1. Product Overview
Vision: To provide an intelligent, multi-agent orchestration service that seamlessly blends qualitative music research (history, theory, artist data) with quantitative computational analysis.
Target User: Music researchers, data-curious audiophiles, and industry analysts.
Problem Solved: Currently, users investigating complex musical queries must manually bounce between search engines to gather facts and code editors to crunch numbers (e.g., "Find the top 5 longest Pink Floyd songs and calculate their average length"). This system automates the workflow via a centralized, state-aware agentic supervisor.
2. System Architecture (LangGraph)
The core of the system is a StateGraph leveraging the Supervisor Pattern, ensuring deterministic routing and dynamic tool execution.
Shared State (AgentState): A typed dictionary tracking the global context. Contains messages (the full interaction history), next (the routing string), and tool_traces to explicitly store raw tool outputs for the Supervisor's context.
The Supervisor Node:
Role: The LLM-powered central router. It does not execute tools itself. It evaluates the AgentState against the user's initial query and makes a structural decision.
Routing Mechanism: To prevent routing hallucinations, the Supervisor's output will use LLM structured outputs (via with_structured_output or function calling) strictly constrained to a predefined Pydantic Enum: ['ResearchAgent', 'AnalysisAgent', 'FINISH'].
Worker Node 1: Research Agent ("The Musicologist"):
Role: Handles all qualitative, knowledge-based retrieval.
Tools: DuckDuckGoSearchRun (for recent news/trends) and WikipediaQueryRun (for historical facts, biographies, and genre origins).
Behavior: Executes tools autonomously, synthesizes the findings, appends the result to messages, and returns control to the Supervisor.
Worker Node 2: Analysis Agent ("The Data Analyst"):
Role: Handles computational tasks, aggregations, and data structuring.
Tools: PythonREPLTool.
Security Constraint: To mitigate Remote Code Execution (RCE) risks, the Python REPL will be strictly sandboxed. For this iteration, we will utilize the ast module to restrict imports and block system-level commands, or run the execution environment in an isolated ephemeral Docker container.
Behavior: Takes quantitative parameters passed from the Supervisor or Research Agent, executes the sandboxed Python code, and returns the computed outputs to the Supervisor.
3. Technical Specifications (FastAPI)
The LangGraph workflow will be exposed via a RESTful API built on FastAPI, designed for robustness, security, and observability.
Endpoint: POST /v1/execute
Input Schema (Pydantic):
JSON
{
  "query": "Who composed the Interstellar soundtrack, and what is the tempo in BPM? If it varies, calculate the average of the primary themes.",
  "thread_id": "uuid-v4-string"
}


Graph Execution Controls:
Circuit Breaker: The app.invoke() call will enforce a strict recursion_limit (e.g., 10 steps) to prevent infinite agent loops where nodes pass state back and forth without progressing.
Error Handling & API Exceptions:
The API will strictly define error schemas and return standard HTTP status codes:
408 Request Timeout: Triggered if the upstream LLM provider hangs or fails to respond within the designated window.
422 Unprocessable Entity: Triggered for malformed JSON inputs or missing required fields.
502 Bad Gateway: Triggered if the underlying model APIs (e.g., OpenAI, Anthropic) experience an outage.
Telemetry & Persistence:
Trace Logging: We will utilize LangSmith natively integrated into LangGraph via environment variables. This provides production-grade observability into tool inputs, outputs, and latency without brittle custom text files. (Fallback: If local logging is mandated, logs will be written in machine-readable JSON Lines (.jsonl) format).
Token Tracking: Utilize get_openai_callback() (or equivalent) wrapping the graph execution. The total cost and token count will be appended to the final API response payload.
State Persistence: Integrate LangGraph's MemorySaver checkpointer keyed to the thread_id to ensure the system can resume or reference past state across FastAPI worker restarts.
4. Evaluation & Optimization Strategy
The system will undergo rigorous stress testing, documented in EVAL.md.
Stress Test Protocol: Execution of 5 complex, multi-hop queries designed to trigger at least one full loop: User -> Supervisor -> Research Agent -> Supervisor -> Analysis Agent -> Supervisor -> Final Answer.
Metrics & Analysis for EVAL.md:
Quantitative Metrics:
Latency: Average end-to-end execution time per query.
Cost: Total token consumption and estimated cost per full graph run.
Reliability: Success rate (e.g., "5/5 queries returned an accurate final answer within the 10-step recursion limit").
Qualitative Analysis:
Agent Persona Justification: Documenting why separating search from computation reduces hallucination rates compared to a monolithic agent.
Routing Logic Diagnostics: Analyzing edge cases (e.g., if the Supervisor prematurely calls FINISH before data computation) and explaining how strict Enum routing mitigated this.
Prompt Evolution: Documenting changes to the Supervisor's system prompt to handle ambiguity (e.g., adding explicit rules like "If mathematical calculation is required, you MUST route to the Analysis Agent before returning FINISH").
5. Deliverables Roadmap
[ ] requirements.txt: Lock all dependencies (langgraph, langchain, fastapi, uvicorn, wikipedia, duckduckgo-search, langsmith).
[ ] graph.py: Define the AgentState TypedDict, initialize tools with sandboxing, build the Worker Nodes, define the structured output Supervisor routing logic, and compile the StateGraph with the MemorySaver checkpointer and recursion_limit.
[ ] main.py: Instantiate FastAPI, define Pydantic input/error schemas, inject the graph execution into POST /v1/execute, and configure LangSmith/token tracking wrappers.
[ ] EVAL.md: Run the 5 stress test queries, extract the LangSmith traces, and write the analytical report addressing the required qualitative and quantitative metrics.

