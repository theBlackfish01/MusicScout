import asyncio
import warnings
from contextlib import contextmanager

# Suppress the Wikipedia GuessedAtParserWarning
warnings.filterwarnings("ignore", "No parser was explicitly specified")

from types import SimpleNamespace
from typing import Any

import httpx
import openai
from fastapi import FastAPI, HTTPException
from langchain_core.messages import BaseMessage
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, Field

from src.graph import invoke_graph, resolve_provider

try:
    from google.api_core.exceptions import InternalServerError as _GInternalError
    from google.api_core.exceptions import ServiceUnavailable as _GUnavailable
    _GOOGLE_SERVER_ERRORS: tuple = (_GInternalError, _GUnavailable)
except ImportError:
    _GOOGLE_SERVER_ERRORS = ()


@contextmanager
def _null_callback():
    yield SimpleNamespace(total_tokens=0, prompt_tokens=0, completion_tokens=0, total_cost=0.0)


try:
    from langchain_community.callbacks.manager import get_openai_callback
except ImportError:
    get_openai_callback = _null_callback

import dotenv

dotenv.load_dotenv()

class TraceStep(BaseModel):
    tool: str
    input: dict[str, Any]
    output: str

class ExecuteResponse(BaseModel):
    answer: str
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    total_cost_usd: float
    trace_steps: list[TraceStep] = Field(default_factory=list) 

class ExecuteRequest(BaseModel):
    query: str = Field(..., min_length=1)
    thread_id: str = Field(..., min_length=1)
    provider: str | None = Field(default=None, description="Optional LLM provider ('openai' or 'gemini')")
    model: str | None = Field(default=None, description="Optional model name")


class ExecuteResponse(BaseModel):
    answer: str
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    total_cost_usd: float


class HealthResponse(BaseModel):
    status: str = "ok"


app = FastAPI()


def _message_to_text(message: Any) -> str:
    """Extract plain text from a message, handling both strings and Gemini's list content."""
    if isinstance(message, BaseMessage):
        content = message.content
    elif isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = str(message)

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(i if isinstance(i, str) else i.get("text", "") for i in content)
    return str(content)


def _extract_usage(state: dict[str, Any]) -> dict[str, int]:
    """Sum token usage from usage_metadata on messages (works for Gemini and others)."""
    usage = {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}
    for msg in state.get("messages", []):
        meta = None
        if isinstance(msg, BaseMessage):
            meta = getattr(msg, "usage_metadata", None)
        elif isinstance(msg, dict):
            meta = msg.get("usage_metadata")

        if isinstance(meta, dict):
            usage["total_tokens"] += meta.get("total_tokens", 0) or 0
            usage["input_tokens"] += meta.get("input_tokens", 0) or 0
            usage["output_tokens"] += meta.get("output_tokens", 0) or 0
    return usage


@app.post("/v1/execute", response_model=ExecuteResponse)
def execute(request: ExecuteRequest) -> ExecuteResponse:
    try:
        provider = resolve_provider(request.provider)
        cb_manager = get_openai_callback if provider == "openai" else _null_callback

        with cb_manager() as cb:
            final_state = invoke_graph(
                query=request.query,
                thread_id=request.thread_id,
                provider=request.provider,
                model=request.model
            )

        # Extract answer from the final message
        messages = final_state.get("messages", [])
        answer = _message_to_text(messages[-1]) if messages else ""

        trace_steps = []
        for i, msg in enumerate(messages):
            # Check if the message is an AIMessage with tool_calls
            tool_calls = getattr(msg, "tool_calls", [])
            if tool_calls:
                for tool_call in tool_calls:
                    tool_id = tool_call.get("id")
                    tool_output = ""
                    
                    # Look ahead for the corresponding ToolMessage by ID
                    for next_msg in messages[i + 1:]:
                        if getattr(next_msg, "type", "") == "tool" and getattr(next_msg, "tool_call_id", "") == tool_id:
                            tool_output = str(next_msg.content)
                            break
                            
                    trace_steps.append(
                        TraceStep(
                            tool=tool_call.get("name", ""),
                            input=tool_call.get("args", {}),
                            output=tool_output
                        )
                    )

        # Extract tokens (preferring callback, falling back to message metadata)
        total_tokens = int(getattr(cb, "total_tokens", 0) or 0)
        prompt_tokens = int(getattr(cb, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(cb, "completion_tokens", 0) or 0)
        total_cost = float(getattr(cb, "total_cost", 0.0) or 0.0)

        if total_tokens == 0:
            usage = _extract_usage(final_state)
            total_tokens = usage["total_tokens"]
            prompt_tokens = usage["input_tokens"]
            completion_tokens = usage["output_tokens"]

        return ExecuteResponse(
            answer=answer,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_cost_usd=total_cost,
            trace_steps=trace_steps,
        )

    except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
        raise HTTPException(status_code=408, detail="Upstream model request timed out.") from exc
    except EnvironmentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GraphRecursionError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "GRAPH_NON_CONVERGENCE", "reason": str(exc)},
        ) from exc
    except openai.APIStatusError as exc:
        if getattr(exc, "status_code", None) and int(exc.status_code) >= 500:
            raise HTTPException(status_code=502, detail="Upstream model provider is unavailable.") from exc
        raise
    except Exception as exc:
        if _GOOGLE_SERVER_ERRORS and isinstance(exc, _GOOGLE_SERVER_ERRORS):
            raise HTTPException(status_code=502, detail="Upstream model provider is unavailable.") from exc
        raise


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")