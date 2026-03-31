import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import httpx
import openai
from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage, BaseMessage
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
    """No-op token-tracking context manager for providers without a LangChain callback."""
    yield SimpleNamespace(total_tokens=0, prompt_tokens=0, completion_tokens=0, total_cost=0.0)


try:
    from langchain_community.callbacks.manager import get_openai_callback
except ImportError:
    get_openai_callback = _null_callback


import dotenv
dotenv.load_dotenv()

class ExecuteRequest(BaseModel):
    query: str = Field(..., min_length=1)
    thread_id: str = Field(..., min_length=1)
    provider: str | None = Field(default=None, description="Optional LLM provider ('openai' or 'gemini')")
    model: str | None = Field(default=None, description="Optional model name (e.g., 'gpt-4o', 'gemini-1.5-pro')")


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
    if isinstance(message, BaseMessage):
        content = message.content
    elif isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = str(message)

    if isinstance(content, str):
        return content
    return str(content)


def _extract_answer(state: dict[str, Any]) -> str:
    messages = state.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return _message_to_text(message)
        if isinstance(message, dict) and message.get("type") == "ai":
            return _message_to_text(message)
    return ""


@app.post("/v1/execute", response_model=ExecuteResponse)
def execute(request: ExecuteRequest) -> ExecuteResponse:
    try:
        resolved_provider = resolve_provider(request.provider)
        cb_manager = get_openai_callback if resolved_provider == "openai" else _null_callback

        with cb_manager() as cb:
            final_state = invoke_graph(
                query=request.query, 
                thread_id=request.thread_id,
                provider=request.provider,
                model=request.model
            )
        answer = _extract_answer(final_state)
        return ExecuteResponse(
            answer=answer,
            total_tokens=int(getattr(cb, "total_tokens", 0) or 0),
            prompt_tokens=int(getattr(cb, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(cb, "completion_tokens", 0) or 0),
            total_cost_usd=float(getattr(cb, "total_cost", 0.0) or 0.0),
        )
    except EnvironmentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
        raise HTTPException(status_code=408, detail="Upstream model request timed out.") from exc
    except openai.APIStatusError as exc:
        status_code = getattr(exc, "status_code", None)
        if status_code is not None and int(status_code) >= 500:
            raise HTTPException(status_code=502, detail="Upstream model provider is unavailable.") from exc
        raise
    except Exception as exc:
        if _GOOGLE_SERVER_ERRORS and isinstance(exc, _GOOGLE_SERVER_ERRORS):
            raise HTTPException(status_code=502, detail="Upstream model provider is unavailable.") from exc
        raise

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")
