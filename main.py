import asyncio
from typing import Any

import httpx
import openai
from fastapi import FastAPI, HTTPException
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel, Field

from graph import invoke_graph


import dotenv
dotenv.load_dotenv()

class ExecuteRequest(BaseModel):
    query: str = Field(..., min_length=1)
    thread_id: str = Field(..., min_length=1)


class ExecuteResponse(BaseModel):
    answer: str
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    total_cost_usd: float


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
        with get_openai_callback() as cb:
            final_state = invoke_graph(query=request.query, thread_id=request.thread_id)
        answer = _extract_answer(final_state)
        return ExecuteResponse(
            answer=answer,
            total_tokens=int(getattr(cb, "total_tokens", 0) or 0),
            prompt_tokens=int(getattr(cb, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(cb, "completion_tokens", 0) or 0),
            total_cost_usd=float(getattr(cb, "total_cost", 0.0) or 0.0),
        )
    except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
        raise HTTPException(status_code=408, detail="Upstream model request timed out.") from exc
    except openai.APIStatusError as exc:
        status_code = getattr(exc, "status_code", None)
        if status_code is not None and int(status_code) >= 500:
            raise HTTPException(status_code=502, detail="Upstream model provider is unavailable.") from exc
        raise


# TODO: Add non-OpenAI callback fallback if model provider changes in future tickets.
