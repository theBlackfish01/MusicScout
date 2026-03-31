from contextlib import contextmanager

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import asyncio
import httpx
import openai

from src import main


class DummyCallback:
    total_tokens = 42
    prompt_tokens = 30
    completion_tokens = 12
    total_cost = 0.00123


@contextmanager
def _dummy_openai_callback():
    yield DummyCallback()


def test_execute_returns_200_with_valid_payload(monkeypatch):
    monkeypatch.setattr(main, "resolve_provider", lambda p: "openai")
    monkeypatch.setattr(
        main,
        "invoke_graph",
        lambda query, thread_id, provider=None, model=None: {
            "messages": [AIMessage(content=f"Answer for: {query}")],
            "next": "FINISH",
            "tool_traces": [],
        },
    )
    monkeypatch.setattr(main, "get_openai_callback", _dummy_openai_callback)

    client = TestClient(main.app)
    response = client.post(
        "/v1/execute",
        json={"query": "Who composed Interstellar soundtrack?", "thread_id": "thread-1"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("Answer for:")
    assert body["total_tokens"] == 42
    assert body["prompt_tokens"] == 30
    assert body["completion_tokens"] == 12
    assert body["total_cost_usd"] == 0.00123


def test_execute_returns_422_when_query_missing():
    client = TestClient(main.app)
    response = client.post("/v1/execute", json={"thread_id": "thread-1"})

    assert response.status_code == 422


def test_execute_returns_422_when_thread_id_missing():
    client = TestClient(main.app)
    response = client.post(
        "/v1/execute", json={"query": "Who composed Interstellar soundtrack?"}
    )

    assert response.status_code == 422


def test_execute_returns_408_on_timeout(monkeypatch):
    monkeypatch.setattr(main, "resolve_provider", lambda p: "openai")
    # Simulate an upstream timeout from the LLM provider
    def mock_invoke_timeout(query, thread_id, provider=None, model=None):
        raise asyncio.TimeoutError("Timeout")

    monkeypatch.setattr(main, "invoke_graph", mock_invoke_timeout)
    monkeypatch.setattr(main, "get_openai_callback", _dummy_openai_callback)

    client = TestClient(main.app)
    response = client.post(
        "/v1/execute", json={"query": "Test timeout", "thread_id": "thread-1"}
    )

    assert response.status_code == 408
    assert response.json()["detail"] == "Upstream model request timed out."


def test_execute_returns_502_on_provider_outage(monkeypatch):
    monkeypatch.setattr(main, "resolve_provider", lambda p: "openai")
    # Simulate an OpenAI 500+ internal server error
    def mock_invoke_outage(query, thread_id, provider=None, model=None):
        # Constructing the expected openai error format
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        response = httpx.Response(503, request=request)
        raise openai.APIStatusError(
            "Service Unavailable",
            response=response,
            body=None
        )

    monkeypatch.setattr(main, "invoke_graph", mock_invoke_outage)
    monkeypatch.setattr(main, "get_openai_callback", _dummy_openai_callback)

    client = TestClient(main.app)
    response = client.post(
        "/v1/execute", json={"query": "Test outage", "thread_id": "thread-1"}
    )

    assert response.status_code == 502
    assert response.json()["detail"] == "Upstream model provider is unavailable."
