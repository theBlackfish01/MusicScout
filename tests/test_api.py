from contextlib import contextmanager

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import main


class DummyCallback:
    total_tokens = 42
    prompt_tokens = 30
    completion_tokens = 12
    total_cost = 0.00123


@contextmanager
def _dummy_openai_callback():
    yield DummyCallback()


def test_execute_returns_200_with_valid_payload(monkeypatch):
    monkeypatch.setattr(
        main,
        "invoke_graph",
        lambda query, thread_id: {
            "messages": [AIMessage(content=f"Answer for: {query}")],
            "next": "FINISH",
            "tool_traces": [],
        },
    )
    monkeypatch.setattr(main, "get_llm_callback", _dummy_openai_callback)

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
