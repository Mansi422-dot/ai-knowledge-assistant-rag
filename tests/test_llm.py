import asyncio

import httpx

from app import llm


class FakeResponse:
    def __init__(self, data=None, status_code=200, json_error=False):
        self.data = data
        self.status_code = status_code
        self.json_error = json_error

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://localhost/api/chat")
            response = httpx.Response(
                self.status_code,
                request=request,
            )
            raise httpx.HTTPStatusError(
                "HTTP error",
                request=request,
                response=response,
            )

    def json(self):
        if self.json_error:
            raise ValueError("Invalid JSON")

        return self.data


class FakeClient:
    def __init__(self, response=None, error=None, **kwargs):
        self.response = response
        self.error = error
        self.payload = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def post(self, url, json):
        self.payload = json

        if self.error:
            raise self.error

        return self.response


def run(coro):
    return asyncio.run(coro)


CONTEXTS = [
    {
        "document": "security.txt",
        "chunk": 0,
        "text": "Access control restricts unauthorized access.",
    }
]


def test_successful_answer(monkeypatch):
    response = FakeResponse(
        {"message": {"content": "Access control restricts access. [1]"}}
    )

    monkeypatch.setattr(
        llm.httpx,
        "AsyncClient",
        lambda **kwargs: FakeClient(response=response, **kwargs),
    )

    result = run(
        llm.generate_answer("What is access control?", CONTEXTS)
    )

    assert result == "Access control restricts access. [1]"


def test_empty_answer_returns_none(monkeypatch):
    response = FakeResponse(
        {"message": {"content": "   "}}
    )

    monkeypatch.setattr(
        llm.httpx,
        "AsyncClient",
        lambda **kwargs: FakeClient(response=response, **kwargs),
    )

    assert run(
        llm.generate_answer("Explain access control", CONTEXTS)
    ) is None


def test_malformed_response_returns_none(monkeypatch):
    response = FakeResponse({"unexpected": "structure"})

    monkeypatch.setattr(
        llm.httpx,
        "AsyncClient",
        lambda **kwargs: FakeClient(response=response, **kwargs),
    )

    assert run(
        llm.generate_answer("Explain access control", CONTEXTS)
    ) is None


def test_invalid_json_returns_none(monkeypatch):
    response = FakeResponse(json_error=True)

    monkeypatch.setattr(
        llm.httpx,
        "AsyncClient",
        lambda **kwargs: FakeClient(response=response, **kwargs),
    )

    assert run(
        llm.generate_answer("Explain access control", CONTEXTS)
    ) is None


def test_http_error_returns_none(monkeypatch):
    response = FakeResponse(status_code=500)

    monkeypatch.setattr(
        llm.httpx,
        "AsyncClient",
        lambda **kwargs: FakeClient(response=response, **kwargs),
    )

    assert run(
        llm.generate_answer("Explain access control", CONTEXTS)
    ) is None


def test_connection_error_returns_none(monkeypatch):
    monkeypatch.setattr(
        llm.httpx,
        "AsyncClient",
        lambda **kwargs: FakeClient(
            error=httpx.ConnectError("Connection failed"),
            **kwargs,
        ),
    )

    assert run(
        llm.generate_answer("Explain access control", CONTEXTS)
    ) is None


def test_empty_context_returns_none():
    assert run(
        llm.generate_answer("Explain access control", [])
    ) is None


def test_invalid_context_returns_none():
    assert run(
        llm.generate_answer(
            "Explain access control",
            [{"document": "security.txt"}],
        )
    ) is None


def test_context_is_encoded_as_json(monkeypatch):
    captured = {}

    class InspectClient(FakeClient):
        async def post(self, url, json):
            captured["payload"] = json
            return FakeResponse(
                {"message": {"content": "Access control. [1]"}}
            )

    monkeypatch.setattr(
        llm.httpx,
        "AsyncClient",
        lambda **kwargs: InspectClient(**kwargs),
    )

    run(
        llm.generate_answer(
            "Explain access control",
            CONTEXTS,
        )
    )

    user_message = captured["payload"]["messages"][1]["content"]

    assert '"document": "security.txt"' in user_message
    assert "USER QUESTION:" in user_message


def test_oversized_context_is_limited():
    large_context = [
        {
            "document": "large.txt",
            "chunk": 0,
            "text": "A" * (llm.MAX_CONTEXT_CHARS + 1000),
        }
    ]

    result = llm._prepare_context(large_context)

    assert result is not None
    assert len(result) < llm.MAX_CONTEXT_CHARS + 500
