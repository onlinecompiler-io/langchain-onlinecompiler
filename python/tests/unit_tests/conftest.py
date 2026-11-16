"""A fake OnlineCompiler.io API on 127.0.0.1, so unit tests send real HTTP requests offline."""

from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest


@dataclass
class Received:
    method: str
    path: str
    headers: dict[str, str]
    body: Any


@dataclass
class Reply:
    status: int
    body: Any
    headers: dict[str, str] = field(default_factory=dict)


class FakeAPI:
    """Answers each request with the next queued reply and records what it received."""

    def __init__(self) -> None:
        self.received: list[Received] = []
        self.replies: deque[Reply] = deque()
        self.url = ""

    def reply(
        self,
        status: int = 200,
        body: Any = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.replies.append(Reply(status, body, headers or {}))

    @property
    def last(self) -> Received:
        return self.received[-1]


def _handler(api: FakeAPI) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            api.received.append(
                Received(
                    "POST",
                    self.path,
                    {key.lower(): value for key, value in self.headers.items()},
                    json.loads(raw) if raw else None,
                ),
            )
            reply = (
                api.replies.popleft() if api.replies else Reply(500, {"error": "no reply queued"})
            )
            data = (
                reply.body.encode()
                if isinstance(reply.body, str)
                else json.dumps(reply.body).encode()
            )
            self.send_response(reply.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for key, value in reply.headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

    return Handler


@pytest.fixture
def fake_api() -> Iterator[FakeAPI]:
    api = FakeAPI()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(api))
    api.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True)
    thread.start()
    try:
        yield api
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture(autouse=True)
def _no_key_in_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit tests never see a real key from the shell that runs them."""
    monkeypatch.delenv("ONLINECOMPILER_API_KEY", raising=False)


RUN_RESULT: dict[str, Any] = {
    "output": "desserts\n",
    "error": "",
    "status": "success",
    "exit_code": 0,
    "signal": None,
    "truncated": False,
    "time": "0.021",
    "total": "0.034",
    "memory": "9216",
    "files": [],
    "files_truncated": False,
    "time_limit": 30,
    "memory_limit": 512,
}

TESTS_RESULT: dict[str, Any] = {
    "verdict": "wrong_answer",
    "score": 33.33,
    "passed": 1,
    "total": 3,
    "time_limit": 2,
    "memory_limit": 512,
    "compile_error": None,
    "tests": [
        {
            "verdict": "accepted",
            "hidden": False,
            "time": "0.02",
            "memory": "9000",
            "exit_code": 0,
            "signal": None,
            "output": "5\n",
            "error": "",
            "truncated": False,
            "files": [],
            "files_truncated": False,
        },
        {
            "verdict": "wrong_answer",
            "hidden": False,
            "time": "0.02",
            "memory": "9000",
            "exit_code": 0,
            "signal": None,
            "output": "4\n",
            "error": "",
            "truncated": False,
            "files": [],
            "files_truncated": False,
        },
        {"verdict": "skipped", "hidden": True},
    ],
    "summary": "1 of 3 sample tests passed. Sample 2 expected 6 but the program printed 4.",
    "submission": "sub_abcdefghijklmnopqrstuvwx",
    "receipt": {"receipt": "rcpt_x", "signature": "sig"},
}
