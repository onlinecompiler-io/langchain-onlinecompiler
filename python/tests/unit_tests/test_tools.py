"""Unit tests for the tools, against a fake API on 127.0.0.1."""

from __future__ import annotations

import copy
import socket
from typing import Any

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from pydantic import ValidationError

from langchain_onlinecompiler import (
    AuthenticationError,
    OnlineCompilerRateLimitError,
    OnlineCompilerRunCode,
    OnlineCompilerRunTests,
    OnlineCompilerToolError,
    OnlineCompilerToolkit,
)
from tests.unit_tests.conftest import RUN_RESULT, TESTS_RESULT, FakeAPI

KEY = "test-key-123"


def run_code(api: FakeAPI, **kwargs: object) -> OnlineCompilerRunCode:
    return OnlineCompilerRunCode(api_key=KEY, base_url=api.url, **kwargs)


def run_tests(api: FakeAPI, **kwargs: object) -> OnlineCompilerRunTests:
    return OnlineCompilerRunTests(api_key=KEY, base_url=api.url, **kwargs)


def schema(tool: BaseTool) -> dict[str, Any]:
    return tool.tool_call_schema.model_json_schema()  # type: ignore[union-attr]


def tool_call(tool: object, args: dict) -> dict:
    return {"args": args, "id": "call_1", "name": tool.name, "type": "tool_call"}  # type: ignore[attr-defined]


# Names and schemas


def test_tool_names_and_argument_names() -> None:
    code = OnlineCompilerRunCode(api_key=KEY)
    tests = OnlineCompilerRunTests(api_key=KEY)
    assert code.name == "onlinecompiler_run_code"
    assert tests.name == "onlinecompiler_run_tests"
    code_schema = schema(code)
    assert list(code_schema["properties"]) == [
        "compiler",
        "code",
        "input",
        "time_limit",
    ]
    assert code_schema["required"] == ["compiler", "code"]
    tests_schema = schema(tests)
    assert list(tests_schema["properties"]) == ["compiler", "code", "tests", "checker"]
    assert tests_schema["required"] == ["compiler", "code", "tests"]
    case = tests_schema["$defs"]["TestCase"]
    assert list(case["properties"]) == ["input", "expected_output"]
    assert case["required"] == ["expected_output"]
    checker = tests_schema["properties"]["checker"]["anyOf"][0]
    assert checker["enum"] == [
        "trim",
        "exact",
        "tokens",
        "float",
        "rows",
        "table",
        "table_ordered",
    ]


def test_descriptions_tell_the_model_the_limits() -> None:
    description = OnlineCompilerRunCode(api_key=KEY).description
    assert "no network" in description
    assert "46 languages" in description
    properties = schema(OnlineCompilerRunCode(api_key=KEY))["properties"]
    assert "python-3.14" in properties["compiler"]["description"]


def test_time_limit_is_checked_before_any_request() -> None:
    tool = OnlineCompilerRunCode(api_key=KEY, base_url="http://127.0.0.1:9")
    with pytest.raises(ValidationError):
        tool.invoke({"compiler": "python-3.14", "code": "print(1)", "time_limit": 31})


# API key


def test_key_from_argument_is_secret() -> None:
    tool = OnlineCompilerRunCode(api_key=KEY)
    assert tool.api_key is not None
    assert tool.api_key.get_secret_value() == KEY
    assert KEY not in repr(tool)
    assert KEY not in str(tool.model_dump())


def test_key_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONLINECOMPILER_API_KEY", "from-env")
    for tool in (OnlineCompilerRunCode(), OnlineCompilerRunTests()):
        assert tool.api_key is not None
        assert tool.api_key.get_secret_value() == "from-env"


def test_missing_key_says_where_to_get_one() -> None:
    for cls in (OnlineCompilerRunCode, OnlineCompilerRunTests, OnlineCompilerToolkit):
        with pytest.raises(ValidationError, match="ONLINECOMPILER_API_KEY") as info:
            cls()
        assert "https://api.onlinecompiler.io/compilers/api-keys/" in str(info.value)


# Run code


def test_run_code_request(fake_api: FakeAPI) -> None:
    fake_api.reply(200, RUN_RESULT)
    text = run_code(fake_api).invoke(
        {
            "compiler": "python-3.14",
            "code": "print(input()[::-1])",
            "input": "stressed",
        },
    )
    request = fake_api.last
    assert request.path == "/api/run-code-sync/"
    assert request.headers["authorization"] == KEY  # the bare key, no Bearer prefix
    assert request.headers["content-type"] == "application/json"
    assert request.body == {
        "compiler": "python-3.14",
        "code": "print(input()[::-1])",
        "input": "stressed",
    }
    assert text == (
        "status: success\n"
        "exit_code: 0\n"
        "time: 0.021 s\n"
        "memory: 9216 KB\n"
        "truncated: false\n"
        "output:\n"
        "desserts\n"
        "error: (empty)"
    )


def test_run_code_sends_time_limit_only_when_given(fake_api: FakeAPI) -> None:
    fake_api.reply(200, RUN_RESULT)
    fake_api.reply(200, RUN_RESULT)
    tool = run_code(fake_api)
    tool.invoke({"compiler": "python-3.14", "code": "print(1)"})
    assert fake_api.last.body == {
        "compiler": "python-3.14",
        "code": "print(1)",
        "input": "",
    }
    tool.invoke({"compiler": "python-3.14", "code": "print(1)", "time_limit": 2.5})
    assert fake_api.last.body["time_limit"] == 2.5


def test_run_code_tool_call_has_full_json_artifact(fake_api: FakeAPI) -> None:
    fake_api.reply(200, RUN_RESULT)
    tool = run_code(fake_api)
    message = tool.invoke(tool_call(tool, {"compiler": "python-3.14", "code": "x"}))
    assert isinstance(message, ToolMessage)
    assert message.status == "success"
    assert message.artifact == RUN_RESULT
    assert str(message.content).startswith("status: success")


def test_run_code_explains_exit_codes_and_lists_files(fake_api: FakeAPI) -> None:
    result = {
        **RUN_RESULT,
        "status": "error",
        "exit_code": 124,
        "output": "",
        "error": "Traceback\n",
        "truncated": True,
        "files": [
            {
                "name": "plot.png",
                "mime": "image/png",
                "size": 1234,
                "content_base64": "iVBORw0KGgo=",
                "url": "https://api.onlinecompiler.io/api/files/f1/",
            },
        ],
    }
    fake_api.reply(200, result)
    text = run_code(fake_api).invoke({"compiler": "python-3.14", "code": "x"})
    assert "exit_code: 124 (the time limit was reached)" in text
    assert "truncated: true" in text
    assert (
        "file: plot.png (image/png, 1234 bytes) https://api.onlinecompiler.io/api/files/f1/" in text
    )
    assert "iVBORw0KGgo=" not in text  # file bytes stay in the artifact
    assert "output: (empty)" in text
    assert "error:\nTraceback" in text


async def test_run_code_async(fake_api: FakeAPI) -> None:
    fake_api.reply(200, RUN_RESULT)
    tool = run_code(fake_api)
    message = await tool.ainvoke(tool_call(tool, {"compiler": "python-3.14", "code": "x"}))
    assert message.artifact == RUN_RESULT
    assert "desserts" in message.content
    assert fake_api.last.path == "/api/run-code-sync/"


# Run tests


TESTS_ARGS = {
    "compiler": "python-3.14",
    "code": "print(sum(map(int, input().split())))",
    "tests": [
        {"input": "2 3", "expected_output": "5"},
        {"input": "2 4", "expected_output": "6"},
        {"input": "1 1", "expected_output": "2"},
    ],
}


def test_run_tests_request_and_text(fake_api: FakeAPI) -> None:
    fake_api.reply(200, TESTS_RESULT)
    text = run_tests(fake_api).invoke({**TESTS_ARGS, "checker": "tokens"})
    request = fake_api.last
    assert request.path == "/api/run-tests/"
    assert request.headers["authorization"] == KEY
    assert request.body == {
        "compiler": "python-3.14",
        "code": "print(sum(map(int, input().split())))",
        "tests": [
            {"input": "2 3", "expected_output": "5"},
            {"input": "2 4", "expected_output": "6"},
            {"input": "1 1", "expected_output": "2"},
        ],
        "checker": "tokens",
    }
    assert text == (
        "verdict: wrong_answer\n"
        "score: 33.33\n"
        "passed: 1 of 3\n"
        "summary: 1 of 3 sample tests passed. Sample 2 expected 6 but the program printed 4.\n"
        "test 1: accepted (0.02 s)\n"
        "test 2: wrong_answer (0.02 s, exit_code 0)\n"
        "  output:\n"
        "4\n"
        "test 3 (hidden): skipped"
    )


def test_run_tests_leaves_out_checker_and_input_defaults(fake_api: FakeAPI) -> None:
    fake_api.reply(200, TESTS_RESULT)
    run_tests(fake_api).invoke(
        {
            "compiler": "python-3.14",
            "code": "print(1)",
            "tests": [{"expected_output": "1"}],
        },
    )
    assert fake_api.last.body == {
        "compiler": "python-3.14",
        "code": "print(1)",
        "tests": [{"input": "", "expected_output": "1"}],
    }


def test_hidden_tests_show_only_their_verdict(fake_api: FakeAPI) -> None:
    result = copy.deepcopy(TESTS_RESULT)
    # Even if an answer carried more for a hidden test, the model sees only the verdict.
    result["tests"][2] = {"verdict": "wrong_answer", "hidden": True, "output": "SECRET"}
    fake_api.reply(200, result)
    text = run_tests(fake_api).invoke(TESTS_ARGS)
    assert "test 3 (hidden): wrong_answer" in text
    assert "SECRET" not in text


def test_run_tests_shows_compile_error(fake_api: FakeAPI) -> None:
    result = {
        **TESTS_RESULT,
        "verdict": "compilation_error",
        "score": 0,
        "passed": 0,
        "compile_error": "main.c:1: error: expected ';'",
        "tests": [
            {"verdict": "compilation_error", "hidden": False},
            {"verdict": "skipped", "hidden": False},
        ],
    }
    fake_api.reply(200, result)
    text = run_tests(fake_api).invoke(TESTS_ARGS)
    assert "score: 0\n" in text
    assert "compile_error:\nmain.c:1: error: expected ';'" in text
    assert "test 2: skipped" in text


def test_run_tests_artifact_keeps_receipt(fake_api: FakeAPI) -> None:
    fake_api.reply(200, TESTS_RESULT)
    tool = run_tests(fake_api)
    message = tool.invoke(tool_call(tool, TESTS_ARGS))
    assert message.artifact == TESTS_RESULT
    assert "rcpt_x" not in message.content


async def test_run_tests_async(fake_api: FakeAPI) -> None:
    fake_api.reply(200, TESTS_RESULT)
    text = await run_tests(fake_api).ainvoke(TESTS_ARGS)
    assert text.startswith("verdict: wrong_answer")


# Errors the model can act on: returned to it as an error ToolMessage


@pytest.mark.parametrize(
    ("status", "error", "expected"),
    [
        (
            400,
            "Unsupported or missing compiler",
            "OnlineCompiler.io refused the request (HTTP 400): Unsupported or missing compiler. "
            "Use a compiler id from https://onlinecompiler.io/docs/compilers, such as python-3.14.",
        ),
        (
            413,
            "The request body is over 32 MB, the most one request may carry.",
            "The request is too large (HTTP 413): The request body is over 32 MB",
        ),
        (
            429,
            "Monthly submission limit reached (100000).",
            "OnlineCompiler.io is at a limit (HTTP 429): Monthly submission limit reached "
            "(100000). Nothing ran and nothing was counted.",
        ),
        (
            503,
            "Execution queue unavailable",
            "OnlineCompiler.io could not run the request (HTTP 503): Execution queue unavailable.",
        ),
    ],
)
def test_refusals_go_back_to_the_model(
    fake_api: FakeAPI, status: int, error: str, expected: str
) -> None:
    fake_api.reply(status, {"error": error})
    tool = run_code(fake_api)
    message = tool.invoke(tool_call(tool, {"compiler": "python-3.14", "code": "x"}))
    assert message.status == "error"
    assert expected in message.content
    assert message.artifact is None


def test_rate_limit_raises_when_errors_are_not_handled(fake_api: FakeAPI) -> None:
    fake_api.reply(429, {"error": "Too many concurrent requests. Please try again shortly."})
    tool = run_tests(fake_api, handle_tool_error=False)
    with pytest.raises(OnlineCompilerRateLimitError) as info:
        tool.invoke(TESTS_ARGS)
    assert info.value.status == 429
    assert "Please try again shortly. Nothing ran" in str(info.value)


def test_bad_request_raises_when_errors_are_not_handled(fake_api: FakeAPI) -> None:
    fake_api.reply(
        400,
        {"error": "tests[0].expected_output must be UTF-8 text without NUL, at most 64KB"},
    )
    with pytest.raises(OnlineCompilerToolError) as info:
        run_tests(fake_api, handle_tool_error=False).invoke(TESTS_ARGS)
    assert info.value.status == 400


def test_no_answer_goes_back_to_the_model() -> None:
    with socket.socket() as sock:  # a port that nothing listens on
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    tool = OnlineCompilerRunCode(api_key=KEY, base_url=f"http://127.0.0.1:{port}")
    message = tool.invoke(tool_call(tool, {"compiler": "python-3.14", "code": "x"}))
    assert message.status == "error"
    assert "Could not reach" in message.content
    assert "No result came back." in message.content


# A refused API key: raised for the developer, never handed to the model


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (404, "Invalid or inactive API key"),
        (403, "This key is not authorized for the REST API"),
        (403, "Request origin not allowed for this API key"),
        (401, "Unauthorized"),
    ],
)
def test_refused_key_is_raised(fake_api: FakeAPI, status: int, error: str) -> None:
    fake_api.reply(status, {"error": error})
    tool = run_code(fake_api)
    with pytest.raises(AuthenticationError) as info:
        tool.invoke(tool_call(tool, {"compiler": "python-3.14", "code": "x"}))
    assert info.value.status == status
    assert error in str(info.value)
    assert "https://api.onlinecompiler.io/compilers/api-keys/" in str(info.value)
    assert KEY not in str(info.value)


async def test_refused_key_is_raised_async(fake_api: FakeAPI) -> None:
    fake_api.reply(404, {"error": "Invalid or inactive API key"})
    with pytest.raises(AuthenticationError):
        await run_tests(fake_api).ainvoke(TESTS_ARGS)


# Toolkit


def test_toolkit_gives_both_tools_with_one_key(fake_api: FakeAPI) -> None:
    tools = OnlineCompilerToolkit(api_key=KEY, base_url=fake_api.url, timeout=5).get_tools()
    assert [tool.name for tool in tools] == [
        "onlinecompiler_run_code",
        "onlinecompiler_run_tests",
    ]
    assert all(tool.api_key.get_secret_value() == KEY for tool in tools)  # type: ignore[attr-defined]
    assert all(tool.timeout == 5 for tool in tools)  # type: ignore[attr-defined]
    fake_api.reply(200, RUN_RESULT)
    assert "desserts" in tools[0].invoke({"compiler": "python-3.14", "code": "x"})


def test_toolkit_key_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONLINECOMPILER_API_KEY", "from-env")
    tools = OnlineCompilerToolkit().get_tools()
    assert all(tool.api_key.get_secret_value() == "from-env" for tool in tools)  # type: ignore[attr-defined]
