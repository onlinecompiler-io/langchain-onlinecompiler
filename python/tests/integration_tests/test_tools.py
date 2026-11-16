"""Live tests against api.onlinecompiler.io. They run only when ONLINECOMPILER_API_KEY is set.

A full run uses about 20 runs of the key's allowance: 4 for the run-code standard tests, 8 for the
run-tests standard tests (2 tests each) and the rest below.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_onlinecompiler import (
    AuthenticationError,
    OnlineCompilerRunCode,
    OnlineCompilerRunTests,
    OnlineCompilerToolkit,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("ONLINECOMPILER_API_KEY"),
    reason="set ONLINECOMPILER_API_KEY to run the live tests",
)

REVERSE = "print(input()[::-1])"


class TestOnlineCompilerRunCodeIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return OnlineCompilerRunCode

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"compiler": "python-3.14", "code": REVERSE, "input": "stressed"}


class TestOnlineCompilerRunTestsIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return OnlineCompilerRunTests

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {
            "compiler": "python-3.14",
            "code": REVERSE,
            "tests": [
                {"input": "stressed", "expected_output": "desserts"},
                {"input": "drawer", "expected_output": "reward"},
            ],
        }


def test_run_code_live() -> None:
    tool = OnlineCompilerRunCode()
    message = tool.invoke(
        {
            "args": {"compiler": "python-3.14", "code": REVERSE, "input": "stressed"},
            "id": "1",
            "name": tool.name,
            "type": "tool_call",
        },
    )
    assert isinstance(message, ToolMessage)
    assert message.status == "success"
    assert "output:\ndesserts" in message.content
    assert message.artifact["exit_code"] == 0
    assert message.artifact["output"] == "desserts\n"


async def test_run_tests_live_wrong_answer() -> None:
    tool = OnlineCompilerToolkit().get_tools()[1]
    message = await tool.ainvoke(
        {
            "args": {
                "compiler": "python-3.14",
                "code": "a, b = map(int, input().split())\nprint(a - b)",
                "tests": [
                    {"input": "5 0", "expected_output": "5"},
                    {"input": "2 3", "expected_output": "5"},
                ],
            },
            "id": "2",
            "name": tool.name,
            "type": "tool_call",
        },
    )
    assert message.status == "success"
    assert message.content.startswith("verdict: wrong_answer\n")
    assert "test 1: accepted" in message.content
    assert "test 2: wrong_answer" in message.content
    assert message.artifact["verdict"] == "wrong_answer"
    assert message.artifact["submission"].startswith("sub_")


def test_unknown_compiler_goes_back_to_the_model() -> None:
    tool = OnlineCompilerRunCode()
    message = tool.invoke(
        {
            "args": {"compiler": "python-2.7", "code": "print 1"},
            "id": "3",
            "name": tool.name,
            "type": "tool_call",
        },
    )
    assert message.status == "error"
    assert "HTTP 400" in message.content
    assert "https://onlinecompiler.io/docs/compilers" in message.content


def test_refused_key_is_raised_live() -> None:
    tool = OnlineCompilerRunCode(api_key="not-a-real-key")
    with pytest.raises(AuthenticationError) as info:
        tool.invoke({"compiler": "python-3.14", "code": "print(1)"})
    assert info.value.status == 404
    assert "Invalid or inactive API key" in str(info.value)
