"""LangChain's standard unit tests for tools."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_onlinecompiler import OnlineCompilerRunCode, OnlineCompilerRunTests


class TestOnlineCompilerRunCodeStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return OnlineCompilerRunCode

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"api_key": "test-key"}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"compiler": "python-3.14", "code": "print('hello')", "input": ""}

    @property
    def init_from_env_params(
        self,
    ) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return {"ONLINECOMPILER_API_KEY": "env-key"}, {}, {"api_key": "env-key"}


class TestOnlineCompilerRunTestsStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return OnlineCompilerRunTests

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"api_key": "test-key"}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {
            "compiler": "python-3.14",
            "code": "print(input()[::-1])",
            "tests": [{"input": "abc", "expected_output": "cba"}],
            "checker": "trim",
        }

    @property
    def init_from_env_params(
        self,
    ) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return {"ONLINECOMPILER_API_KEY": "env-key"}, {}, {"api_key": "env-key"}
