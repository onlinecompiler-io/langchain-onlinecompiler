"""LangChain tools that run and judge code with the OnlineCompiler.io API."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Literal, Self, TypeVar

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from langchain_core.utils import secret_from_env
from onlinecompiler import (
    AuthenticationError,
    Case,
    NotFoundError,
    OnlineCompiler,
    OnlineCompilerError,
    RateLimitError,
    RunResult,
    RunTestsResult,
)
from pydantic import BaseModel, Field, SecretStr, model_validator

DEFAULT_BASE_URL = "https://api.onlinecompiler.io"
# Long enough for a 30-second run, or a run-tests request (it ends within 60 seconds), plus
# queueing.
DEFAULT_TIMEOUT = 90.0
API_KEY_ENV = "ONLINECOMPILER_API_KEY"
SIGNUP_URL = "https://api.onlinecompiler.io/compilers/api-keys/?utm_source=langchain"
COMPILERS_URL = "https://onlinecompiler.io/docs/compilers"
NO_API_KEY = (
    f"No OnlineCompiler.io API key: pass api_key or set {API_KEY_ENV}. "
    f"Get a free key (100,000 runs a month) at {SIGNUP_URL}"
)

Checker = Literal["trim", "exact", "tokens", "float", "rows", "table", "table_ordered"]

COMPILER_DESCRIPTION = (
    "The compiler id, for example python-3.14, nodejs-24, typescript-deno, gcc-15, "
    "g++-15, openjdk-25, go-1.26, rust-1.93, dotnet-csharp-10, php-8.5, ruby-4.0, "
    f"bash-5.2 or sqlite-3. All 46 languages: {COMPILERS_URL}"
)
CODE_DESCRIPTION = "The whole program, as one source file."

# What an exit code means, for the exit codes the sandbox itself sets.
EXIT_CODE_MEANINGS = {
    124: "the time limit was reached",
    137: "killed at the memory limit",
    153: "output limit exceeded",
    71: "too many processes",
    -1: "the run failed on the OnlineCompiler.io side",
}


class RunCodeInput(BaseModel):
    """Arguments of the onlinecompiler_run_code tool."""

    compiler: str = Field(description=COMPILER_DESCRIPTION)
    code: str = Field(min_length=1, description=CODE_DESCRIPTION)
    input: str = Field(
        default="",
        description="Standard input for the program. Optional.",
    )
    time_limit: float | None = Field(
        default=None,
        gt=0,
        le=30,
        description="Seconds the program may run, above 0 and at most 30 (the default). Optional.",
    )


class TestCase(BaseModel):
    """One test case: the program's standard input and the output it should print."""

    __test__ = False  # not a pytest test class, despite the name

    input: str = Field(default="", description="Standard input for this test.")
    expected_output: str = Field(description="The output a correct program prints.")


class RunTestsInput(BaseModel):
    """Arguments of the onlinecompiler_run_tests tool."""

    compiler: str = Field(description=COMPILER_DESCRIPTION)
    code: str = Field(min_length=1, description=CODE_DESCRIPTION)
    tests: list[TestCase] = Field(
        min_length=1,
        description="The test cases, each with input and expected_output.",
    )
    checker: Checker | None = Field(
        default=None,
        description=(
            "How output is compared: trim ignores trailing whitespace (default), exact is "
            "byte for byte, tokens ignores all whitespace differences, float compares "
            "numbers within a small tolerance, rows ignores line order, table compares CSV "
            "tables with a header row by column name with rows in any order, and "
            "table_ordered does the same with rows in order. Optional."
        ),
    )


class OnlineCompilerToolError(ToolException):
    """The API refused or failed a request in a way the model can act on.

    A request that is malformed or too large (HTTP 400 or 413), a limit that was hit
    (429), a failure on the service's side (5xx) or no answer at all (status 0). The
    tools turn it into an error message for the model, unless they were made with
    `handle_tool_error=False`.
    """

    def __init__(self, message: str, status: int = 0) -> None:
        """Make the error from its message and HTTP status (0 when no response came)."""
        super().__init__(message)
        self.message = message
        self.status = status


class OnlineCompilerRateLimitError(OnlineCompilerToolError):
    """HTTP 429: a limit was reached, so nothing ran and nothing was counted.

    The message says which limit: runs at once, runs a minute, pending runs, busy
    sandboxes or the monthly allowance, and how to raise a plan's limit.
    """


T = TypeVar("T")


def _sentence(text: str) -> str:
    """The API's message as a sentence: with a full stop at the end."""
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else f"{text}."


def _call(base_url: str, request: Callable[[], T]) -> T:
    """Run one API request and turn its errors into the package's errors.

    An API key the service refuses (HTTP 401, 403 or 404) is raised as
    `onlinecompiler.AuthenticationError`, which is not a `ToolException`: the model
    cannot fix it, so it reaches the developer instead of the model.
    """
    try:
        return request()
    except AuthenticationError as exc:
        msg = (
            f"OnlineCompiler.io refused the API key (HTTP {exc.status}): "
            f"{_sentence(exc.message)} Pass api_key or set {API_KEY_ENV} to an active "
            f"key of type API from {SIGNUP_URL}"
        )
        raise AuthenticationError(msg, exc.status, exc.body) from None
    except NotFoundError as exc:
        msg = (
            f"{base_url} answered HTTP 404 ({exc.message}). Check base_url; the API is at "
            f"{DEFAULT_BASE_URL}"
        )
        raise NotFoundError(msg, exc.status, exc.body) from None
    except RateLimitError as exc:
        msg = (
            f"OnlineCompiler.io is at a limit (HTTP 429): {_sentence(exc.message)} "
            "Nothing ran and nothing was counted."
        )
        raise OnlineCompilerRateLimitError(msg, exc.status) from None
    except OnlineCompilerError as exc:
        raise OnlineCompilerToolError(_refusal(exc), exc.status) from None


def _refusal(exc: OnlineCompilerError) -> str:
    if exc.status == 0:
        return f"{_sentence(exc.message)} No result came back."
    if exc.status == 413:
        return (
            f"The request is too large (HTTP 413): {_sentence(exc.message)} "
            "Send a smaller input or fewer tests."
        )
    if exc.status == 400:
        hint = ""
        if "compiler" in exc.message.lower():
            hint = f" Use a compiler id from {COMPILERS_URL}, such as python-3.14."
        return f"OnlineCompiler.io refused the request (HTTP 400): {_sentence(exc.message)}{hint}"
    return (
        f"OnlineCompiler.io could not run the request (HTTP {exc.status}): {_sentence(exc.message)}"
    )


def _block(label: str, text: str) -> str:
    body = text.rstrip("\n")
    if not body:
        return f"{label}: (empty)"
    return f"{label}:\n{body}"


def _exit_code(code: int | None) -> str:
    meaning = EXIT_CODE_MEANINGS.get(code) if code is not None else None
    return f"{code} ({meaning})" if meaning else str(code)


def format_run_result(result: RunResult) -> str:
    """Write a run's result as the short text the model reads."""
    lines = [
        f"status: {result.get('status', '')}",
        f"exit_code: {_exit_code(result.get('exit_code'))}",
        f"time: {result.get('time', '')} s",
        f"memory: {result.get('memory', '')} KB",
        f"truncated: {str(bool(result.get('truncated'))).lower()}",
    ]
    for file in result.get("files") or []:
        where = file.get("url") or "content in the artifact"
        lines.append(
            f"file: {file.get('name')} ({file.get('mime')}, {file.get('size')} bytes) {where}",
        )
    lines.append(_block("output", result.get("output", "")))
    lines.append(_block("error", result.get("error", "")))
    return "\n".join(lines)


def format_tests_result(result: RunTestsResult) -> str:
    """Write a judged result as the short text the model reads.

    A hidden test shows only its verdict. A visible test that was not accepted also
    shows what the program printed.
    """
    lines = [
        f"verdict: {result.get('verdict', '')}",
        f"score: {result.get('score', 0):g}",
        f"passed: {result.get('passed', 0)} of {result.get('total', 0)}",
    ]
    if result.get("summary"):
        lines.append(f"summary: {result['summary']}")
    if result.get("verdict") == "incomplete":
        lines.append(
            "note: some tests were skipped by the request's time budget; send it again",
        )
    if result.get("compile_error"):
        lines.append(_block("compile_error", result["compile_error"] or ""))
    for number, test in enumerate(result.get("tests") or [], start=1):
        verdict = test.get("verdict", "")
        if test.get("hidden"):
            lines.append(f"test {number} (hidden): {verdict}")
            continue
        details = []
        if "time" in test:
            details.append(f"{test['time']} s")
        if verdict not in ("accepted", "skipped") and "exit_code" in test:
            details.append(f"exit_code {_exit_code(test['exit_code'])}")
        lines.append(f"test {number}: {verdict}" + (f" ({', '.join(details)})" if details else ""))
        if verdict in ("accepted", "skipped"):
            continue
        if "output" in test:
            lines.append(_block("  output", test.get("output", "")))
        if test.get("error"):
            lines.append(_block("  error", test.get("error", "")))
    return "\n".join(lines)


class _OnlineCompilerTool(BaseTool):
    """What both tools share: the API key, where the API is, and the error handling."""

    api_key: SecretStr | None = Field(
        default_factory=secret_from_env(API_KEY_ENV, default=None),
    )
    """An OnlineCompiler.io API key of type API. Defaults to `ONLINECOMPILER_API_KEY`."""

    base_url: str = DEFAULT_BASE_URL
    """Where the API is."""

    timeout: float = DEFAULT_TIMEOUT
    """Seconds to wait for an answer."""

    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    """The model reads a short text; the full JSON answer is the `ToolMessage` artifact."""

    @model_validator(mode="before")
    @classmethod
    def _handle_errors_by_default(cls, values: Any) -> Any:
        # A refusal the model can act on (a wrong compiler id, a limit) goes back to it as
        # an error message instead of stopping the agent. Pass handle_tool_error=False to
        # get OnlineCompilerToolError raised instead.
        if isinstance(values, dict) and "handle_tool_error" not in values:
            return {**values, "handle_tool_error": True}
        return values

    @model_validator(mode="after")
    def _require_api_key(self) -> Self:
        if self.api_key is None or not self.api_key.get_secret_value():
            raise ValueError(NO_API_KEY)
        return self

    def _client(self) -> OnlineCompiler:
        key = self.api_key.get_secret_value() if self.api_key else None
        return OnlineCompiler(key, base_url=self.base_url, timeout=self.timeout)


class OnlineCompilerRunCode(_OnlineCompilerTool):
    r"""Run a program in an OnlineCompiler.io sandbox.

    Each call is one run against your API key's allowance. The model gets the output,
    error, exit code, time, memory and whether output was cut; the full JSON answer is
    the `ToolMessage` artifact.

    Setup:
        Install `langchain-onlinecompiler` and set `ONLINECOMPILER_API_KEY` to a free
        key from https://api.onlinecompiler.io/compilers/api-keys/?utm_source=langchain.

        ```bash
        pip install -U langchain-onlinecompiler
        export ONLINECOMPILER_API_KEY="your-api-key"
        ```

    Instantiation:
        ```python
        from langchain_onlinecompiler import OnlineCompilerRunCode

        tool = OnlineCompilerRunCode()
        ```

    Invocation with args:
        ```python
        tool.invoke({"compiler": "python-3.14", "code": "print(2 ** 100)"})
        ```

        ```text
        status: success
        exit_code: 0
        time: 0.0309 s
        memory: 13296 KB
        truncated: false
        output:
        1267650600228229401496703205376
        error: (empty)
        ```

    Invocation with ToolCall:
        ```python
        tool.invoke(
            {
                "args": {"compiler": "python-3.14", "code": "print(2 ** 100)"},
                "id": "1",
                "name": tool.name,
                "type": "tool_call",
            }
        )
        ```

        Returns a `ToolMessage` whose content is the text above and whose artifact is
        the API's JSON answer.
    """

    name: str = "onlinecompiler_run_code"
    description: str = (
        "Run a program in a fresh OnlineCompiler.io sandbox and get its standard output, "
        "standard error, exit code, time and memory. 46 languages; no network access; up "
        "to 30 seconds and 512 MB per run. Use it to compute answers or to check code "
        "instead of guessing. Exit code 124 means the time limit was reached and 137 the "
        "memory limit. Files the program writes to $OUTPUT_DIR come back as links."
    )
    args_schema: type[BaseModel] = RunCodeInput

    def _execute(
        self,
        compiler: str,
        code: str,
        input: str,  # noqa: A002
        time_limit: float | None,
    ) -> tuple[str, RunResult]:
        client = self._client()
        result = _call(
            self.base_url,
            lambda: client.run(compiler, code, input, time_limit=time_limit),
        )
        return format_run_result(result), result

    def _run(
        self,
        compiler: str,
        code: str,
        input: str = "",  # noqa: A002
        time_limit: float | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, RunResult]:
        """Run the program and wait for its result."""
        return self._execute(compiler, code, input, time_limit)

    async def _arun(
        self,
        compiler: str,
        code: str,
        input: str = "",  # noqa: A002
        time_limit: float | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> tuple[str, RunResult]:
        """Run the program without blocking the event loop."""
        return await asyncio.to_thread(self._execute, compiler, code, input, time_limit)


class OnlineCompilerRunTests(_OnlineCompilerTool):
    r"""Judge a program against test cases with OnlineCompiler.io.

    Each test runs in its own fresh sandbox and counts as one run against your API key's
    allowance. The model gets the overall verdict, the score, a plain-words summary and
    a verdict per test (hidden tests show only their verdict); the full JSON answer,
    with its signed receipt, is the `ToolMessage` artifact.

    Setup:
        Install `langchain-onlinecompiler` and set `ONLINECOMPILER_API_KEY` to a free
        key from https://api.onlinecompiler.io/compilers/api-keys/?utm_source=langchain.

    Instantiation:
        ```python
        from langchain_onlinecompiler import OnlineCompilerRunTests

        tool = OnlineCompilerRunTests()
        ```

    Invocation with args:
        ```python
        tool.invoke(
            {
                "compiler": "python-3.14",
                "code": "print(sum(map(int, input().split())))",
                "tests": [
                    {"input": "2 3", "expected_output": "5"},
                    {"input": "10 -4", "expected_output": "6"},
                ],
            }
        )
        ```

        ```text
        verdict: accepted
        score: 100
        passed: 2 of 2
        summary: 2 of 2 sample tests passed.
        test 1: accepted (0.0379 s)
        test 2: accepted (0.0387 s)
        ```
    """

    name: str = "onlinecompiler_run_tests"
    description: str = (
        "Judge a program against test cases in fresh OnlineCompiler.io sandboxes: each "
        "test gives the program its input on standard input and compares what it prints "
        "with expected_output. Returns an overall verdict (accepted, wrong_answer, "
        "time_limit_exceeded, runtime_error, compilation_error and others), a score from 0 "
        "to 100 and a verdict per test. 46 languages; no network access; each test may "
        "run for 2 seconds and counts as one run."
    )
    args_schema: type[BaseModel] = RunTestsInput

    def _execute(
        self,
        compiler: str,
        code: str,
        tests: list[TestCase] | list[dict[str, Any]],
        checker: Checker | None,
    ) -> tuple[str, RunTestsResult]:
        cases = [_case(test) for test in tests]
        client = self._client()
        result = _call(
            self.base_url,
            lambda: client.run_tests(compiler, code, cases, checker=checker),
        )
        return format_tests_result(result), result

    def _run(
        self,
        compiler: str,
        code: str,
        tests: list[TestCase] | list[dict[str, Any]],
        checker: Checker | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, RunTestsResult]:
        """Judge the program and wait for the verdicts."""
        return self._execute(compiler, code, tests, checker)

    async def _arun(
        self,
        compiler: str,
        code: str,
        tests: list[TestCase] | list[dict[str, Any]],
        checker: Checker | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> tuple[str, RunTestsResult]:
        """Judge the program without blocking the event loop."""
        return await asyncio.to_thread(self._execute, compiler, code, tests, checker)


def _case(test: TestCase | dict[str, Any]) -> Case:
    if isinstance(test, TestCase):
        return {"input": test.input, "expected_output": test.expected_output}
    return {"input": test.get("input", ""), "expected_output": test["expected_output"]}
