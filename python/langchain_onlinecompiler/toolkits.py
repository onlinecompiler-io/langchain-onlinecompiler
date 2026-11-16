"""Both OnlineCompiler.io tools, made from one set of settings."""

from __future__ import annotations

from typing import Self

from langchain_core.tools import BaseTool, BaseToolkit
from langchain_core.utils import secret_from_env
from pydantic import Field, SecretStr, model_validator

from langchain_onlinecompiler.tools import (
    API_KEY_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    NO_API_KEY,
    OnlineCompilerRunCode,
    OnlineCompilerRunTests,
)


class OnlineCompilerToolkit(BaseToolkit):
    """The OnlineCompiler.io tools: run code, and judge code against test cases.

    Setup:
        Set `ONLINECOMPILER_API_KEY` to a free key from
        https://api.onlinecompiler.io/compilers/api-keys/, or pass `api_key`.

    Instantiation:
        ```python
        from langchain_onlinecompiler import OnlineCompilerToolkit

        tools = OnlineCompilerToolkit().get_tools()
        ```

    Tools:
        `onlinecompiler_run_code` (`OnlineCompilerRunCode`) and
        `onlinecompiler_run_tests` (`OnlineCompilerRunTests`).
    """

    api_key: SecretStr | None = Field(
        default_factory=secret_from_env(API_KEY_ENV, default=None),
    )
    """An OnlineCompiler.io API key of type API. Defaults to `ONLINECOMPILER_API_KEY`."""

    base_url: str = DEFAULT_BASE_URL
    """Where the API is."""

    timeout: float = DEFAULT_TIMEOUT
    """Seconds to wait for an answer."""

    @model_validator(mode="after")
    def _require_api_key(self) -> Self:
        if self.api_key is None or not self.api_key.get_secret_value():
            raise ValueError(NO_API_KEY)
        return self

    def get_tools(self) -> list[BaseTool]:
        """Return the run-code tool and the run-tests tool."""
        return [
            OnlineCompilerRunCode(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
            ),
            OnlineCompilerRunTests(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
            ),
        ]
