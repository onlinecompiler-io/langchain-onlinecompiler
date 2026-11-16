"""The tools inside a LangChain agent (create_agent), with a scripted model and the fake API."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pytest
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from langchain_onlinecompiler import AuthenticationError, OnlineCompilerToolkit
from tests.unit_tests.conftest import RUN_RESULT, FakeAPI

agents = pytest.importorskip("langchain.agents")


class ScriptedModel(GenericFakeChatModel):
    """Answers with the scripted messages in turn, and accepts any tools."""

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self


def agent_for(api: FakeAPI, tool_args: dict[str, Any]) -> Any:
    model = ScriptedModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "onlinecompiler_run_code",
                            "args": tool_args,
                            "id": "call_1",
                            "type": "tool_call",
                        },
                    ],
                ),
                AIMessage(content="The program printed desserts."),
            ],
        ),
    )
    tools = OnlineCompilerToolkit(api_key="test-key", base_url=api.url).get_tools()
    return agents.create_agent(model=model, tools=tools)


def tool_messages(messages: list[BaseMessage]) -> list[ToolMessage]:
    return [message for message in messages if isinstance(message, ToolMessage)]


ARGS = {"compiler": "python-3.14", "code": "print(input()[::-1])", "input": "stressed"}


def test_agent_runs_the_tool(fake_api: FakeAPI) -> None:
    fake_api.reply(200, RUN_RESULT)
    result = agent_for(fake_api, ARGS).invoke({"messages": [{"role": "user", "content": "go"}]})
    [message] = tool_messages(result["messages"])
    assert message.status == "success"
    assert "output:\ndesserts" in message.content
    assert message.artifact == RUN_RESULT
    assert result["messages"][-1].content == "The program printed desserts."


async def test_agent_gets_a_refusal_as_an_error_message(fake_api: FakeAPI) -> None:
    fake_api.reply(400, {"error": "Unsupported or missing compiler"})
    agent = agent_for(fake_api, {**ARGS, "compiler": "python-2.7"})
    result = await agent.ainvoke({"messages": [{"role": "user", "content": "go"}]})
    [message] = tool_messages(result["messages"])
    assert message.status == "error"
    assert "Unsupported or missing compiler" in message.content


def test_agent_stops_on_a_refused_key(fake_api: FakeAPI) -> None:
    fake_api.reply(404, {"error": "Invalid or inactive API key"})
    with pytest.raises(AuthenticationError):
        agent_for(fake_api, ARGS).invoke({"messages": [{"role": "user", "content": "go"}]})
