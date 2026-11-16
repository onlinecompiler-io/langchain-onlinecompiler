# langchain-onlinecompiler

LangChain tools for [OnlineCompiler.io](https://onlinecompiler.io): let an agent run code, or judge code against test cases, in 46 languages. Each run gets a fresh sandbox with no network access.

- `OnlineCompilerRunCode` (tool name `onlinecompiler_run_code`) runs a program and returns its output, errors, exit code, time and memory.
- `OnlineCompilerRunTests` (tool name `onlinecompiler_run_tests`) judges a program against test cases and returns an overall verdict, a score from 0 to 100 and a verdict per test.
- `OnlineCompilerToolkit` gives you both.

The tools call the OnlineCompiler.io REST API through the official [`onlinecompiler`](https://pypi.org/project/onlinecompiler/) Python client.

## Install

```bash
pip install -U langchain-onlinecompiler
```

## Get an API key

Create a free key at [api.onlinecompiler.io/compilers/api-keys/](https://api.onlinecompiler.io/compilers/api-keys/) (key type **API**). The free plan includes 100,000 runs a month. Then set it in the environment:

```bash
export ONLINECOMPILER_API_KEY="your-api-key"
```

or pass it to a tool or the toolkit as `api_key="..."`. The key is held as a pydantic `SecretStr`, so it does not show up in reprs or logs.

## Use the tools directly

```python
from langchain_onlinecompiler import OnlineCompilerRunCode, OnlineCompilerRunTests

run_code = OnlineCompilerRunCode()
print(run_code.invoke({"compiler": "python-3.14", "code": "print(2 ** 100)"}))
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

```python
run_tests = OnlineCompilerRunTests()
print(
    run_tests.invoke(
        {
            "compiler": "python-3.14",
            "code": "a, b = map(int, input().split())\nprint(a - b)",
            "tests": [
                {"input": "2 3", "expected_output": "5"},
                {"input": "10 4", "expected_output": "14"},
            ],
        }
    )
)
```

```text
verdict: wrong_answer
score: 0
passed: 0 of 2
summary: 0 of 2 sample tests passed. Sample 1 expected 5 but the program printed -1.
test 1: wrong_answer (0.0338 s, exit_code 0)
  output:
-1
test 2: wrong_answer (0.0378 s, exit_code 0)
  output:
6
```

## Use them in an agent

```python
# pip install -U langchain langchain-openai
from langchain.agents import create_agent
from langchain_onlinecompiler import OnlineCompilerToolkit

agent = create_agent(
    model="openai:gpt-5-mini",
    tools=OnlineCompilerToolkit().get_tools(),
    system_prompt="Run code to check your answers instead of guessing.",
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the 90th Fibonacci number? Compute it.",
            }
        ]
    }
)
print(result["messages"][-1].content)
```

Any chat model with tool calling works. The tools also work with LangGraph's `ToolNode` and anything else that takes LangChain tools.

## The tools

### `onlinecompiler_run_code`

Calls `POST /api/run-code-sync/`.

| Argument | Type | Description |
| --- | --- | --- |
| `compiler` | string | A compiler id such as `python-3.14`, `nodejs-24`, `gcc-15`, `g++-15`, `openjdk-25`, `go-1.26`, `rust-1.93` or `sqlite-3`. All ids: [onlinecompiler.io/docs/compilers](https://onlinecompiler.io/docs/compilers). |
| `code` | string | The whole program, as one source file. |
| `input` | string, optional | Standard input for the program. |
| `time_limit` | number, optional | Seconds the program may run, above 0 and at most 30 (the default). |

The model reads the status, exit code (with its meaning for 124, 137 and the other codes the sandbox sets), time, memory, whether output was cut, the output and the error text. Files the program writes to `$OUTPUT_DIR` are listed by name and link.

### `onlinecompiler_run_tests`

Calls `POST /api/run-tests/`.

| Argument | Type | Description |
| --- | --- | --- |
| `compiler` | string | A compiler id, as above. |
| `code` | string | The whole program, as one source file. |
| `tests` | list | Test cases, each `{"input": "...", "expected_output": "..."}`; `input` is optional. |
| `checker` | string, optional | How output is compared: `trim` (default), `exact`, `tokens`, `float`, `rows`, `table` or `table_ordered`. |

The model reads the overall verdict, the score, the passed count, a plain-words summary, any compile error and a line per test. A visible test that failed also shows what the program printed; a hidden test shows only its verdict. Each test runs for up to 2 seconds in its own sandbox.

### Full results as the artifact

Both tools use `response_format="content_and_artifact"`. Called with a tool call (as an agent does), a tool returns a `ToolMessage` whose `content` is the short text above and whose `artifact` is the API's full JSON answer: for a run, every field including output file contents; for run-tests, every test result, the submission id and the signed receipt.

```python
message = run_code.invoke(
    {
        "args": {"compiler": "python-3.14", "code": "print('hi')"},
        "id": "call_1",
        "name": "onlinecompiler_run_code",
        "type": "tool_call",
    }
)
message.artifact["exit_code"]  # 0
```

### Async

Both tools support `ainvoke`. The `onlinecompiler` client is synchronous (standard library only), so an async call runs it in a worker thread and does not block the event loop.

## The sandbox

- 46 languages, from Python, JavaScript, TypeScript, C, C++, Java, Go and Rust to SQL, R and COBOL.
- Every run gets a fresh, isolated sandbox with **no network access**: code cannot call APIs or install packages. Popular packages are preinstalled (for Python: NumPy, pandas, SciPy, scikit-learn, Matplotlib and more).
- Up to 30 seconds and 512 MB per run; output is capped at 64 KB per stream.
- Each run counts as one run against your key's allowance; for run-tests, each test counts as one run.

Current limits: [onlinecompiler.io/docs](https://onlinecompiler.io/docs).

## Errors

| HTTP status | What the tool does |
| --- | --- |
| 400 (bad request, such as an unknown compiler id) | Returns an error `ToolMessage` to the model with the API's message, so it can correct the call. |
| 413 (request body over 32 MB) | Returns an error `ToolMessage` to the model. |
| 429 (a limit: runs at once, runs a minute, pending runs or the monthly allowance) | Returns an error `ToolMessage` that says which limit; nothing ran and nothing was counted. |
| 5xx, or no answer | Returns an error `ToolMessage`. |
| 401, 403 or 404 (the API key was refused) | Raises `onlinecompiler.AuthenticationError` (also exported from this package), since the model cannot fix a key. |

Errors the model can act on are `OnlineCompilerToolError` (429 is the subclass `OnlineCompilerRateLimitError`), a `ToolException`. The tools set `handle_tool_error=True` so these come back to the model; pass `handle_tool_error=False` to have them raised instead. Neither tool retries on its own, because every accepted run counts.

## Settings

| Field | Default | Description |
| --- | --- | --- |
| `api_key` | `ONLINECOMPILER_API_KEY` | Your API key (type API). |
| `base_url` | `https://api.onlinecompiler.io` | Where the API is. |
| `timeout` | `90` | Seconds to wait for an answer. |

## Development

```bash
pip install -e . pytest pytest-asyncio pytest-socket "langchain-tests>=1.1.9,<1.2"
pytest tests/unit_tests --allow-hosts=127.0.0.1          # offline, against a fake API
ONLINECOMPILER_API_KEY=... pytest tests/integration_tests  # live; uses about 20 runs
```

## License

MIT. Copyright (c) 2026 RunCode Inc.
