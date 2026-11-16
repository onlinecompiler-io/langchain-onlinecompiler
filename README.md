# LangChain integration for OnlineCompiler.io

Tools that let a LangChain agent run code, or judge code against test cases, in 46 languages with [OnlineCompiler.io](https://onlinecompiler.io). Each run gets a fresh sandbox with no network access.

| Package | Folder | Registry |
| --- | --- | --- |
| `langchain-onlinecompiler` (Python) | [`python/`](python/) | [PyPI](https://pypi.org/project/langchain-onlinecompiler/) |

```bash
pip install -U langchain-onlinecompiler
export ONLINECOMPILER_API_KEY="your-api-key"  # free: https://api.onlinecompiler.io/compilers/api-keys/
```

```python
from langchain.agents import create_agent
from langchain_onlinecompiler import OnlineCompilerToolkit

agent = create_agent(model="openai:gpt-5-mini", tools=OnlineCompilerToolkit().get_tools())
```

See [`python/README.md`](python/README.md) for the tools, their arguments, results and errors.

## Releases

Pushing a tag `python-vX.Y.Z` runs [`.github/workflows/python-release.yml`](.github/workflows/python-release.yml): it runs the unit tests, checks that the tag matches `__version__`, builds, and publishes to PyPI by trusted publishing. No token is stored anywhere.

## License

MIT. Copyright (c) 2026 RunCode Inc.
