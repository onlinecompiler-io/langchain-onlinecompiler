.PHONY: test integration_tests lint format type build

# Unit tests talk only to a fake API on 127.0.0.1.
test:
	pytest tests/unit_tests --allow-hosts=127.0.0.1

# Live tests: they need ONLINECOMPILER_API_KEY and use about 20 runs of its allowance.
integration_tests:
	pytest -v tests/integration_tests

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

type:
	mypy langchain_onlinecompiler tests

build:
	python -m build
