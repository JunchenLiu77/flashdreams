set dotenv-load

default:
  just --list

# Setup the repository
setup:
  uv python install 3.10
  uv tool install "pre-commit>=4.3.0"
  pre-commit install -c .pre-commit-config-base.yaml

# Run pre-commit hooks
pre-commit *args: setup
  pre-commit run -a {{args}} || pre-commit run -a {{args}}

pyrefly *args:
  uv run pyrefly check --output-format=min-text {{args}}

# Run pyrefly and whitelist all errors
pyrefly-ignore *args: (pyrefly '--suppress-errors' args)

# Run linting and formatting
lint: pre-commit

alias format := lint

# Auto-fix or whitelist all ruff issues
_ruff-fix: setup
  -pre-commit run -a --hook-stage manual ruff-fix
  -pre-commit run -a --hook-stage manual ruff-noqa
  -pre-commit run -a --hook-stage manual ruff-format
  -pre-commit run -a --hook-stage manual ruff-noqa
