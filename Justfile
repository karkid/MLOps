[default]
[doc("List all available recipes")]
default:
	@just --list


# Set UV to use copy mode instead of hardlinks to avoid warnings
export UV_LINK_MODE := "copy"

[doc("Create a virtual environment and synchronize dependencies")]
init: check-uv
    @echo "Creating virtual environment and syncing dependencies..."
    uv venv
    uv sync
    @echo "Installing development dependencies..."
    uv pip install -e ".[dev]"

[confirm("This will delete your virtual environment and reinstall all dependencies. Continue?")]
[doc("Reinstall all dependencies from scratch")]
reinstall:
	@echo "Deleting old virtual environment..."
	@rm -rf .venv
	just init

[doc("Run all quality checks (formatting, linting, tests)")]
check: lint test

[doc("Run pytest test suite")]
test:
    uv run pytest

[doc("Run linting and formatting (Ruff)")]
lint:
    uv run ruff check reml --fix
    uv run ruff format reml

[doc("Run the main program")]
run:
    uv run python -m reml

[doc("Show uv version and environment details")]
info:
    @echo "📦 Environment Info:"
    uv --version
    uv pip list

[private]
[doc("Check if uv is installed")]
check-uv:
   @uv --version >/dev/null 2>&1 || (echo "Error: uv is not installed" && exit 1)

[windows]
clean:
    @echo "Cleaning up virtual environment and build artifacts..."
    @powershell -Command 'Remove-Item -Recurse -Force dist, build, .mypy_cache, .pytest_cache, .ruff_cache, .coverage -ErrorAction SilentlyContinue; Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force'

[unix]
clean:
    @echo "Cleaning up virtual environment and build artifacts..."
    @rm -rf dist build .mypy_cache .pytest_cache .ruff_cache .coverage
    @find . -type d -name "__pycache__" -exec rm -rf {} +

[doc("Add a development setup recipe for installing additional tools")]
dev-setup:
    @echo "Installing development tools..."
    uv pip install -e ".[dev]"

[doc("Run tests with coverage report")]
coverage:
    uv run pytest --cov=reml --cov-report=term-missing

[doc("Build package for distribution")]
build:
    uv pip install build
    uv run python -m build

[doc("Update all dependencies to their latest compatible versions")]
update-deps:
    uv pip install -U pip
    uv pip install --upgrade-deps