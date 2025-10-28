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
    @echo "Installing dependencies..."
    uv pip install -e ".[dev,notebook]"

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
    @powershell -Command 'Remove-Item -Recurse -Force .venv, dist, build, .mypy_cache, .pytest_cache, .ruff_cache, .coverage -ErrorAction SilentlyContinue; Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force'

[unix]
clean:
    @echo "Cleaning up virtual environment and build artifacts..."
    @rm -rf .venv dist build .mypy_cache .pytest_cache .ruff_cache .coverage
    @find . -type d -name "__pycache__" -exec rm -rf {} +

[doc("Add a development dependencies recipe for installing additional tools")]
dev-deps:
    @echo "Installing development tools..."
    uv pip install -e ".[dev]"

[doc("Install notebook runtime dependencies using uv and the pyproject extras")]
notebook-deps:
    @echo "Installing notebook extras via pyproject.toml (uv)"
    uv pip install -e ".[notebook]"
    @echo "Registering ipykernel for the environment"
    uv run python -m ipykernel install --user --name reml-venv --display-name "Python (uv - reml)"

[doc("Add a package to project metadata")]
add PACKAGE FLAG="":
    @echo "Adding package '{{PACKAGE}}' (flag='{{FLAG}}')"
    @FLAG='{{FLAG}}'; \
    if [ -n "$FLAG" ]; then \
        uv add '{{PACKAGE}}' --optional {{FLAG}}; \
    else \
        uv add '{{PACKAGE}}'; \
    fi

[doc("Remove a package from project metadata")]
remove PACKAGE FLAG="":
    @echo "Removing package '{{PACKAGE}}' (flag='{{FLAG}}')"
    @FLAG='{{FLAG}}'; \
    if [ -n "$FLAG" ]; then \
        uv remove '{{PACKAGE}}' --optional {{FLAG}}; \
    else \
        uv remove '{{PACKAGE}}'; \
    fi

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