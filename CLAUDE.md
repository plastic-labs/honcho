# Honcho Overview

## What is Honcho?

Honcho is an infrastructure layer for building AI agents with social cognition and theory of mind capabilities. Its primary purposes include:

- Imbuing agents with a sense of identity
- Personalizing user experiences through understanding user psychology
- Providing a Dialectic API that injects personal context just-in-time
- Supporting development of LLM-powered applications that adapt to end users

Honcho leverages the inherent theory-of-mind capabilities of LLMs to build coherent models of user psychology over time, enabling more personalized and effective AI interactions.

## Development Guide

### Commands

- Setup: `uv sync`
- Run server: `fastapi dev src/main.py`
- Run tests: `pytest tests/`
- Run single test: `pytest tests/path/to/test_file.py::test_function`
- Linting: `ruff check src/`
- Format code: `ruff format src/`

### Code Style

- Follow isort conventions with absolute imports preferred
- Use explicit type hints with SQLAlchemy mapped_column annotations
- snake_case for variables/functions; PascalCase for classes
- Line length: 88 chars (Black compatible)
- Explicit error handling with appropriate exception types
- Docstrings: Use Google style docstrings

### Project Structure

- FastAPI routes in src/routers/
- SQLAlchemy ORM models in src/models.py with proper type annotations
- Pydantic schemas in src/schemas.py for API validation
- Tests in pytest with fixtures in tests/conftest.py
- Use environment variables via python-dotenv (.env)

### Error Handling

- Custom exceptions defined in src/exceptions.py
- Use specific exception types (ResourceNotFoundException, ValidationException, etc.)
- Proper logging with context instead of print statements
- Global exception handlers defined in main.py
- See docs/contributing/error-handling.mdx for details

### Notes

- Always use `uv run` or `uv` to prefix any commands related to python to ensure you use the virtual environment

