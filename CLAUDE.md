# Honcho Development Guide

## Commands
- Setup: `uv sync`
- Run server: `fastapi dev src/main.py`
- Run tests: `pytest tests/`
- Run single test: `pytest tests/path/to/test_file.py::test_function`
- Linting: `ruff check src/`
- Format code: `ruff format src/`

## Code Style
- Follow isort conventions with absolute imports preferred
- Use explicit type hints with SQLAlchemy mapped_column annotations
- snake_case for variables/functions; PascalCase for classes
- Line length: 88 chars (Black compatible)
- Explicit error handling with appropriate exception types
- Docstrings: Use Google style docstrings

## Project Structure
- FastAPI routes in src/routers/
- SQLAlchemy ORM models in src/models.py with proper type annotations
- Pydantic schemas in src/schemas.py for API validation
- Tests in pytest with fixtures in tests/conftest.py
- Use environment variables via python-dotenv (.env)

## Error Handling
- Custom exceptions defined in src/exceptions.py
- Use specific exception types (ResourceNotFoundException, ValidationException, etc.)
- Proper logging with context instead of print statements
- Global exception handlers defined in main.py
- See docs/contributing/error-handling.mdx for details