# Honcho Benchmark Suite

This directory contains benchmarking tools for evaluating Honcho's long-term memory capabilities.

## Available Benchmarks

- **LongMemEval**: Tests memory retention across multi-session conversations
- **BEAM**: Beyond a Million Tokens - comprehensive long-term memory evaluation across 10 memory abilities
- **LoCoMo**: Long conversation memory benchmark across multi-hop and temporal questions
- **OOLONG**: Long-context aggregation benchmark with `synth` and `real` variants

## Benchmark Workflow

Use a harness-first workflow for all benchmark runs:

1. Start Honcho locally with the benchmark harness:

```bash
python tests/bench/harness.py
```

2. Run one of the benchmark runners in another terminal:

```bash
# LongMemEval
python -m tests.bench.longmem --test-file tests/bench/longmemeval_data/longmemeval_oracle.json

# LoCoMo
python -m tests.bench.locomo --data-file tests/bench/locomo_data/locomo10.json

# BEAM
python -m tests.bench.beam --context-length 100K
```

3. For OOLONG, point `--data-dir` at your local dataset clone:

```bash
# OOLONG-synth
python -m tests.bench.oolong --variant synth --data-dir /path/to/oolong-synth

# OOLONG-real
python -m tests.bench.oolong --variant real --data-dir /path/to/oolong-real

# OOLONG-synth with label-augmented context (upstream optional mode)
python -m tests.bench.oolong --variant synth --data-dir /path/to/oolong-synth --labels
```

Notes for OOLONG runs:

- By default, synth uses `context_window_text` (upstream baseline behavior).
- Use `--labels` to switch synth ingestion to `context_window_text_with_labels`.
- Default `--min-context-len` is `1024` and filtering uses strict `>` matching upstream.

Expected local dataset layout:

```text
oolong-synth/
  data/
    test-*.parquet
    validation-*.parquet

oolong-real/
  dnd/
    test.jsonl
    validation.jsonl
```

## Development Harness

The development harness script makes it easy to run Honcho locally with a Docker database.

## Overview

The `harness.py` script orchestrates the complete Honcho development environment:

1. **Database Setup**: Starts a PostgreSQL database in Docker with a configurable port
2. **Database Provisioning**: Runs Alembic migrations to set up the database schema
3. **Configuration**: Uses environment variables to configure Honcho's database connection
4. **Service Startup**: Starts both the FastAPI server and deriver process
5. **Configuration Verification**: Prints the actual configuration that Honcho is using
6. **Monitoring**: Provides real-time logs from all services
7. **Cleanup**: Gracefully shuts down all services when stopped

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Honcho project dependencies installed (`uv sync`)

## Usage

### Basic Usage

Run the harness with default settings (database on port 5433):

```bash
python tests/bench/harness.py
```

### Custom Database Port

Run with a custom database port:

```bash
python tests/bench/harness.py --port 5434
```

### Custom Project Root

If running from a different directory:

```bash
python tests/bench/harness.py --project-root /path/to/honcho
```

### Command Line Options

- `--port`: Port for the PostgreSQL database (default: 5433)
- `--project-root`: Path to the Honcho project root (default: current directory)

## What Gets Started

When you run the harness, it will start:

1. **PostgreSQL Database**: Running in Docker on the specified port
2. **FastAPI Server**: Available at <http://localhost:8000>
3. **API Documentation**: Available at <http://localhost:8000/docs>
4. **Deriver Process**: Background worker for processing messages

## Configuration

The harness uses environment variables to configure Honcho's database connection:

- `DB_CONNECTION_URI`: Set to `postgresql+psycopg://testuser:testpwd@localhost:{port}/honcho`

The script will print the actual configuration that Honcho is using after the FastAPI server starts. This gives you complete visibility into how Honcho's configuration system resolved the settings from environment variables, config files, and defaults.

## Stopping the Services

Press `Ctrl+C` to gracefully stop all services. The harness will:

1. Stop the FastAPI server and deriver processes
2. Stop the Docker database container
3. Clean up temporary files (Docker Compose configuration)

## Troubleshooting

### Database Connection Issues

If the database fails to start or connect:

1. Check if port 5433 (or your custom port) is already in use
2. Ensure Docker is running
3. Try a different port: `--port 5434`

### Configuration Issues

The script will print the actual configuration being used. If you see unexpected values:

1. Check if you have a `config.toml` file that might be overriding environment variables
2. Verify that the environment variables are being set correctly
3. Check the Honcho configuration documentation for precedence rules

## Integration with CI/CD

This harness can be used in CI/CD pipelines for integration testing. The script will:

- Use temporary directories for isolation
- Clean up all resources on exit
- Provide clear error messages for debugging
- Exit with appropriate status codes
- Use environment variables for configuration (no file conflicts)

## Test Runner

The `run_tests.py` script executes JSON-formatted tests against a running Honcho instance. The harness must be running.

### Running Tests

1. **Start Honcho using the harness**:

   ```bash
   python tests/bench/harness.py
   ```

2. **In another terminal, run the tests**:

   ```bash
   # Run all tests
   python tests/bench/run_tests.py

   # Run a specific test
   # Test judge uses claude 3.5 sonnet
   python tests/bench/run_tests.py --test 1.json
   ```

### Test Workflow

For each test, the runner:

1. **Creates a workspace** for the test
2. **Adds all messages** from the JSON to sessions
3. **Waits for deriver queue** to be empty
4. **Executes queries** as `.chat()` calls
5. **Judges responses** using expected_response field

### Test JSON Format

Tests are defined in JSON files with this structure:

```json
{
    "sessions": {
        "session1": {
            "messages": [
                {
                    "peer": "alice",
                    "content": "Hello, how are you?"
                },
                {
                    "peer": "bob",
                    "content": "I'm good, thank you!"
                }
            ]
        }
    },
    "queries": [
        {
            "query": "How is Bob doing?",
            "expected_response": "Good",
            "session": "session1",  // optional
            "peer": "alice"         // optional
        }
    ]
}
```

### Command Line Options

- `--tests-dir`: Directory containing JSON test files (default: tests/bench/tests)
- `--test`: Run a specific test file
- `--honcho-url`: URL of running Honcho instance (default: <http://localhost:8000>)
- `--anthropic-api-key`: Anthropic API key for response judging, uses LLM_ANTHROPIC_API_KEY if not given
- `--timeout`: Timeout for deriver queue to empty (default: 60 seconds)
