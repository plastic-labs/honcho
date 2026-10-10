# Honcho mock provider

A standalone, deterministic OpenAI-compatible chat and embedding service. It
requires Python 3.11+ and has no Honcho server, SDK, database, or test-suite dependency.

```bash
uv run --package honcho-mock-provider honcho-mock-provider --port 8106
```

The wheel exposes the same command and `python -m honcho_mock_provider` outside
the repository. The default bind address is `127.0.0.1`. Hosts can embed the app
with `honcho_mock_provider.main.create_app()`.

Endpoints are `/health`, `/v1/chat/completions`, and `/v1/embeddings`; the provider
also accepts chat and embedding requests without `/v1`. Structured output,
streaming, usage accounting, and hash-derived embeddings preserve the existing
mock behavior. Unsupported POST endpoints fail explicitly. Embeddings are
synthetic and do not represent semantic similarity.

The server's `src.mock_provider` imports remain compatibility adapters. CLI
users can start the real API and deriver with `--providers mock`; that preset
routes model paths and fallbacks to this service and disables external telemetry.
No package is downloaded implicitly by native startup.
