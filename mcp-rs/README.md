# Honcho MCP Rust Server

Native Rust MCP server for local and Docker Compose self-hosted Honcho deployments.
It mirrors the tool contract from the TypeScript Cloudflare Worker in `../mcp`
but avoids the Node, npm, and Wrangler runtime in the default Compose stack.

## Run Locally

```bash
HONCHO_API_URL=http://127.0.0.1:8000 cargo run
```

The server listens on `0.0.0.0:8787` by default. Override with
`MCP_BIND_ADDRESS`, for example:

```bash
MCP_BIND_ADDRESS=127.0.0.1:8787 HONCHO_API_URL=http://127.0.0.1:8000 cargo run
```

## Docker Compose

The root Compose file builds this package for the default `mcp` service:

```bash
docker compose up -d --build mcp
```

The legacy TypeScript/Wrangler development server is still available for
rollback or Worker debugging:

```bash
docker compose --profile typescript-mcp up -d --build mcp-typescript
```

That profiled service binds host port `8788` and still serves the MCP endpoint
on container port `8787`.

## Required Headers

Clients must send the same headers as the hosted Worker:

| Header | Required | Default |
| --- | --- | --- |
| `Authorization: Bearer <key>` | Yes | none |
| `X-Honcho-User-Name` | Yes | none |
| `X-Honcho-Assistant-Name` | No | `Assistant` |
| `X-Honcho-Workspace-ID` | No | `default` |

## Contract Tests

```bash
cargo test --manifest-path mcp-rs/Cargo.toml
```

The tests pin the MCP tool names, key schemas, header parsing behavior, response
text wrapping, session peer config conversion, and Honcho API URL encoding.
