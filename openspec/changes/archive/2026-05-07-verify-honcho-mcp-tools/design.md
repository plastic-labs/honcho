## Context

The `honcho-mcp` service translates Honcho's capabilities into Model Context Protocol (MCP) tools using a Cloudflare Worker architecture over an HTTP/SSE transport. Since these tools are the bridge for agents (like Windsurf/Cursor) to interact with Honcho, it is critical to verify they function statelessly and correctly per their domain definitions (workspace, peers, sessions, conclusions, system).

## Goals / Non-Goals

**Goals:**
- Execute a comprehensive End-to-End verification of the 5 tool domains of `honcho-mcp`.
- Confirm stateless headers (`X-Honcho-*`) correctly isolate context.
- Produce a reproducible QA report.

**Non-Goals:**
- Modifying the underlying `honcho-mcp` code.
- Stress or load testing.

## Decisions

- Use the IDE MCP client to execute the tools directly in sequence.
- The verification will follow the 5 phases defined in the exploration doc:
  - Setup & Lifecycle Validation
  - Interaction & Data Insertion
  - Context & Search Verification
  - Knowledge Graph & Reasoning
  - Teardown Validation

## Risks / Trade-offs

- **Risk:** Some tools (like `schedule_dream`) run asynchronously.
  - **Mitigation:** Use `get_queue_status` to wait or check status instead of assuming instantaneous completion.
