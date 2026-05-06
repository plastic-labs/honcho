# Agent Integration Patterns

How to wire your Honcho self-hosted instance into AI agent systems.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Athena     │────▶│              │◀────│   Hermes    │
│  Pentest    │     │   Honcho     │     │   Agent     │
└─────────────┘     │  (self-host) │     └─────────────┘
                    │              │
┌─────────────┐     │  ParadeDB    │     ┌─────────────┐
│  Multica    │────▶│  + Redis     │◀────│  Future     │
│  Platform   │     │              │     │  Agents     │
└─────────────┘     └──────────────┘     └─────────────┘
```

## Tenant Setup

Each agent system gets its own tenant for isolation:

```bash
# Create tenants (admin JWT required)
curl -X POST http://localhost:8000/v3/tenants \
  -H "Authorization: Bearer $ADMIN_JWT" \
  -H "Content-Type: application/json" \
  -d '{"name": "athena-pentest"}'

curl -X POST http://localhost:8000/v3/tenants \
  -H "Authorization: Bearer $ADMIN_JWT" \
  -H "Content-Type: application/json" \
  -d '{"name": "hermes-agent"}'

curl -X POST http://localhost:8000/v3/tenants \
  -H "Authorization: Bearer $ADMIN_JWT" \
  -H "Content-Type: application/json" \
  -d '{"name": "multica"}'
```

## Pattern 1: Athena Pentest — Engagement Memory

**Workspace per engagement, peer per target, messages for findings.**

```python
import requests

HONCHO_URL = "http://localhost:8000/v3"
ENGAGEMENT = "speccon-2026-q2"

# Create workspace for this engagement
requests.post(f"{HONCHO_URL}/workspaces", json={
    "id": ENGAGEMENT,
    "tenant_id": "athena-tenant-id"
})

# Create peer for each target
target = requests.post(f"{HONCHO_URL}/peers", json={
    "id": "speccon.co.za",
    "workspace_id": ENGAGEMENT,
    "metadata": {"type": "target", "ip": "196.xx.xx.xx"}
})

# Log findings as messages (Honcho's Deriver extracts observations)
requests.post(f"{HONCHO_URL}/messages", json={
    "session_id": ENGAGEMENT,
    "messages": [{
        "peer_id": "speccon.co.za",
        "content": "Discovered Apache 2.4.49 with mod_cgi enabled. CVE-2021-41773 present.",
        "metadata": {"severity": "critical", "cvss": 9.8}
    }]
})

# Query memory later — no need to re-scan
response = requests.post(f"{HONCHO_URL}/peers/speccon.co.za/chat", json={
    "query": "What Apache vulnerabilities were found on this target?",
    "agentic": True
})
# → "CVE-2021-41773 (path traversal) was found on Apache 2.4.49..."
```

## Pattern 2: Hermes Agent — Cross-Session Memory

**Workspace per user, peer per agent instance, sessions for conversations.**

Hermes already has a Honcho memory plugin (`hermes honcho setup`). Point it at your self-hosted instance:

```bash
# In ~/.hermes/config.yaml
memory:
  provider: honcho
  honcho:
    base_url: http://localhost:8000/v3
    api_key: ${HONCHO_API_KEY}
    workspace_id: hermes-user-theo
```

If using the Hermes Honcho plugin directly:

```python
from honcho import Honcho

honcho = Honcho(
    base_url="http://localhost:8000/v3",
    api_key="your-jwt-token",
    workspace_id="hermes-user-theo"
)

# Each user gets a peer
user = honcho.peer("theo")
agent = honcho.peer("hermes-agent")

# Sessions persist across conversations
session = honcho.session("session-2026-05-06")
session.add_messages([
    user.message("Deploy honcho self-hosted with ParadeDB"),
    agent.message("Starting deployment... Phase 1 complete.")
])

# Later: recall what we were working on
context = session.context(summary=True, tokens=10000)
```

## Pattern 3: Multica — Project Memory

**Workspace per project, peer per user, sessions for task threads.**

```python
# Each Multica project gets a workspace
workspace = honcho.workspace("multica-frontend-redesign")

# Team members as peers
alice = honcho.peer("alice", metadata={"role": "frontend"})
bob = honcho.peer("bob", metadata={"role": "backend"})

# Task threads as sessions
session = honcho.session("task-42-auth-flow")
session.add_messages([
    alice.message("Auth flow needs OAuth2 PKCE — spec in Notion"),
    bob.message("On it. Using the new middleware from PR #108.")
])

# Query project context
response = alice.chat("What's the current state of the auth flow?")
```

## Python Client Helper

```python
"""Minimal Honcho client for self-hosted instances."""
import requests
from typing import Optional


class HonchoClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def create_workspace(self, name: str, tenant_id: Optional[str] = None) -> dict:
        body = {"id": name}
        if tenant_id:
            body["tenant_id"] = tenant_id
        return self._post("/workspaces", body)

    def create_peer(self, name: str, workspace_id: str, metadata: dict = None) -> dict:
        return self._post("/peers", {
            "id": name,
            "workspace_id": workspace_id,
            "metadata": metadata or {}
        })

    def create_session(self, name: str, workspace_id: str, peer_ids: list[str]) -> dict:
        return self._post("/sessions", {
            "id": name,
            "workspace_id": workspace_id,
            "peer_ids": peer_ids
        })

    def add_messages(self, session_id: str, messages: list[dict]) -> dict:
        return self._post("/messages", {
            "session_id": session_id,
            "messages": messages
        })

    def chat(self, peer_id: str, query: str) -> dict:
        return self._post(f"/peers/{peer_id}/chat", {
            "query": query,
            "agentic": True
        })

    def _post(self, path: str, data: dict) -> dict:
        r = self.session.post(f"{self.base_url}{path}", json=data)
        r.raise_for_status()
        return r.json()
```

## LLM Requirements

Honcho's Deriver/Dialectic/Dreamer agents need LLM API access. Configure in `.env`:

```bash
# At minimum one of:
LLM_OPENAI_API_KEY=sk-...
LLM_ANTHROPIC_API_KEY=sk-ant-...

# Embedding model for vector search
EMBEDDING_MODEL_CONFIG__TRANSPORT=openai
EMBEDDING_MODEL_CONFIG__MODEL=text-embedding-3-small
```
