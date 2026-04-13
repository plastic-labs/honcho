---
name: honcho-cli-debug
version: 0.1.0
description: Debug Honcho peer representations and memory
---

# Honcho CLI — Debug Skills

## Rules

- Check queue status when derivation seems stalled
- Compare peer card with conclusions to understand memory state

## Debugging Memory Issues

### Peer not learning?

```bash
# Check if observation is enabled
honcho peer inspect <peer_id> --json | jq '.configuration'

# Check queue — are messages being processed?
honcho workspace queue-status --json

# Check what conclusions exist
honcho conclusion list --observer <peer_id> --json
honcho conclusion search "expected topic" --observer <peer_id> --json
```

### Session context looks wrong?

```bash
# See raw context
honcho session context <session_id> --json

# Check summaries
honcho session summaries <session_id> --json

# Check message history
honcho session messages <session_id> --last 50 --json
```

### Dialectic giving bad answers?

```bash
# Check what the peer card says
honcho peer card <peer_id> --json

# Check conclusions for the specific topic
honcho conclusion search "topic" --observer <peer_id> --json

# Try the dialectic directly
honcho peer chat <peer_id> "what do you know about X?" --json
```
