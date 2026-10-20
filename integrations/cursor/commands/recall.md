---
name: recall
description: Ask Honcho what it remembers about a topic from past sessions
---

# Recall from memory

Ask Honcho about the topic given after `/recall`. If no topic was given, ask
for one.

1. Resolve the workspace and user peer as the `honcho-memory` skill describes.
2. Call `chat` with the user peer as `peer_id`, the topic as the question, and
   `reasoning_level: "low"`. Honcho reasons over its conclusions and past
   messages, so this is the answer to report.
3. Call `search` with the topic only if `chat` comes back empty, or the user
   wants an exact quote, error message or file name. Leave out `peer_id` so
   both sides of past conversations match.
4. Summarize what was found, with session names and dates where available.
   If nothing was found, say so plainly.
