---
name: recall
description: Search Honcho's memory for something from past sessions
---

# Recall from memory

Search Honcho for the topic given after `/recall`. If no topic was given, ask
for one.

1. Resolve the workspace and user peer as the `honcho-memory` skill describes.
2. Call `search` with the topic. If the results are thin, also ask `chat`
   with the same question at `reasoning_level: "low"`.
3. Summarize what was found, with session names and dates where available.
   If nothing was found, say so plainly.
