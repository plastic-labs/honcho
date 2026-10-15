---
name: remember
description: Save a durable fact or preference about the user to Honcho
---

# Remember this

Store the text given after `/remember` as a conclusion about the user. If no
text was given, ask what to remember.

1. Resolve the workspace and user peer as the `honcho-memory` skill describes.
2. Call `search` with the key phrase to avoid storing a duplicate.
3. Call `create_conclusions` for the user peer with the fact, phrased as a
   statement about the user.
4. Confirm what was saved in one line. Never store secrets.
