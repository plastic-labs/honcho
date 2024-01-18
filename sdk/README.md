# Honcho

A User context management solution for building AI Agents and LLM powered
applications.

Read about the motivation of this project [here](https://blog.plasticlabs.ai).

## Installation

Install honcho:

```bash
pip install honcho-ai
```

or 

```bash
poetry add honcho-ai
```

## Getting Started

The Honcho SDK exposes a top level client that contains methods for managing the
lifecycle of different conversations and sessions in an LLM powered application.

```python
from honcho import Client as HonchoClient

honcho = HonchoClient(base_url="http://localhost:8000")
user_id = "test"
session = honcho.create_session(user_id=user_id)

session_id = session["session_id"]

honcho.create_message_for_session(user_id=user_id, session_id=session_id, is_user=True, content="Hello")
```

The honcho sdk code contains docstrings - see the full sdk on
[GitHub](https://github.com/plastic-labs/honcho/tree/main/sdk/honcho/client.py)

See more examples of how to use the SDK on [GitHub](https://github.com/plastic-labs/honcho/tree/main/example)
