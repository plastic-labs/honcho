# Honcho

A User context management solution for building AI Agents and LLM powered
applications.

Read about the motivation of this project [here](https://blog.plasticlabs.ai).

Read the full documentation of this project [here](https://docs.honcho.dev)

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

There is a demo server being run at https://demo.honcho.dev that the client uses
by default if no other string is provided.

```python
from uuid import uuid4
from honcho import Client as HonchoClient

app_id = str(uuid4())
honcho = HonchoClient(app_id=app_id)
user_id = "test"
session = honcho.create_session(user_id=user_id)


session.create_message(is_user=True, content="Hello I'm a human")
session.create_message(is_user=False, content="Hello I'm an AI")
```

The honcho sdk code contains docstrings — see the full sdk on
[GitHub](https://github.com/plastic-labs/honcho/tree/main/sdk/honcho/client.py)

See more examples of how to use the SDK on [GitHub](https://github.com/plastic-labs/honcho/tree/main/example)
