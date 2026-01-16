import random
import uuid

from honcho import Honcho

# Create a Honcho client with the default workspace
honcho = Honcho(environment="local")

peers = [
    honcho.peer("alice"),
    honcho.peer("bob"),
    honcho.peer("charlie"),
]

# Create a new session
session = honcho.session("context_test_" + str(uuid.uuid4()))

# Generate some random messages from alice, bob, and charlie and add them to the session
messages = []
for i in range(10):
    random_peer = random.choice(peers)
    messages.append(
        random_peer.message(f"Hello from {random_peer}! This is message {i}.")
    )

session.add_messages(messages)

# Get some context of the session
# Set the token limit super low so we only get a few of the tiny messages created
context = session.context(summary=True, tokens=50)
print("context returned:", context)
