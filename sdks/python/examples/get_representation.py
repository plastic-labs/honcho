import random
import sys
import uuid
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import Honcho  # noqa: E402

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

alice = peers[0]
bob = peers[1]

# Get alice's working representation in the session
representation = session.working_rep(alice)
print("working representation returned:", representation)

# Get alice's working representation *of bob* in the session
representation = session.working_rep(alice, target=bob)
print("working representation returned:", representation)
