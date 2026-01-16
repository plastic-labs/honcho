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
session = honcho.session("chat_test_" + str(uuid.uuid4()))

# Generate some random messages from alice, bob, and charlie and add them to the session
messages = []
for i in range(10):
    random_peer = random.choice(peers)
    messages.append(
        random_peer.message(f"Hello from {random_peer}! This is message {i}.")
    )

session.add_messages(messages)

honcho.poll_queue_status()

# Chat with alice
alice = peers[0]
response = alice.chat("what did alice have for breakfast today?")
print("response returned:", response)

# Chat with alice in the session
response = alice.chat("what did alice have for breakfast today?", session=session.id)
print("response returned:", response)

# Chat with alice in the session with a target
# This means you are querying alice's theory-of-mind representation of bob
bob = peers[1]
response = alice.chat("what did bob have for breakfast today?", target=bob)
print("response returned:", response)
