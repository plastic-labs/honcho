import random
import uuid

from honcho import Honcho, MessageCreateParams

# Create a Honcho client with the default workspace
honcho = Honcho(environment="local")

peers = [
    honcho.peer("alice"),
    honcho.peer("bob"),
    honcho.peer("charlie"),
]

alice = peers[0]

# Create a new session
session = honcho.session("search_test_" + str(uuid.uuid4()))

# Create a message with our special keyword
keyword = f"~special-{str(uuid.uuid4())}~"
session.add_messages(alice.message(f"I am a {keyword} message"))

# Generate some random messages from alice, bob, and charlie and add them to the session
messages: list[MessageCreateParams] = []
for i in range(10):
    random_peer = random.choice(peers)
    messages.append(
        random_peer.message(f"Hello from {random_peer}! This is message {i}.")
    )

session.add_messages(messages)

# Search the session for the special keyword
search_results = session.search(keyword)
print("searching the session")
print("search results returned:", [message for message in search_results])

# Search the workspace for the special keyword
search_results = honcho.search(keyword)
print("searching the workspace")
print("search results returned:", [message for message in search_results])

# Search alice's messages for the special keyword
search_results = alice.search(keyword)
print("searching alice's messages")
print("search results returned:", [message for message in search_results])
