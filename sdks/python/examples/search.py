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
session = honcho.session("search_test_" + str(uuid.uuid4()))

# Create a message with our special keyword
keyword = f"~special-{str(uuid.uuid4())}~"
session.add_messages(peers[0].message(f"I am a {keyword} message"))

# Generate some random messages from alice, bob, and charlie and add them to the session
messages = []
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

alice = peers[0]

# Add a different message to alice's global representation
different_keyword = f"~different-{str(uuid.uuid4())}~"
alice.add_messages(alice.message(f"I am a {different_keyword} message"))

# Search alice's global representation for the different message
search_results = alice.search(different_keyword)
print("searching alice's global representation")
print("search results returned:", [message for message in search_results])
