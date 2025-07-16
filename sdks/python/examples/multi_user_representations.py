import time
import uuid

from honcho import Honcho
from honcho.session import SessionPeerConfig

# Create a Honcho client with the default workspace
honcho = Honcho(environment="local")

alice = honcho.peer("alice")
bob = honcho.peer("bob")

# Create a new session
session = honcho.session("chat_test_" + str(uuid.uuid4()))

session.add_peers(
    [
        (alice, SessionPeerConfig(observe_me=True, observe_others=True)),
        (bob, SessionPeerConfig(observe_me=True, observe_others=True)),
    ]
)

# Generate messages with personal information
messages = []
messages.append(alice.message("I had a great breakfast today!"))
messages.append(bob.message("What did you eat?"))
messages.append(alice.message("I had pancakes and eggs and bacon."))

session.add_messages(messages)

time.sleep(5)

# Create a separate session
session2 = honcho.session("chat_test_" + str(uuid.uuid4()))
session2.add_peers(
    [
        (alice, SessionPeerConfig(observe_me=True, observe_others=True)),
        (bob, SessionPeerConfig(observe_me=True, observe_others=True)),
    ]
)
session2.add_messages(
    [
        alice.message(
            "Hey remember when I told you I had a great breakfast today? I lied. I actually skipped breakfast."
        ),
        bob.message("WTF is wrong with you??"),
    ]
)

# wait for the deriver to process the messages
print("waiting for the deriver to process all the messages")
deriver_status = honcho.poll_deriver_status()
print("deriver status:", deriver_status)


# # Chat with alice's honcho-level representation
# print(
#     "\n\n\033[1m asking alice's honcho-level representation what she had for breakfast \033[0m"
# )
# response = alice.chat("what did alice have for breakfast today?", session_id=session.id)
# print("response:", response)

# Chat with bob's internal representation of alice
print(
    "\n\n\033[1m asking bob what alice had for breakfast -- scoped to session 1 \033[0m"
)
response = bob.chat(
    "what did alice have for breakfast today?", target=alice, session_id=session.id
)
print("response:", response)

print(
    "\n\n\033[1m asking bob what alice had for breakfast -- scoped to session 2 \033[0m"
)
response = bob.chat(
    "what did alice have for breakfast today?", target=alice, session_id=session2.id
)
print("response:", response)

print("\n\n\033[1m asking bob what alice had for breakfast -- global scope \033[0m")
response = bob.chat("what did alice have for breakfast today?", target=alice)
print("response:", response)
