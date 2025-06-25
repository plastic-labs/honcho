import logging

from honcho import Honcho
from honcho.session import SessionPeerConfig

logging.basicConfig(level=logging.INFO)


# HONCHO_API_KEY is an environment variable
# HONCHO_URL is an *optional* environment variable
# HONCHO_WORKSPACE_ID is an *optional* environment variable
# Using local server for this example
honcho = Honcho(environment="local", workspace_id="test")

workspaces = honcho.get_workspaces()

# these don't make any API calls, just produce a Peer object in SDK
# in practice, these would be UUIDs, as peer IDs are unique within their workspace
assistant = honcho.peer(id="bob")
alice = honcho.peer(id="alice")

# empty since peers are not created until they are used
peers = honcho.get_peers()

# workspace-level metadata
_m = honcho.get_metadata()
honcho.set_metadata({"test": "test"})

# calling the dialectic chat endpoint makes an API call.
# when this call occurs, the "alice" peer will be get_or_create'd
# response will be None because we haven't talked yet!
response = alice.chat("what did alice have for breakfast today?")

# sessions are scoped to a set of peers and contain messages/content
# this is not an API call, like peers this is created lazily
my_session = honcho.session(id="session_1")

# API call
my_session.add_peers(
    [alice, (assistant, SessionPeerConfig(observe_others=False, observe_me=False))]
)

# adding/removing peers from sessions creates a bidirectional relationship,
# so no need for operations like `alice.join(my_session)`.

# this will return a list of sessions [my_session]
# this is also an API call
_sessions = alice.get_sessions()

# API call to create 1 or more messages (overload, can be Message or list[Message]
my_session.add_messages(
    [
        # creates a Message object with peer_id="alice", etc etc
        assistant.message("what did you have for breakfast today, alice?"),
        alice.message("i had oatmeal."),
    ]
)

m = my_session.get_metadata()
m["test"] = "test2"
my_session.set_metadata(m)

# peers have one "omnipresent" global representation, comprised of all
# the content associated with that peer in this honcho instance.

# they also have a potentially infinite number of "local" representations,
# each one from the perspective of *another* peer in the honcho instance.

# this is a query to alice's global representation--no scope
response = alice.chat("what did the user have for breakfast today?")

# this is a query to alice's local representation *of the assistant*
response = alice.chat("does alice know what bob had for breakfast?", target=assistant)

# this is a query to the assistant's local representation *of alice* in this session
response = assistant.chat(
    "does the assistant know what alice had for breakfast?",
    target=alice,
    session_id=my_session.id,
)

# API call to store non-message content under a peer + optional session
alice.add_messages("this might be a document about alice, say, a journal entry.")

charlie = honcho.peer(id="charlie")

my_session.add_messages(charlie.message("hello world!"))

# session now has 3 members: alice, bob, and charlie. a message automatically adds a peer to a session.

# peers, sessions, and messages all have metadata which can be modified and used in queries.

# API call to get metadata?
charlie_metadata = charlie.get_metadata()

charlie_metadata["location"] = "the moon"

# API call to store metadata?
charlie.set_metadata(charlie_metadata)

# response will tell you that charlie is on the moon
response = charlie.chat("where is the user?")

# you can get the messages from a session, either fully or partially.
# (API call)
messages = my_session.get_messages()

context = my_session.get_context()

messages = context.to_openai(assistant=assistant.id)

messages = context.to_anthropic(assistant=assistant.id)

my_session.add_messages(
    assistant.message("This is a test message using the property syntax")
)

print("Sample code executed successfully!")
