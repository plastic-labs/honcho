import uuid

from honcho import Honcho

# Create a Honcho client with the default workspace
honcho = Honcho(environment="local")

# Create a new session
session = honcho.session("file_upload_test_" + str(uuid.uuid4()))

# Upload the current file directly using a file object
with open(__file__, "rb") as file:
    session.upload_file(file, peer="alice")

# get the messages from the session
# should contain the contents of this file!
messages = session.messages()
for message in messages:
    print(str(message))
