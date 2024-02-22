from uuid import uuid1

import pytest

from honcho import (
    AsyncGetDocumentPage,
    AsyncGetMessagePage,
    AsyncGetMetamessagePage,
    AsyncGetSessionPage,
    AsyncSession,
    Document,
    Message,
    Metamessage,
)
from honcho import AsyncHoncho as Honcho


@pytest.mark.asyncio
async def test_session_creation_retrieval():
    app_name = str(uuid1())
    honcho = Honcho(app_name, "http://localhost:8000")
    await honcho.initialize()
    user_name = str(uuid1())
    user = await honcho.create_user(user_name)
    created_session = await user.create_session()
    retrieved_session = await user.get_session(created_session.id)
    assert retrieved_session.id == created_session.id
    assert retrieved_session.is_active is True
    assert retrieved_session.location_id == "default"
    assert retrieved_session.metadata == {}


@pytest.mark.asyncio
async def test_session_multiple_retrieval():
    app_name = str(uuid1())
    user_name = str(uuid1())
    honcho = Honcho(app_name, "http://localhost:8000")
    await honcho.initialize()
    user = await honcho.create_user(user_name)
    created_session_1 = await user.create_session()
    created_session_2 = await user.create_session()
    response = await user.get_sessions()
    retrieved_sessions = response.items

    assert len(retrieved_sessions) == 2
    assert retrieved_sessions[0].id == created_session_1.id
    assert retrieved_sessions[1].id == created_session_2.id


@pytest.mark.asyncio
async def test_session_update():
    user_name = str(uuid1())
    app_name = str(uuid1())
    honcho = Honcho(app_name, "http://localhost:8000")
    await honcho.initialize()
    user = await honcho.create_user(user_name)
    created_session = await user.create_session()
    assert await created_session.update({"foo": "bar"})
    retrieved_session = await user.get_session(created_session.id)
    assert retrieved_session.metadata == {"foo": "bar"}


@pytest.mark.asyncio
async def test_session_deletion():
    user_name = str(uuid1())
    app_name = str(uuid1())
    honcho = Honcho(app_name, "http://localhost:8000")
    await honcho.initialize()
    user = await honcho.create_user(user_name)
    created_session = await user.create_session()
    assert created_session.is_active is True
    await created_session.close()
    assert created_session.is_active is False
    retrieved_session = await user.get_session(created_session.id)
    assert retrieved_session.is_active is False
    assert retrieved_session.id == created_session.id


@pytest.mark.asyncio
async def test_messages():
    user_name = str(uuid1())
    app_name = str(uuid1())
    honcho = Honcho(app_name, "http://localhost:8000")
    await honcho.initialize()
    user = await honcho.create_user(user_name)
    created_session = await user.create_session()
    await created_session.create_message(is_user=True, content="Hello")
    await created_session.create_message(is_user=False, content="Hi")
    retrieved_session = await user.get_session(created_session.id)
    response = await retrieved_session.get_messages()
    messages = response.items
    assert len(messages) == 2
    user_message, ai_message = messages
    assert user_message.content == "Hello"
    assert user_message.is_user is True
    assert ai_message.content == "Hi"
    assert ai_message.is_user is False


@pytest.mark.asyncio
async def test_rate_limit():
    app_name = str(uuid1())
    user_name = str(uuid1())
    honcho = Honcho(app_name, "http://localhost:8000")
    await honcho.initialize()
    user = await honcho.create_user(user_name)
    created_session = await user.create_session()
    with pytest.raises(Exception):
        for _ in range(105):
            await created_session.create_message(is_user=True, content="Hello")
            await created_session.create_message(is_user=False, content="Hi")


@pytest.mark.asyncio
async def test_app_name_security():
    app_name_1 = str(uuid1())
    app_name_2 = str(uuid1())
    user_name = str(uuid1())
    honcho_1 = Honcho(app_name_1, "http://localhost:8000")
    await honcho_1.initialize()
    honcho_2 = Honcho(app_name_2, "http://localhost:8000")
    await honcho_2.initialize()
    user_1 = await honcho_1.create_user(user_name)
    user_2 = await honcho_2.create_user(user_name)
    created_session = await user_1.create_session()
    await created_session.create_message(is_user=True, content="Hello")
    await created_session.create_message(is_user=False, content="Hi")
    with pytest.raises(Exception):
        await user_2.get_session(created_session.id)


@pytest.mark.asyncio
async def test_paginated_sessions():
    app_name = str(uuid1())
    user_name = str(uuid1())
    honcho = Honcho(app_name, "http://localhost:8000")
    await honcho.initialize()
    user = await honcho.create_user(user_name)
    for i in range(10):
        await user.create_session()

    page = 1
    page_size = 2
    get_session_response = await user.get_sessions(page=page, page_size=page_size)
    assert len(get_session_response.items) == page_size

    assert get_session_response.pages == 5

    new_session_response = await get_session_response.next()
    assert new_session_response is not None
    assert isinstance(new_session_response, AsyncGetSessionPage)
    assert len(new_session_response.items) == page_size

    final_page = await user.get_sessions(page=5, page_size=page_size)

    assert len(final_page.items) == 2
    next_page = await final_page.next()
    assert next_page is None


@pytest.mark.asyncio
async def test_paginated_sessions_generator():
    app_name = str(uuid1())
    user_name = str(uuid1())
    honcho = Honcho(app_name, "http://localhost:8000")
    await honcho.initialize()
    user = await honcho.create_user(user_name)
    for i in range(3):
        await user.create_session()

    gen = user.get_sessions_generator()
    # print(type(gen))

    item = await gen.__anext__()
    assert item.user.id == user.id
    assert isinstance(item, AsyncSession)
    assert await gen.__anext__() is not None
    assert await gen.__anext__() is not None
    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()


@pytest.mark.asyncio
async def test_paginated_out_of_bounds():
    app_name = str(uuid1())
    user_name = str(uuid1())
    honcho = Honcho(app_name, "http://localhost:8000")
    await honcho.initialize()
    user = await honcho.create_user(user_name)
    for i in range(3):
        await user.create_session()
    page = 2
    page_size = 50
    get_session_response = await user.get_sessions(page=page, page_size=page_size)

    assert get_session_response.pages == 1
    assert get_session_response.page == 2
    assert get_session_response.page_size == 50
    assert get_session_response.total == 3
    assert len(get_session_response.items) == 0


@pytest.mark.asyncio
async def test_paginated_messages():
    app_name = str(uuid1())
    user_name = str(uuid1())
    honcho = Honcho(app_name, "http://localhost:8000")
    await honcho.initialize()
    user = await honcho.create_user(user_name)
    created_session = await user.create_session()
    for i in range(10):
        await created_session.create_message(is_user=True, content="Hello")
        await created_session.create_message(is_user=False, content="Hi")

    page_size = 7
    get_message_response = await created_session.get_messages(
        page=1, page_size=page_size
    )

    assert get_message_response is not None
    assert isinstance(get_message_response, AsyncGetMessagePage)
    assert len(get_message_response.items) == page_size

    new_message_response = await get_message_response.next()

    assert new_message_response is not None
    assert isinstance(new_message_response, AsyncGetMessagePage)
    assert len(new_message_response.items) == page_size

    final_page = await created_session.get_messages(page=3, page_size=page_size)

    assert len(final_page.items) == 20 - ((3 - 1) * 7)

    next_page = await final_page.next()

    assert next_page is None


@pytest.mark.asyncio
async def test_paginated_messages_generator():
    app_name = str(uuid1())
    user_name = str(uuid1())
    honcho = Honcho(app_name, "http://localhost:8000")
    await honcho.initialize()
    user = await honcho.create_user(user_name)
    created_session = await user.create_session()
    await created_session.create_message(is_user=True, content="Hello")
    await created_session.create_message(is_user=False, content="Hi")
    gen = created_session.get_messages_generator()

    item = await gen.__anext__()
    assert isinstance(item, Message)
    assert item.content == "Hello"
    assert item.is_user is True
    item2 = await gen.__anext__()
    assert item2 is not None
    assert item2.content == "Hi"
    assert item2.is_user is False
    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()


@pytest.mark.asyncio
async def test_paginated_metamessages():
    app_name = str(uuid1())
    user_name = str(uuid1())
    honcho = Honcho(app_name, "http://localhost:8000")
    await honcho.initialize()
    user = await honcho.create_user(user_name)
    created_session = await user.create_session()
    message = await created_session.create_message(is_user=True, content="Hello")
    for i in range(10):
        await created_session.create_metamessage(
            message=message, metamessage_type="thought", content=f"Test {i}"
        )
        await created_session.create_metamessage(
            message=message, metamessage_type="reflect", content=f"Test {i}"
        )

    page_size = 7
    page = await created_session.get_metamessages(page=1, page_size=page_size)

    assert page is not None
    assert isinstance(page, AsyncGetMetamessagePage)
    assert len(page.items) == page_size

    new_page = await page.next()

    assert new_page is not None
    assert isinstance(new_page, AsyncGetMetamessagePage)
    assert len(new_page.items) == page_size

    final_page = await created_session.get_metamessages(page=3, page_size=page_size)

    assert len(final_page.items) == 20 - ((3 - 1) * 7)

    next_page = await final_page.next()

    assert next_page is None


@pytest.mark.asyncio
async def test_paginated_metamessages_generator():
    app_name = str(uuid1())
    user_name = str(uuid1())
    honcho = Honcho(app_name, "http://localhost:8000")
    await honcho.initialize()
    user = await honcho.create_user(user_name)
    created_session = await user.create_session()
    message = await created_session.create_message(is_user=True, content="Hello")
    await created_session.create_metamessage(
        message=message, metamessage_type="thought", content="Test 1"
    )
    await created_session.create_metamessage(
        message=message, metamessage_type="thought", content="Test 2"
    )
    gen = created_session.get_metamessages_generator()

    item = await gen.__anext__()
    assert isinstance(item, Metamessage)
    assert item.content == "Test 1"
    assert item.metamessage_type == "thought"
    item2 = await gen.__anext__()
    assert item2 is not None
    assert item2.content == "Test 2"
    assert item2.metamessage_type == "thought"
    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()


@pytest.mark.asyncio
async def test_collections():
    col_name = str(uuid1())
    app_name = str(uuid1())
    user_name = str(uuid1())
    honcho = Honcho(app_name, "http://localhost:8000")
    await honcho.initialize()
    user = await honcho.create_user(user_name)
    # Make a collection
    collection = await user.create_collection(col_name)

    # Add documents
    doc1 = await collection.create_document(
        content="This is a test of documents - 1", metadata={"foo": "bar"}
    )
    doc2 = await collection.create_document(
        content="This is a test of documents - 2", metadata={}
    )
    doc3 = await collection.create_document(
        content="This is a test of documents - 3", metadata={}
    )

    # Get all documents
    page = await collection.get_documents(page=1, page_size=3)
    # Verify size
    assert page is not None
    assert isinstance(page, AsyncGetDocumentPage)
    assert len(page.items) == 3
    # delete a doc
    result = await collection.delete_document(doc1)
    assert result is True
    # Get all documents with a generator this time
    gen = collection.get_documents_generator()
    # Verfy size
    item = await gen.__anext__()
    item2 = await gen.__anext__()
    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()
    # delete the collection
    result = await collection.delete()
    # confirm documents are gone
    with pytest.raises(Exception):
        new_col = await user.get_collection(col_name)


@pytest.mark.asyncio
async def test_collection_name_collision():
    col_name = str(uuid1())
    new_col_name = str(uuid1())
    app_name = str(uuid1())
    user_name = str(uuid1())
    honcho = Honcho(app_name, "http://localhost:8000")
    await honcho.initialize()
    user = await honcho.create_user(user_name)
    # Make a collection
    collection = await user.create_collection(col_name)
    # Make another collection
    with pytest.raises(Exception):
        await user.create_collection(col_name)

    # Change the name of original collection
    result = await collection.update(new_col_name)
    assert result is True

    # Try again to add another collection
    collection2 = await user.create_collection(col_name)
    assert collection2 is not None
    assert collection2.name == col_name
    assert collection.name == new_col_name

    # Get all collections
    page = await user.get_collections()
    assert page is not None
    assert len(page.items) == 2


@pytest.mark.asyncio
async def test_collection_query():
    col_name = str(uuid1())
    app_name = str(uuid1())
    user_name = str(uuid1())
    honcho = Honcho(app_name, "http://localhost:8000")
    await honcho.initialize()
    user = await honcho.create_user(user_name)
    # Make a collection
    collection = await user.create_collection(col_name)

    # Add documents
    doc1 = await collection.create_document(
        content="The user loves puppies", metadata={}
    )
    doc2 = await collection.create_document(content="The user owns a dog", metadata={})
    doc3 = await collection.create_document(content="The user is a doctor", metadata={})

    result = await collection.query(query="does the user own pets", top_k=2)

    assert result is not None
    assert len(result) == 2
    assert isinstance(result[0], Document)

    doc3 = await collection.update_document(
        doc3, metadata={"test": "test"}, content="the user has owned pets in the past"
    )
    assert doc3 is not None
    assert doc3.metadata == {"test": "test"}
    assert doc3.content == "the user has owned pets in the past"

    result = await collection.query(query="does the user own pets", top_k=2)

    assert result is not None
    assert len(result) == 2
    assert isinstance(result[0], Document)
