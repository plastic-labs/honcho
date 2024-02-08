import pytest
from honcho import AsyncGetSessionPage, AsyncGetMessagePage, AsyncSession, Message
from honcho import AsyncClient as Honcho
from uuid import uuid1


@pytest.mark.asyncio
async def test_session_creation_retrieval():
    app_id = str(uuid1())
    client = Honcho(app_id, "http://localhost:8000")
    user_id = str(uuid1())
    created_session = await client.create_session(user_id)
    retrieved_session = await client.get_session(user_id, created_session.id)
    assert retrieved_session.id == created_session.id
    assert retrieved_session.is_active is True
    assert retrieved_session.location_id == "default"
    assert retrieved_session.session_data == {}


@pytest.mark.asyncio
async def test_session_multiple_retrieval():
    app_id = str(uuid1())
    client = Honcho(app_id, "http://localhost:8000")
    user_id = str(uuid1())
    created_session_1 = await client.create_session(user_id)
    created_session_2 = await client.create_session(user_id)
    response = await client.get_sessions(user_id)
    retrieved_sessions = response.items

    assert len(retrieved_sessions) == 2
    assert retrieved_sessions[0].id == created_session_1.id
    assert retrieved_sessions[1].id == created_session_2.id


@pytest.mark.asyncio
async def test_session_update():
    user_id = str(uuid1())
    app_id = str(uuid1())
    client = Honcho(app_id, "http://localhost:8000")
    created_session = await client.create_session(user_id)
    assert await created_session.update({"foo": "bar"})
    retrieved_session = await client.get_session(user_id, created_session.id)
    assert retrieved_session.session_data == {"foo": "bar"}


@pytest.mark.asyncio
async def test_session_deletion():
    user_id = str(uuid1())
    app_id = str(uuid1())
    client = Honcho(app_id, "http://localhost:8000")
    created_session = await client.create_session(user_id)
    assert created_session.is_active is True
    await created_session.close()
    assert created_session.is_active is False
    retrieved_session = await client.get_session(user_id, created_session.id)
    assert retrieved_session.is_active is False
    assert retrieved_session.id == created_session.id


@pytest.mark.asyncio
async def test_messages():
    user_id = str(uuid1())
    app_id = str(uuid1())
    client = Honcho(app_id, "http://localhost:8000")
    created_session = await client.create_session(user_id)
    await created_session.create_message(is_user=True, content="Hello")
    await created_session.create_message(is_user=False, content="Hi")
    retrieved_session = await client.get_session(user_id, created_session.id)
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
    app_id = str(uuid1())
    user_id = str(uuid1())
    client = Honcho(app_id, "http://localhost:8000")
    created_session = await client.create_session(user_id)
    with pytest.raises(Exception):
        for _ in range(105):
            await created_session.create_message(is_user=True, content="Hello")
            await created_session.create_message(is_user=False, content="Hi")

@pytest.mark.asyncio
async def test_app_id_security():
    app_id_1 = str(uuid1())
    app_id_2 = str(uuid1())
    user_id = str(uuid1())
    client_1 = Honcho(app_id_1, "http://localhost:8000")
    client_2 = Honcho(app_id_2, "http://localhost:8000")
    created_session = await client_1.create_session(user_id)
    await created_session.create_message(is_user=True, content="Hello")
    await created_session.create_message(is_user=False, content="Hi")
    with pytest.raises(Exception):
        await client_2.get_session(user_id, created_session.id)


@pytest.mark.asyncio
async def test_paginated_sessions():
    app_id = str(uuid1())
    user_id = str(uuid1())
    client = Honcho(app_id, "http://localhost:8000")
    for i in range(10):
        await client.create_session(user_id)
    
    page = 1
    page_size = 2
    get_session_response = await client.get_sessions(user_id, page=page, page_size=page_size)
    assert len(get_session_response.items) == page_size

    assert get_session_response.pages == 5

    new_session_response = await get_session_response.next()
    assert new_session_response is not None
    assert isinstance(new_session_response, AsyncGetSessionPage)
    assert len(new_session_response.items) == page_size

    final_page = await client.get_sessions(user_id, page=5, page_size=page_size)

    assert len(final_page.items) == 2
    next_page = await final_page.next()
    assert next_page is None


@pytest.mark.asyncio
async def test_paginated_sessions_generator():
    app_id = str(uuid1())
    user_id = str(uuid1())
    client = Honcho(app_id, "http://localhost:8000")
    for i in range(3):
        await client.create_session(user_id)

    gen = client.get_sessions_generator(user_id)
    # print(type(gen))

    item = await gen.__anext__()
    assert item.user_id == user_id
    assert isinstance(item, AsyncSession)
    assert await gen.__anext__() is not None
    assert await gen.__anext__() is not None
    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()

@pytest.mark.asyncio
async def test_paginated_out_of_bounds():
    app_id = str(uuid1())
    user_id = str(uuid1())
    client = Honcho(app_id, "http://localhost:8000")
    for i in range(3):
        await client.create_session(user_id)
    page = 2
    page_size = 50
    get_session_response = await client.get_sessions(user_id, page=page, page_size=page_size)

    assert get_session_response.pages == 1
    assert get_session_response.page == 2
    assert get_session_response.page_size == 50
    assert get_session_response.total == 3
    assert len(get_session_response.items) == 0 


@pytest.mark.asyncio
async def test_paginated_messages():
    app_id = str(uuid1())
    user_id = str(uuid1())
    client = Honcho(app_id, "http://localhost:8000")
    created_session = await client.create_session(user_id)
    for i in range(10):
        await created_session.create_message(is_user=True, content="Hello")
        await created_session.create_message(is_user=False, content="Hi")

    page_size = 7
    get_message_response = await created_session.get_messages(page=1, page_size=page_size)

    assert get_message_response is not None
    assert isinstance(get_message_response, AsyncGetMessagePage)
    assert len(get_message_response.items) == page_size

    new_message_response = await get_message_response.next()
    
    assert new_message_response is not None
    assert isinstance(new_message_response, AsyncGetMessagePage)
    assert len(new_message_response.items) == page_size

    final_page = await created_session.get_messages(page=3, page_size=page_size)

    assert len(final_page.items) == 20 - ((3-1) * 7)

    next_page = await final_page.next()

    assert next_page is None


@pytest.mark.asyncio
async def test_paginated_messages_generator():
    app_id = str(uuid1())
    user_id = str(uuid1())
    client = Honcho(app_id, "http://localhost:8000")
    created_session = await client.create_session(user_id)
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

