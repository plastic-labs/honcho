import datetime
from collections.abc import Sequence
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import Select, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

# from sqlalchemy.orm import Session
from . import models, schemas

load_dotenv(override=True)

openai_client = OpenAI()

########################################################
# app methods
########################################################


async def get_app(db: AsyncSession, app_id: str) -> Optional[models.App]:
    stmt = select(models.App).where(models.App.public_id == app_id)
    result = await db.execute(stmt)
    app = result.scalar_one_or_none()
    return app


async def get_app_by_name(db: AsyncSession, name: str) -> Optional[models.App]:
    stmt = select(models.App).where(models.App.name == name)
    result = await db.execute(stmt)
    app = result.scalar_one_or_none()
    return app


# def get_apps(db: AsyncSession) -> Sequence[models.App]:
#     return db.query(models.App).all()


async def create_app(db: AsyncSession, app: schemas.AppCreate) -> models.App:
    honcho_app = models.App(name=app.name, h_metadata=app.metadata)
    db.add(honcho_app)
    await db.commit()
    # await db.refresh(honcho_app)
    return honcho_app


async def update_app(
    db: AsyncSession, app_id: str, app: schemas.AppUpdate
) -> models.App:
    honcho_app = await get_app(db, app_id)
    if honcho_app is None:
        raise ValueError("App not found")
    if app.name is not None:
        honcho_app.name = app.name
    if app.metadata is not None:
        honcho_app.h_metadata = app.metadata

    await db.commit()
    # await db.refresh(honcho_app)
    return honcho_app


# def delete_app(db: AsyncSession, app_id: str) -> bool:
#     existing_app = get_app(db, app_id)
#     if existing_app is None:
#         return False
#     db.delete(existing_app)
#     db.commit()
#     return True


########################################################
# user methods
########################################################


async def create_user(
    db: AsyncSession, app_id: str, user: schemas.UserCreate
) -> models.User:
    honcho_user = models.User(
        app_id=app_id,
        name=user.name,
        h_metadata=user.metadata,
    )
    db.add(honcho_user)
    await db.commit()
    # await db.refresh(honcho_user)
    return honcho_user


async def get_user(
    db: AsyncSession, app_id: str, user_id: str
) -> Optional[models.User]:
    stmt = (
        select(models.User)
        .where(models.User.app_id == app_id)
        .where(models.User.public_id == user_id)
    )
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    return user


async def get_user_by_name(
    db: AsyncSession, app_id: str, name: str
) -> Optional[models.User]:
    stmt = (
        select(models.User)
        .where(models.User.app_id == app_id)
        .where(models.User.name == name)
    )
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    return user


async def get_users(
    db: AsyncSession,
    app_id: str,
    reverse: bool = False,
    filter: Optional[dict] = None,
) -> Select:
    stmt = select(models.User).where(models.User.app_id == app_id)

    if filter is not None:
        stmt = stmt.where(models.User.h_metadata.contains(filter))

    if reverse:
        stmt = stmt.order_by(models.User.created_at.desc())
    else:
        stmt = stmt.order_by(models.User.created_at)

    return stmt


async def update_user(
    db: AsyncSession, app_id: str, user_id: str, user: schemas.UserUpdate
) -> models.User:
    honcho_user = await get_user(db, app_id, user_id)
    if honcho_user is None:
        raise ValueError("User not found")
    if user.name is not None:
        honcho_user.name = user.name
    if user.metadata is not None:
        honcho_user.h_metadata = user.metadata

    await db.commit()
    # await db.refresh(honcho_user)
    return honcho_user


# def delete_user(db: AsyncSession, app_id: str, user_id: str) -> bool:
#     existing_user = get_user(db, app_id, user_id)
#     if existing_user is None:
#         return False
#     db.delete(existing_user)
#     db.commit()
#     return True

########################################################
# session methods
########################################################


async def get_session(
    db: AsyncSession,
    app_id: str,
    session_id: str,
    user_id: Optional[str] = None,
) -> Optional[models.Session]:
    stmt = (
        select(models.Session)
        .join(models.User, models.User.public_id == models.Session.user_id)
        .where(models.User.app_id == app_id)
        .where(models.Session.public_id == session_id)
    )
    if user_id is not None:
        stmt = stmt.where(models.Session.user_id == user_id)
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()
    return session


async def get_sessions(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    reverse: Optional[bool] = False,
    is_active: Optional[bool] = False,
    filter: Optional[dict] = None,
) -> Select:
    stmt = (
        select(models.Session)
        .join(models.User, models.User.public_id == models.Session.user_id)
        .where(models.User.app_id == app_id)
        .where(models.Session.user_id == user_id)
    )

    if is_active:
        stmt = stmt.where(models.Session.is_active.is_(True))

    if filter is not None:
        stmt = stmt.where(models.Session.h_metadata.contains(filter))

    if reverse:
        stmt = stmt.order_by(models.Session.created_at.desc())
    else:
        stmt = stmt.order_by(models.Session.created_at)

    return stmt


async def create_session(
    db: AsyncSession,
    session: schemas.SessionCreate,
    app_id: str,
    user_id: str,
) -> models.Session:
    honcho_user = await get_user(db, app_id=app_id, user_id=user_id)
    if honcho_user is None:
        raise ValueError("User not found")
    honcho_session = models.Session(
        user_id=user_id,
        h_metadata=session.metadata,
    )
    db.add(honcho_session)
    # print("====== Testing State of ORM Object ====")
    # print(honcho_session)
    # print("=======================================")
    #
    # await db.flush()
    #
    # print("====== Testing State of ORM Object ====")
    # print(honcho_session)
    # print("=======================================")

    await db.commit()
    # await db.refresh(honcho_session)
    return honcho_session


async def update_session(
    db: AsyncSession,
    session: schemas.SessionUpdate,
    app_id: str,
    user_id: str,
    session_id: str,
) -> bool:
    honcho_session = await get_session(
        db, app_id=app_id, session_id=session_id, user_id=user_id
    )
    if honcho_session is None:
        raise ValueError("Session not found or does not belong to user")
    if (
        session.metadata is not None
    ):  # Need to explicitly be there won't make it empty by default
        honcho_session.h_metadata = session.metadata
    await db.commit()
    # await db.refresh(honcho_session)
    return honcho_session


async def delete_session(
    db: AsyncSession, app_id: str, user_id: str, session_id: str
) -> bool:
    stmt = (
        select(models.Session)
        .join(models.User, models.User.public_id == models.Session.user_id)
        .where(models.Session.public_id == session_id)
        .where(models.User.app_id == app_id)
        .where(models.Session.user_id == user_id)
    )
    result = await db.execute(stmt)
    honcho_session = result.scalar_one_or_none()
    if honcho_session is None:
        raise ValueError("Session not found or does not belong to user")
    honcho_session.is_active = False
    await db.commit()
    return True


########################################################
# Message Methods
########################################################


async def create_message(
    db: AsyncSession,
    message: schemas.MessageCreate,
    app_id: str,
    user_id: str,
    session_id: str,
) -> models.Message:
    honcho_session = await get_session(
        db, app_id=app_id, session_id=session_id, user_id=user_id
    )
    if honcho_session is None:
        raise ValueError("Session not found or does not belong to user")

    honcho_message = models.Message(
        session_id=session_id,
        is_user=message.is_user,
        content=message.content,
        h_metadata=message.metadata,
    )
    db.add(honcho_message)
    await db.commit()
    # await db.refresh(honcho_message, attribute_names=["id", "content", "h_metadata"])
    # await db.refresh(honcho_message)
    return honcho_message


async def get_messages(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    session_id: str,
    reverse: Optional[bool] = False,
    filter: Optional[dict] = None,
) -> Select:
    stmt = (
        select(models.Message)
        .join(models.Session, models.Session.public_id == models.Message.session_id)
        .join(models.User, models.User.public_id == models.Session.user_id)
        .join(models.App, models.App.public_id == models.User.app_id)
        .where(models.App.public_id == app_id)
        .where(models.User.public_id == user_id)
        .where(models.Message.session_id == session_id)
    )

    if filter is not None:
        stmt = stmt.where(models.Message.h_metadata.contains(filter))

    if reverse:
        stmt = stmt.order_by(models.Message.created_at.desc())
    else:
        stmt = stmt.order_by(models.Message.created_at)

    return stmt


async def get_message(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    session_id: str,
    message_id: str,
) -> Optional[models.Message]:
    stmt = (
        select(models.Message)
        .join(models.Session, models.Session.public_id == models.Message.session_id)
        .join(models.User, models.User.public_id == models.Session.user_id)
        .join(models.App, models.App.public_id == models.User.app_id)
        .where(models.App.public_id == app_id)
        .where(models.User.public_id == user_id)
        .where(models.Message.session_id == session_id)
        .where(models.Message.public_id == message_id)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def update_message(
    db: AsyncSession,
    message: schemas.MessageUpdate,
    app_id: str,
    user_id: str,
    session_id: str,
    message_id: str,
) -> bool:
    honcho_message = await get_message(
        db, app_id=app_id, session_id=session_id, user_id=user_id, message_id=message_id
    )
    if honcho_message is None:
        raise ValueError("Message not found or does not belong to user")
    if (
        message.metadata is not None
    ):  # Need to explicitly be there won't make it empty by default
        honcho_message.h_metadata = message.metadata
    await db.commit()
    # await db.refresh(honcho_message)
    return honcho_message


########################################################
# metamessage methods
########################################################


async def create_metamessage(
    db: AsyncSession,
    metamessage: schemas.MetamessageCreate,
    app_id: str,
    user_id: str,
    session_id: str,
):
    message = await get_message(
        db,
        app_id=app_id,
        session_id=session_id,
        user_id=user_id,
        message_id=metamessage.message_id,
    )
    if message is None:
        raise ValueError("Session not found or does not belong to user")

    honcho_metamessage = models.Metamessage(
        message_id=metamessage.message_id,
        metamessage_type=metamessage.metamessage_type,
        content=metamessage.content,
        h_metadata=metamessage.metadata,
    )

    db.add(honcho_metamessage)
    await db.commit()
    # await db.refresh(honcho_metamessage)
    return honcho_metamessage


async def get_metamessages(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    session_id: str,
    message_id: Optional[str],
    metamessage_type: Optional[str] = None,
    filter: Optional[dict] = None,
    reverse: Optional[bool] = False,
) -> Select:
    stmt = (
        select(models.Metamessage)
        .join(models.Message, models.Message.public_id == models.Metamessage.message_id)
        .join(models.Session, models.Message.session_id == models.Session.public_id)
        .join(models.User, models.User.public_id == models.Session.user_id)
        .join(models.App, models.App.public_id == models.User.app_id)
        .where(models.App.public_id == app_id)
        .where(models.User.public_id == user_id)
        .where(models.Message.session_id == session_id)
    )

    if message_id is not None:
        stmt = stmt.where(models.Metamessage.message_id == message_id)

    if metamessage_type is not None:
        stmt = stmt.where(models.Metamessage.metamessage_type == metamessage_type)

    if filter is not None:
        stmt = stmt.where(models.Metamessage.h_metadata.contains(filter))

    if reverse:
        stmt = stmt.order_by(models.Metamessage.created_at.desc())
    else:
        stmt = stmt.order_by(models.Metamessage.created_at)

    return stmt


async def get_metamessage(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    session_id: str,
    message_id: str,
    metamessage_id: str,
) -> Optional[models.Metamessage]:
    stmt = (
        select(models.Metamessage)
        .join(models.Message, models.Message.public_id == models.Metamessage.message_id)
        .join(models.Session, models.Message.session_id == models.Session.public_id)
        .join(models.User, models.User.public_id == models.Session.user_id)
        .join(models.App, models.App.public_id == models.User.app_id)
        .where(models.App.public_id == app_id)
        .where(models.User.public_id == user_id)
        .where(models.Message.session_id == session_id)
        .where(models.Metamessage.message_id == message_id)
        .where(models.Metamessage.public_id == metamessage_id)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def update_metamessage(
    db: AsyncSession,
    metamessage: schemas.MetamessageUpdate,
    app_id: str,
    user_id: str,
    session_id: str,
    metamessage_id: str,
) -> bool:
    honcho_metamessage = await get_metamessage(
        db,
        app_id=app_id,
        session_id=session_id,
        user_id=user_id,
        message_id=metamessage.message_id,
        metamessage_id=metamessage_id,
    )
    if honcho_metamessage is None:
        raise ValueError("Metamessage not found or does not belong to user")
    if (
        metamessage.metadata is not None
    ):  # Need to explicitly be there won't make it empty by default
        honcho_metamessage.h_metadata = metamessage.metadata
    if metamessage.metamessage_type is not None:
        honcho_metamessage.metamessage_type = metamessage.metamessage_type
    await db.commit()
    # await db.refresh(honcho_metamessage)
    return honcho_metamessage


########################################################
# collection methods
########################################################

# Should be very similar to the session methods


async def get_collections(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    reverse: Optional[bool] = False,
    filter: Optional[dict] = None,
) -> Select:
    """Get a distinct list of the names of collections associated with a user"""
    stmt = (
        select(models.Collection)
        .join(models.User, models.User.public_id == models.Collection.user_id)
        .where(models.User.app_id == app_id)
        .where(models.User.public_id == user_id)
    )

    if filter is not None:
        stmt = stmt.where(models.Collection.h_metadata.contains(filter))

    if reverse:
        stmt = stmt.order_by(models.Collection.created_at.desc())
    else:
        stmt = stmt.order_by(models.Collection.created_at)

    return stmt


async def get_collection_by_id(
    db: AsyncSession, app_id: str, user_id: str, collection_id: str
) -> Optional[models.Collection]:
    stmt = (
        select(models.Collection)
        .join(models.User, models.User.public_id == models.Collection.user_id)
        .where(models.User.app_id == app_id)
        .where(models.User.public_id == user_id)
        .where(models.Collection.public_id == collection_id)
    )
    result = await db.execute(stmt)
    collection = result.scalar_one_or_none()
    return collection


async def get_collection_by_name(
    db: AsyncSession, app_id: str, user_id: str, name: str
) -> Optional[models.Collection]:
    stmt = (
        select(models.Collection)
        .join(models.User, models.User.public_id == models.Collection.user_id)
        .where(models.User.app_id == app_id)
        .where(models.User.public_id == user_id)
        .where(models.Collection.name == name)
    )
    result = await db.execute(stmt)
    collection = result.scalar_one_or_none()
    return collection


async def create_collection(
    db: AsyncSession,
    collection: schemas.CollectionCreate,
    app_id: str,
    user_id: str,
) -> models.Collection:
    honcho_collection = models.Collection(
        user_id=user_id,
        name=collection.name,
        h_metadata=collection.metadata,
    )
    try:
        db.add(honcho_collection)
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise ValueError("Collection already exists") from None
    # await db.refresh(honcho_collection)
    return honcho_collection


async def update_collection(
    db: AsyncSession,
    collection: schemas.CollectionUpdate,
    app_id: str,
    user_id: str,
    collection_id: str,
) -> models.Collection:
    honcho_collection = await get_collection_by_id(
        db, app_id=app_id, user_id=user_id, collection_id=collection_id
    )
    if honcho_collection is None:
        raise ValueError("collection not found or does not belong to user")
    if collection.metadata is not None:
        honcho_collection.h_metadata = collection.metadata
    try:
        if collection.name is not None:
            honcho_collection.name = collection.name
            await db.commit()
    except IntegrityError:
        await db.rollback()
        raise ValueError("Collection already exists") from None
    # await db.refresh(honcho_collection)
    return honcho_collection


async def delete_collection(
    db: AsyncSession, app_id: str, user_id: str, collection_id: str
) -> bool:
    """
    Delete a Collection and all documents associated with it. Takes advantage of
    the orm cascade feature
    """
    stmt = (
        select(models.Collection)
        .join(models.User, models.User.public_id == models.Collection.user_id)
        .where(models.User.app_id == app_id)
        .where(models.User.public_id == user_id)
        .where(models.Collection.public_id == collection_id)
    )
    result = await db.execute(stmt)
    honcho_collection = result.scalar_one_or_none()
    if honcho_collection is None:
        raise ValueError("collection not found or does not belong to user")
    await db.delete(honcho_collection)
    await db.commit()
    return True


########################################################
# document methods
########################################################

# Should be similar to the messages methods outside of query


async def get_documents(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    collection_id: str,
    reverse: Optional[bool] = False,
    filter: Optional[dict] = None,
) -> Select:
    stmt = (
        select(models.Document)
        .join(
            models.Collection,
            models.Collection.public_id == models.Document.collection_id,
        )
        .join(models.User, models.User.public_id == models.Collection.user_id)
        .where(models.User.app_id == app_id)
        .where(models.User.public_id == user_id)
        .where(models.Document.collection_id == collection_id)
    )

    if filter is not None:
        stmt = stmt.where(models.Document.h_metadata.contains(filter))

    if reverse:
        stmt = stmt.order_by(models.Document.created_at.desc())
    else:
        stmt = stmt.order_by(models.Document.created_at)

    return stmt


async def get_document(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    collection_id: str,
    document_id: str,
) -> Optional[models.Document]:
    stmt = (
        select(models.Document)
        .join(
            models.Collection,
            models.Collection.public_id == models.Document.collection_id,
        )
        .join(models.User, models.User.public_id == models.Collection.user_id)
        .where(models.User.app_id == app_id)
        .where(models.User.public_id == user_id)
        .where(models.Document.collection_id == collection_id)
        .where(models.Document.public_id == document_id)
    )

    result = await db.execute(stmt)
    document = result.scalar_one_or_none()
    return document


async def query_documents(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    collection_id: str,
    query: str,
    filter: Optional[dict] = None,
    top_k: int = 5,
) -> Sequence[models.Document]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-small", input=query
    )
    embedding_query = response.data[0].embedding
    stmt = (
        select(models.Document)
        .join(
            models.Collection,
            models.Collection.public_id == models.Document.collection_id,
        )
        .join(models.User, models.User.public_id == models.Collection.user_id)
        .where(models.User.app_id == app_id)
        .where(models.User.public_id == user_id)
        .where(models.Document.collection_id == collection_id)
        # .limit(top_k)
    )
    if filter is not None:
        stmt = stmt.where(models.Document.h_metadata.contains(filter))
    stmt = stmt.limit(top_k).order_by(
        models.Document.embedding.cosine_distance(embedding_query)
    )
    result = await db.execute(stmt)
    return result.scalars().all()


async def create_document(
    db: AsyncSession,
    document: schemas.DocumentCreate,
    app_id: str,
    user_id: str,
    collection_id: str,
) -> models.Document:
    """Embed a message as a vector and create a document"""
    collection = await get_collection_by_id(
        db, app_id=app_id, collection_id=collection_id, user_id=user_id
    )
    if collection is None:
        raise ValueError("Session not found or does not belong to user")

    response = openai_client.embeddings.create(
        input=document.content, model="text-embedding-3-small"
    )

    embedding = response.data[0].embedding

    honcho_document = models.Document(
        collection_id=collection_id,
        content=document.content,
        h_metadata=document.metadata,
        embedding=embedding,
    )
    db.add(honcho_document)
    await db.commit()
    # await db.refresh(honcho_document)
    return honcho_document


async def update_document(
    db: AsyncSession,
    document: schemas.DocumentUpdate,
    app_id: str,
    user_id: str,
    collection_id: str,
    document_id: str,
) -> bool:
    honcho_document = await get_document(
        db,
        app_id=app_id,
        collection_id=collection_id,
        user_id=user_id,
        document_id=document_id,
    )
    if honcho_document is None:
        raise ValueError("Session not found or does not belong to user")
    if document.content is not None:
        honcho_document.content = document.content
        response = openai_client.embeddings.create(
            input=document.content, model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        honcho_document.embedding = embedding
        honcho_document.created_at = datetime.datetime.utcnow()

    if document.metadata is not None:
        honcho_document.h_metadata = document.metadata
    await db.commit()
    # await db.refresh(honcho_document)
    return honcho_document


async def delete_document(
    db: AsyncSession,
    app_id: str,
    user_id: str,
    collection_id: str,
    document_id: str,
) -> bool:
    stmt = (
        select(models.Document)
        .join(
            models.Collection,
            models.Collection.public_id == models.Document.collection_id,
        )
        .join(models.User, models.User.public_id == models.Collection.user_id)
        .where(models.User.app_id == app_id)
        .where(models.User.public_id == user_id)
        .where(models.Document.collection_id == collection_id)
        .where(models.Document.public_id == document_id)
    )
    result = await db.execute(stmt)
    document = result.scalar_one_or_none()
    if document is None:
        return False
    await db.delete(document)
    await db.commit()
    return True
