import uuid
from typing import Optional, Sequence

from openai import OpenAI

from sqlalchemy import select, Select
from sqlalchemy.orm import Session

from . import models, schemas

openai_client = OpenAI()

def get_session(db: Session, app_id: str, session_id: uuid.UUID, user_id: Optional[str] = None) -> Optional[models.Session]:
    stmt = select(models.Session).where(models.Session.app_id == app_id).where(models.Session.id == session_id)
    if user_id is not None:
        stmt = stmt.where(models.Session.user_id == user_id)
    session = db.scalars(stmt).one_or_none()
    return session

def get_sessions(
        db: Session, app_id: str, user_id: str, location_id: str | None = None
) -> Select:
    stmt = (
        select(models.Session)
        .where(models.Session.app_id == app_id)
        .where(models.Session.user_id == user_id)
        .where(models.Session.is_active.is_(True))
        .order_by(models.Session.created_at)
    )

    if location_id is not None:
        stmt = stmt.where(models.Session.location_id == location_id)

    return stmt

def create_session(
    db: Session, session: schemas.SessionCreate, app_id: str, user_id: str
) -> models.Session:
    honcho_session = models.Session(
        app_id=app_id,
        user_id=user_id,
        location_id=session.location_id,
        h_metadata=session.metadata,
    )
    db.add(honcho_session)
    db.commit()
    db.refresh(honcho_session)
    return honcho_session


def update_session(
    db: Session, session: schemas.SessionUpdate, app_id: str, user_id: str, session_id: uuid.UUID
) -> bool:
    honcho_session = get_session(db, app_id=app_id, session_id=session_id, user_id=user_id)
    if honcho_session is None:
        raise ValueError("Session not found or does not belong to user")
    if session.metadata is not None: # Need to explicitly be there won't make it empty by default
        honcho_session.h_metadata = session.metadata
    db.commit()
    db.refresh(honcho_session)
    return honcho_session

def delete_session(db: Session, app_id: str, user_id: str, session_id: uuid.UUID) -> bool:
    stmt = (
        select(models.Session)
        .where(models.Session.id == session_id)
        .where(models.Session.app_id == app_id)
        .where(models.Session.user_id == user_id)
    )
    honcho_session = db.scalars(stmt).one_or_none()
    if honcho_session is None:
        return False
    honcho_session.is_active = False
    db.commit()
    return True

def create_message(
        db: Session, message: schemas.MessageCreate, app_id: str, user_id: str, session_id: uuid.UUID
) -> models.Message:
    honcho_session = get_session(db, app_id=app_id, session_id=session_id, user_id=user_id)
    if honcho_session is None:
        raise ValueError("Session not found or does not belong to user")

    honcho_message = models.Message(
        session_id=session_id,
        is_user=message.is_user,
        content=message.content,
    )
    db.add(honcho_message)
    db.commit()
    db.refresh(honcho_message)
    return honcho_message

def get_messages(
    db: Session, app_id: str, user_id: str, session_id: uuid.UUID
) -> Select:
    stmt = (
        select(models.Message)
        .join(models.Session, models.Session.id == models.Message.session_id)
        .where(models.Session.app_id == app_id)
        .where(models.Session.user_id == user_id)
        .where(models.Message.session_id == session_id)
        .order_by(models.Message.created_at)
    )
    return stmt

def get_message(
        db: Session, app_id: str, user_id: str, session_id: uuid.UUID, message_id: uuid.UUID     
) -> Optional[models.Message]:
    stmt = (
        select(models.Message)
        .join(models.Session, models.Session.id == models.Message.session_id)
        .where(models.Session.app_id == app_id)
        .where(models.Session.user_id == user_id)
        .where(models.Message.session_id == session_id)
        .where(models.Message.id == message_id)

    )
    return db.scalars(stmt).one_or_none()

########################################################
# metamessage methods
########################################################

def get_metamessages(db: Session, app_id: str, user_id: str, session_id: uuid.UUID, message_id: Optional[uuid.UUID], metamessage_type: Optional[str] = None) -> Select:
    stmt = (
        select(models.Metamessage)
        .join(models.Message, models.Message.id == models.Metamessage.message_id)
        .join(models.Session, models.Message.session_id == models.Session.id)
        .where(models.Session.app_id == app_id)
        .where(models.Session.user_id == user_id)
        .where(models.Message.session_id == session_id)
        .order_by(models.Metamessage.created_at)
    )
    if message_id is not None:
        stmt = stmt.where(models.Metamessage.message_id == message_id)
    if metamessage_type is not None:
        stmt = stmt.where(models.Metamessage.metamessage_type == metamessage_type)
    return stmt

def get_metamessage(
        db: Session, app_id: str, user_id: str, session_id: uuid.UUID, message_id: uuid.UUID, metamessage_id: uuid.UUID
) -> Optional[models.Metamessage]: 
    stmt = (
         select(models.Metamessage)
        .join(models.Message, models.Message.id == models.Metamessage.message_id)
        .join(models.Session, models.Message.session_id == models.Session.id)
        .where(models.Session.app_id == app_id)
        .where(models.Session.user_id == user_id)
        .where(models.Message.session_id == session_id)
        .where(models.Metamessage.message_id == message_id)
        .where(models.Metamessage.id == metamessage_id)
       
    )
    return db.scalars(stmt).one_or_none()

def create_metamessage(
    db: Session,
    metamessage: schemas.MetamessageCreate,
    app_id: str,
    user_id: str,
    session_id: uuid.UUID,
):
    message = get_message(db, app_id=app_id, session_id=session_id, user_id=user_id, message_id=metamessage.message_id)
    if message is None:
        raise ValueError("Session not found or does not belong to user")

    honcho_metamessage = models.Metamessage(
        message_id=metamessage.message_id,
        metamessage_type=metamessage.metamessage_type,
        content=metamessage.content,
    )

    db.add(honcho_metamessage)
    db.commit()
    db.refresh(honcho_metamessage)
    return honcho_metamessage

########################################################
# vector methods
########################################################

# Should be very similar to the session methods

def get_vectors(db: Session, app_id: str, user_id: str) -> Select:
    """Get a distinct list of the names of vectors associated with a user"""
    stmt = (
        select(models.VectorCollection)
        .where(models.VectorCollection.app_id == app_id)
        .where(models.VectorCollection.user_id == user_id)
        .order_by(models.VectorCollection.created_at)
    )
    return stmt

def get_vector(db: Session, app_id: str, user_id: str, vector_id: uuid.UUID, name: Optional[str] = None ) -> Optional[models.VectorCollection]:
    stmt = ( 
        select(models.VectorCollection)
        .where(models.VectorCollection.app_id == app_id)
        .where(models.VectorCollection.user_id == user_id)
        .where(models.VectorCollection.id == vector_id)
    )
    # TODO determine if indexing by name or id is better 
    if name is not None:
        stmt = stmt.where(models.VectorCollection.name == name)

    vector = db.scalars(stmt).one_or_none()
    return vector

def create_vector(
    db: Session, vector: schemas.VectorCreate, app_id: str, user_id: str
) -> models.VectorCollection:
    honcho_vector = models.VectorCollection(
        app_id=app_id,
        user_id=user_id,
        location_id=vector.name,
    )
    db.add(honcho_vector)
    db.commit()
    db.refresh(honcho_vector)
    return honcho_vector

def update_vector(
        db: Session, vector: schemas.VectorUpdate, app_id: str, user_id: str, vector_id: uuid.UUID
) -> models.VectorCollection:
    honcho_vector = get_vector(db, app_id=app_id, user_id=user_id, vector_id=vector_id)
    if honcho_vector is None:
        raise ValueError("vector not found or does not belong to user")
    honcho_vector.name = vector.name
    db.commit()
    db.refresh(honcho_vector)
    return honcho_vector

def delete_vector(
    db: Session, app_id: str, user_id: str, vector_id: uuid.UUID
) -> bool:
    """
    Delete a Vector Collection and all documents associated with it. Takes advantage of
    the orm cascade feature
    """
    stmt = (
        select(models.VectorCollection)
        .where(models.VectorCollection.id == vector_id)
        .where(models.VectorCollection.app_id == app_id)
        .where(models.VectorCollection.user_id == user_id)
    )
    honcho_vector = db.scalars(stmt).one_or_none()
    if honcho_vector is None:
        return False
    db.delete(honcho_vector)
    db.commit()
    return True

########################################################
# document methods
########################################################

# Should be similar to the messages methods outside of query

def get_documents(
    db: Session, app_id: str, user_id: str, vector_id: uuid.UUID
) -> Select:
    stmt = (
        select(models.Document)
        .join(models.VectorCollection, models.VectorCollection.id == models.Document.vector_id)
        .where(models.VectorCollection.app_id == app_id)
        .where(models.VectorCollection.user_id == user_id)
        .where(models.Document.vector_id == vector_id)
        .order_by(models.Document.created_at)
    )
    return stmt

def get_document(
        db: Session, app_id: str, user_id: str, vector_id: uuid.UUID, document_id: uuid.UUID 
) -> Optional[models.Document]:
    stmt = ( 
        select(models.Document)
        .join(models.VectorCollection, models.VectorCollection.id == models.Document.vector_id)
        .where(models.VectorCollection.app_id == app_id)
        .where(models.VectorCollection.user_id == user_id)
        .where(models.Document.vector_id == vector_id)
        .where(models.Document.id == document_id)
    )

    vector = db.scalars(stmt).one_or_none()
    return vector


def query_documents(db: Session, app_id: str, user_id: str, vector_id: uuid.UUID, query: str, top_k: int = 5) -> Sequence[models.Document]:
    response = openai_client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    embedding_query = response.data[0].embedding
    stmt = (
            select(models.Document)
            .join(models.VectorCollection, models.VectorCollection.id == models.Document.vector_id)
            .where(models.VectorCollection.app_id == app_id)
            .where(models.VectorCollection.user_id == user_id)
            .where(models.Document.vector_id == vector_id)
            .order_by(models.Document.cosine_distance(embedding_query))
            .limit(top_k)
            )
    # if metadata is not None:
        # stmt = stmt.where(models.Document.h_metadata.contains(metadata))
    return db.scalars(stmt).all()

def create_document(
        db: Session, document: schemas.DocumentCreate, app_id: str, user_id: str, vector_id: uuid.UUID
) -> models.Document:
    """Embed a message as a vector and create a document"""
    vector = get_vector(db, app_id=app_id, vector_id=vector_id, user_id=user_id)
    if vector is None:
        raise ValueError("Session not found or does not belong to user")

    response = openai_client.embeddings.create(
            input=document.content, 
            model="text-embedding-3-small"
    )

    embedding = response.data[0].embedding

    honcho_document = models.Document(
        vector_id=vector_id,
        content=document.content,
        h_metadata=document.metadata,
        embedding=embedding
    )
    db.add(honcho_document)
    db.commit()
    db.refresh(honcho_document)
    return honcho_document

def update_document(
        db: Session, document: schemas.DocumentUpdate, app_id: str, user_id: str, vector_id: uuid.UUID, document_id: uuid.UUID
) -> bool:
    honcho_document = get_document(db, app_id=app_id, vector_id=vector_id, user_id=user_id, document_id=document_id)
    if honcho_document is None:
        raise ValueError("Session not found or does not belong to user")
    if document.content is not None:
        honcho_document.content = document.content
        response = openai_client.embeddings.create(
                    input=document.content, 
                    model="text-embedding-3-small"
            )

        embedding = response.data[0].embedding
        honcho_document.embedding = embedding

    if document.metadata is not None:
        honcho_document.h_metadata = document.metadata
    db.commit()
    db.refresh(honcho_document)
    return honcho_document

def delete_document(db: Session, app_id: str, user_id: str, vector_id: uuid.UUID, document_id: uuid.UUID) -> bool:
    stmt = (
        select(models.Document)
        .where(models.Document.app_id == app_id)
        .where(models.Document.user_id == user_id)
        .where(models.Document.vector_id == vector_id)
        .where(models.Document.id == document_id)
    )
    document = db.scalars(stmt).one_or_none()
    if document is None:
        return False
    db.delete(document)
    db.commit()
    return True


