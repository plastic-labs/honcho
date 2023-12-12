from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict
import asyncio

# Local
from chain import BloomChain
from mediator import SupabaseMediator
from cache import Conversation

import os
from dotenv import load_dotenv
import sentry_sdk

load_dotenv()

rate = 0.2 if os.getenv("SENTRY_ENVIRONMENT") == "production" else 1.0
sentry_sdk.init(
    dsn=os.environ["SENTRY_DSN"], traces_sample_rate=rate, profiles_sample_rate=rate
)

app = FastAPI()

MEDIATOR = SupabaseMediator()
LOCK = asyncio.Lock()


### User Routes ###
class User(BaseModel):
    user_id: str
    metadata: Optional[Dict]


@app.get("/users")
async def get_users():
    # TODO update to return a list of ID's
    pass


@app.get("/users/{user_id}")
async def get_user(user_id: str):
    # TODO: return user metadata
    pass


### Session Meta Routes ###
class UnknownSession(BaseModel):
    location_id: str = "default"


class SessionMeta(BaseModel):
    location_id: str = "default"
    metadata: Optional[Dict] = None


@app.get("/users/{user_id}/sessions")
async def get_sessions(user_id, inp: UnknownSession):
    """Return session ids and metadata associated with a user and location"""
    print(user_id, inp)
    # TODO update to return a list of ID's
    async with LOCK:
        data: str = MEDIATOR.get_sessions(user_id, inp.location_id)
    print(data)
    return JSONResponse(status_code=200, content=data)


@app.post("/users/{user_id}/sessions")
async def add_session(user_id, inp: SessionMeta):
    async with LOCK:
        data = MEDIATOR.add_session(user_id, inp.location_id, inp.metadata)
    return JSONResponse(status_code=200, content=data)


@app.get("/users/{user_id}/sessions/{session_id}")
async def get_session(user_id, session_id):
    """Return session metadata"""
    async with LOCK:
        data: str = MEDIATOR.get_session(session_id)
    return JSONResponse(status_code=200, content=data)


@app.put("/users/{user_id}/sessions/{session_id}")
async def update_session(user_id, session_id, inp: SessionMeta):
    async with LOCK:
        MEDIATOR.update_session(session_id, inp.metadata)
    return JSONResponse(status_code=200, content={"message": "OK"})


@app.delete("/users/{user_id}/sessions/{session_id}")
async def delete_session(user_id, session_id):
    """Delete a specific session"""
    async with LOCK:
        MEDIATOR.delete_session(session_id)
    return JSONResponse(status_code=200, content={"message": "OK"})


### Session Message Routes ###
class Message(BaseModel):
    message: str
    message_type: str


@app.get("/users/{user_id}/sessions/{session_id}/messages")
async def get_messages(user_id, session_id):
    """Return messages associated with a session"""
    async with LOCK:
        data: str = MEDIATOR.get_messages(session_id, "response")
    return JSONResponse(status_code=200, content=data)


@app.post("/users/{user_id}/sessions/{session_id}/messages")
async def add_message(user_id, session_id, inp: Message):
    """Add a message to a session"""
    async with LOCK:
        MEDIATOR.add_message(session_id, inp.message_type, inp.message)
    return JSONResponse(status_code=200, content={"message": "OK"})


class SessionInput(BaseModel):
    user_id: str
    session_id: str
    message: str
    message_type: str


class Session(BaseModel):
    session_id: str
    user_id: str
    location_id: Optional[str]
    metadata: Optional[Dict]


### API Fundamentals
# @app.post('/tenant/new')
# @app.post('application/new')
# @app.get('/tenant')


## Honcho Utilities
# @app.get('/theoryofmind')
# @app.get('/voe')


@app.post("/chat")
async def voe(inp: SessionInput):
    async with LOCK:
        session = Conversation(MEDIATOR, user_id=inp.user_id, session_id=inp.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Item not found")
    thought, response = await BloomChain.chat(session, inp.message)
    return {"thought": thought, "response": response}


@app.post("/stream")
async def stream(inp: SessionInput):
    async with LOCK:
        session = Conversation(MEDIATOR, user_id=inp.user_id, session_id=inp.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return StreamingResponse(BloomChain.stream(session, inp.message))
