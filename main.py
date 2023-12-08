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
    dsn=os.environ['SENTRY_DSN'],
    traces_sample_rate=rate,
    profiles_sample_rate=rate
)

app = FastAPI()

MEDIATOR = SupabaseMediator()
LOCK = asyncio.Lock()

class SessionInput(BaseModel):
    user_id: str
    session_id: str
    message: str
    message_type: str

@app.get("/")
def root():
    return {"message": "Hello World"}

class Session(BaseModel):
    session_id: str
    user_id: str
    location_id: Optional[str]
    metadata: Optional[Dict]

### Session Interface Routes ###
@app.get('/messages')
async def get_messages(inp: Session):
    async with LOCK:
        return MEDIATOR.messages(inp.session_id, inp.user_id, 'response')

@app.post('/messages/add')
async def add_message(inp: SessionInput):
    async with LOCK:
        MEDIATOR.add_message(inp.session_id, inp.user_id, inp.message_type, inp.message)

### Session Meta Routes ###
class UnknownSession(BaseModel):
    user_id: str
    location_id: Optional[str]

@app.get('/session')
async def get_session(inp: UnknownSession):
    """Return session ids and metadata associated with a user and location"""
    location = "default" if inp.location_id is None else inp.location_id 
    async with LOCK:
        id: str = MEDIATOR.session(location, inp.user_id)
    return JSONResponse(status_code=200, content={"session_id": id})

@app.post('/session/add')
async def add_session(inp: UnknownSession):
    location = "default" if inp.location_id is None else inp.location_id 
    async with LOCK:
        MEDIATOR.add_session(location, inp.user_id)
    return JSONResponse(status_code=200, content={"message": "OK"})

class SessionMeta(BaseModel):
    session_id: str
    user_id: str
    metadata: Optional[Dict]

@app.delete('/session/delete')
async def delete_session(inp: SessionMeta):
    """Delete a specific session"""
    async with LOCK:
        MEDIATOR.delete_session(inp.user_id, inp.session_id)
    return JSONResponse(status_code=200, content={"message": "OK"})

@app.patch('/session/update')
async def update_session(inp: SessionMeta):
    async with LOCK:
        MEDIATOR.update_session(inp.user_id, inp.session_id, inp.metadata)
    return JSONResponse(status_code=200, content={"message": "OK"})

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
    return {
        "thought": thought,
        "response": response
    }

@app.post("/stream")
async def stream(inp: SessionInput):
    async with LOCK:
        session = Conversation(MEDIATOR, user_id=inp.user_id, session_id=inp.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return StreamingResponse(BloomChain.stream(session, inp.message))
