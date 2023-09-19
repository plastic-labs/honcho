from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
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

class ConversationInput(BaseModel):
    user_id: str
    conversation_id: str
    message: str

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/chat")
async def voe(inp: ConversationInput):
    async with LOCK:
        conversation = Conversation(MEDIATOR, user_id=inp.user_id, conversation_id=inp.conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Item not found")
    thought, response = await BloomChain.chat(conversation, inp.message)
    return {
        "thought": thought,
        "response": response
    }

@app.post("/stream")
async def stream(inp: ConversationInput):
    async with LOCK:
        conversation = Conversation(MEDIATOR, user_id=inp.user_id, conversation_id=inp.conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return StreamingResponse(BloomChain.stream(conversation, inp.message))
