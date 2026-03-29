"""
Honcho Memory Integration for Open WebUI

This function integrates Honcho's memory system with Open WebUI to provide
personalized, context-aware responses based on stored user memories.
"""

import os
import requests
from typing import Optional, Generator
from pydantic import BaseModel, Field


class Filter:
    """Open WebUI Filter for Honcho Memory Integration"""
    
    class Valves(BaseModel):
        """Configuration values for Honcho integration"""
        HONCHO_API_URL: str = Field(
            default="http://api:8000",
            description="Honcho API base URL"
        )
        HONCHO_WORKSPACE_ID: str = Field(
            default="klimashift-bot",
            description="Honcho workspace ID"
        )
        ENABLE_MEMORY: bool = Field(
            default=True,
            description="Enable memory retrieval"
        )
        MAX_CONTEXT_TOKENS: int = Field(
            default=500,
            description="Maximum tokens for memory context"
        )
    
    def __init__(self):
        self.valves = self.Valves()
        self.honcho_api = os.getenv("HONCHO_API_URL", "http://api:8000")
        self.workspace_id = os.getenv("HONCHO_WORKSPACE_ID", "klimashift-bot")
    
    def get_or_create_peer(self, user_id: str) -> Optional[str]:
        """Get or create a Honcho peer for the user"""
        try:
            # Try to get existing peer
            response = requests.get(
                f"{self.honcho_api}/v1/peers",
                params={
                    "workspace_id": self.workspace_id,
                    "name": user_id
                },
                timeout=5
            )
            
            if response.status_code == 200:
                peers = response.json()
                if peers and len(peers) > 0:
                    return peers[0]["id"]
            
            # Create new peer
            response = requests.post(
                f"{self.honcho_api}/v1/peers",
                json={
                    "workspace_id": self.workspace_id,
                    "name": user_id,
                    "metadata": {"source": "openwebui"}
                },
                timeout=5
            )
            
            if response.status_code == 200:
                return response.json()["id"]
                
        except Exception as e:
            print(f"Error getting/creating peer: {e}")
        
        return None
    
    def get_honcho_context(self, peer_id: str, query: str) -> str:
        """Retrieve relevant memories from Honcho"""
        try:
            # Use Honcho's dialectic chat endpoint for context-aware retrieval
            response = requests.post(
                f"{self.honcho_api}/v1/peers/{peer_id}/chat",
                json={
                    "query": query,
                    "agentic": False,  # Use non-agentic for faster response
                    "max_tokens": self.valves.MAX_CONTEXT_TOKENS
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("content", "")
            
        except Exception as e:
            print(f"Error retrieving Honcho context: {e}")
        
        return ""
    
    def save_interaction(self, peer_id: str, session_id: str, user_msg: str, assistant_msg: str):
        """Save the conversation to Honcho"""
        try:
            # Save user message
            requests.post(
                f"{self.honcho_api}/v1/messages",
                json={
                    "session_id": session_id,
                    "peer_id": peer_id,
                    "content": user_msg,
                    "metadata": {"role": "user"}
                },
                timeout=5
            )
            
            # Save assistant message
            requests.post(
                f"{self.honcho_api}/v1/messages",
                json={
                    "session_id": session_id,
                    "peer_id": peer_id,
                    "content": assistant_msg,
                    "metadata": {"role": "assistant"}
                },
                timeout=5
            )
        except Exception as e:
            print(f"Error saving interaction: {e}")
    
    def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Pre-process request before sending to LLM
        Inject Honcho memory context into the conversation
        """
        if not self.valves.ENABLE_MEMORY:
            return body
        
        if not user:
            return body
        
        user_id = user.get("id", "anonymous")
        
        # Get or create Honcho peer
        peer_id = self.get_or_create_peer(user_id)
        if not peer_id:
            print("Failed to get/create peer, skipping memory")
            return body
        
        # Store peer_id for use in outlet
        body["__honcho_peer_id"] = peer_id
        
        # Get user's last message
        messages = body.get("messages", [])
        if not messages:
            return body
        
        last_message = messages[-1]
        if last_message.get("role") != "user":
            return body
        
        user_query = last_message.get("content", "")
        
        # Retrieve relevant memories
        memory_context = self.get_honcho_context(peer_id, user_query)
        
        # Prepare context and personality injection
        klimashift_personality = """You are KlimaShift Assistant, powered by Avrio Energy's AI platform.

Your role: Help with energy management, smart meter operations, site installations, and building specialized AI agents from templates.

Style:
- Be direct, clear, and technically precise
- Use bullet points for operational tasks
- Keep answers compact unless depth is requested
- Include concrete commands or file paths when relevant

With technical team: Be terse, include DB queries and commands
With customers: Be friendly, jargon-free, focus on outcomes"""

        # Find or create system message
        system_msg_idx = None
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                system_msg_idx = i
                break
        
        if system_msg_idx is not None:
            # Append to existing system message
            existing_content = messages[system_msg_idx].get("content", "")
            messages[system_msg_idx]["content"] = f"{klimashift_personality}\n\n{existing_content}"
            if memory_context:
                messages[system_msg_idx]["content"] += f"\n\n[MEMORY CONTEXT]\n{memory_context}"
        else:
            # Create new system message
            content = klimashift_personality
            if memory_context:
                content += f"\n\n[MEMORY CONTEXT]\n{memory_context}"
            messages.insert(0, {"role": "system", "content": content})
        
        body["messages"] = messages
        
        return body
    
    def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Post-process response after LLM generation
        Save the interaction to Honcho for future memory
        """
        if not self.valves.ENABLE_MEMORY:
            return body
        
        # Extract peer_id stored during inlet
        peer_id = body.get("__honcho_peer_id")
        if not peer_id:
            return body
        
        # Get session ID from body or create one
        session_id = body.get("chat_id", "openwebui-session")
        
        # Get the conversation
        messages = body.get("messages", [])
        if len(messages) < 2:
            return body
        
        # Find last user and assistant messages
        user_msg = None
        assistant_msg = None
        
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and not assistant_msg:
                assistant_msg = msg.get("content", "")
            elif msg.get("role") == "user" and not user_msg:
                user_msg = msg.get("content", "")
            
            if user_msg and assistant_msg:
                break
        
        if user_msg and assistant_msg:
            self.save_interaction(peer_id, session_id, user_msg, assistant_msg)
        
        # Clean up temporary data
        if "__honcho_peer_id" in body:
            del body["__honcho_peer_id"]
        
        return body
