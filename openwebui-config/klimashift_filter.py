"""
title: KlimaShift Assistant with Honcho Memory
author: Avrio Energy
author_url: https://github.com/avrio
funding_url: https://github.com/open-webui
version: 1.0.0
license: MIT
description: Integrates KlimaShift Assistant personality with Honcho memory system
requirements: requests, pydantic
"""

import os
import requests
from typing import Optional
from pydantic import BaseModel, Field


class Filter:
    """Open WebUI Filter for KlimaShift + Honcho Integration"""
    
    class Valves(BaseModel):
        """Configuration values"""
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
        ENABLE_KLIMASHIFT_PERSONALITY: bool = Field(
            default=True,
            description="Inject KlimaShift Assistant personality"
        )
    
    def __init__(self):
        self.valves = self.Valves()
        self.honcho_api = os.getenv("HONCHO_API_URL", "http://api:8000")
        self.workspace_id = os.getenv("HONCHO_WORKSPACE_ID", "klimashift-bot")
        
        # KlimaShift personality
        self.klimashift_personality = """You are KlimaShift Assistant, the AI agent for Avrio Energy's energy intelligence platform.

**Your Role:**
Help with energy management, smart meter operations, site installations, and building specialized AI agents from templates.

**Your Style:**
- Be direct, clear, and technically precise
- Use bullet points for operational tasks
- Keep answers compact unless depth is requested
- Include concrete commands, SQL queries, or file paths when relevant

**Audience Awareness:**
- With technical team (Sanjay, Tushar, engineers): Be terse, include DB queries and SSH commands
- With customers: Be friendly, jargon-free, focus on outcomes

**What You Can Help With:**
- Energy management and smart meter operations
- Site installations and hardware troubleshooting
- Creating specialized AI agents from templates
- Database queries and system operations
- Climate data analysis and recommendations"""
    
    def get_or_create_peer(self, user_id: str) -> Optional[str]:
        """Get or create a Honcho peer for the user"""
        try:
            # Check for existing peer using v3 API (POST method)
            response = requests.post(
                f"{self.honcho_api}/v3/workspaces/{self.workspace_id}/peers/list",
                json={"name": user_id},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                peers = data.get("items", [])
                if peers and len(peers) > 0:
                    print(f"[KlimaShift] Found existing peer: {peers[0]['id']}")
                    return peers[0]["id"]
            
            # Create new peer using v3 API
            print(f"[KlimaShift] Creating new peer for user: {user_id}")
            response = requests.post(
                f"{self.honcho_api}/v3/workspaces/{self.workspace_id}/peers",
                json={
                    "name": user_id,
                    "metadata": {"source": "openwebui"}
                },
                timeout=5
            )
            
            if response.status_code in [200, 201]:
                peer_id = response.json()["id"]
                print(f"[KlimaShift] Created peer: {peer_id}")
                return peer_id
            else:
                print(f"[KlimaShift] Peer creation failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"[KlimaShift] Error with peer: {e}")
        
        return None
    
    def get_memory_context(self, peer_id: str, query: str) -> str:
        """Retrieve relevant memories from Honcho"""
        try:
            # Use v3 API chat endpoint
            response = requests.post(
                f"{self.honcho_api}/v3/workspaces/{self.workspace_id}/peers/{peer_id}/chat",
                json={
                    "query": query,
                    "agentic": False,
                    "max_tokens": 300
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("content", "")
            
        except Exception as e:
            print(f"[KlimaShift] Error retrieving memory: {e}")
        
        return ""
    
    def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Pre-process request before sending to LLM
        Inject KlimaShift personality and Honcho memory context
        """
        print(f"[KlimaShift] Filter activated for user: {user.get('id') if user else 'unknown'}")
        
        messages = body.get("messages", [])
        if not messages:
            return body
        
        # Get user info
        user_id = user.get("id", "anonymous") if user else "anonymous"
        last_message = messages[-1].get("content", "") if messages else ""
        
        # Store peer_id for outlet
        if self.valves.ENABLE_MEMORY:
            peer_id = self.get_or_create_peer(user_id)
            body["__klimashift_peer_id"] = peer_id
            
            # Get memory context
            if peer_id:
                memory_context = self.get_memory_context(peer_id, last_message)
                body["__klimashift_memory"] = memory_context
        
        # Inject KlimaShift personality
        if self.valves.ENABLE_KLIMASHIFT_PERSONALITY:
            system_content = self.klimashift_personality
            
            # Add memory context if available
            if self.valves.ENABLE_MEMORY and body.get("__klimashift_memory"):
                system_content += f"\n\n**Context from previous conversations:**\n{body['__klimashift_memory']}"
            
            # Find or create system message
            system_msg_idx = None
            for i, msg in enumerate(messages):
                if msg.get("role") == "system":
                    system_msg_idx = i
                    break
            
            if system_msg_idx is not None:
                # Prepend to existing system message
                existing = messages[system_msg_idx].get("content", "")
                messages[system_msg_idx]["content"] = f"{system_content}\n\n{existing}"
            else:
                # Insert new system message at start
                messages.insert(0, {"role": "system", "content": system_content})
            
            body["messages"] = messages
            print(f"[KlimaShift] Personality injected, memory: {bool(body.get('__klimashift_memory'))}")
        
        return body
    
    def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Post-process response after LLM generation
        Save conversation to Honcho
        """
        if not self.valves.ENABLE_MEMORY:
            return body
        
        peer_id = body.get("__klimashift_peer_id")
        if not peer_id:
            return body
        
        # Extract conversation
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
        
        # Save to Honcho using v3 API
        if user_msg and assistant_msg:
            try:
                chat_id = body.get("chat_id", "openwebui-session")
                
                # Create or get session
                session_response = requests.post(
                    f"{self.honcho_api}/v3/workspaces/{self.workspace_id}/sessions",
                    json={
                        "location_id": chat_id,
                        "metadata": {"source": "openwebui"}
                    },
                    timeout=5
                )
                
                if session_response.status_code in [200, 201, 409]:  # 409 = already exists
                    if session_response.status_code == 409:
                        # Session exists, get it
                        list_response = requests.post(
                            f"{self.honcho_api}/v3/workspaces/{self.workspace_id}/sessions/list",
                            json={"location_id": chat_id},
                            timeout=5
                        )
                        if list_response.status_code == 200:
                            data = list_response.json()
                            sessions = data.get("items", [])
                            if sessions:
                                session_id = sessions[0]["id"]
                            else:
                                print(f"[KlimaShift] No session found for location_id: {chat_id}")
                                return body
                    else:
                        session_id = session_response.json()["id"]
                    
                    # Add peer to session if needed
                    requests.post(
                        f"{self.honcho_api}/v3/workspaces/{self.workspace_id}/sessions/{session_id}/peers",
                        json={"peer_id": peer_id},
                        timeout=5
                    )
                    
                    # Save messages
                    messages_to_save = [
                        {"peer_id": peer_id, "content": user_msg, "metadata": {"role": "user"}},
                        {"peer_id": peer_id, "content": assistant_msg, "metadata": {"role": "assistant"}}
                    ]
                    
                    requests.post(
                        f"{self.honcho_api}/v3/workspaces/{self.workspace_id}/sessions/{session_id}/messages",
                        json={"messages": messages_to_save},
                        timeout=5
                    )
                    
                    print(f"[KlimaShift] Conversation saved to Honcho")
            except Exception as e:
                print(f"[KlimaShift] Error saving to Honcho: {e}")
        
        # Clean up temporary data
        body.pop("__klimashift_peer_id", None)
        body.pop("__klimashift_memory", None)
        
        return body
