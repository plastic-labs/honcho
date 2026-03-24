"""
Hermes Agent Gateway Pipeline for Open WebUI

This pipeline integrates Hermes Agent framework with Open WebUI,
allowing users to interact with Hermes agents through the web interface.
"""

import os
import json
import subprocess
from typing import List, Optional, Generator, Iterator
from pydantic import BaseModel, Field


class Pipeline:
    """Open WebUI Pipeline for Hermes Agent Integration"""
    
    class Valves(BaseModel):
        """Configuration values for Hermes integration"""
        HERMES_HOME: str = Field(
            default="/root/.hermes",
            description="Hermes installation directory"
        )
        DEFAULT_AGENT: str = Field(
            default="default",
            description="Default agent to use"
        )
        AWS_BEDROCK_MODEL: str = Field(
            default="us.anthropic.claude-3-5-haiku-20241022-v1:0",
            description="AWS Bedrock model for Hermes"
        )
        ENABLE_STREAMING: bool = Field(
            default=True,
            description="Enable streaming responses"
        )
    
    def __init__(self):
        self.type = "manifold"
        self.id = "hermes_agent"
        self.name = "Hermes Agent"
        self.valves = self.Valves()
        
        # Environment setup
        self.env = os.environ.copy()
        self.env["HERMES_HOME"] = self.valves.HERMES_HOME
        
    def pipes(self) -> List[dict]:
        """Define available agent pipelines"""
        return [
            {
                "id": "hermes-klimashift",
                "name": "KlimaShift Agent (Hermes)"
            },
            {
                "id": "hermes-default",
                "name": "Default Hermes Agent"
            }
        ]
    
    def _run_hermes_command(self, command: List[str], input_text: Optional[str] = None) -> str:
        """Execute a Hermes command"""
        try:
            # Run Hermes via installed CLI
            result = subprocess.run(
                command,
                input=input_text,
                capture_output=True,
                text=True,
                env=self.env,
                timeout=60,
                cwd=self.valves.HERMES_HOME
            )
            
            if result.returncode != 0:
                return f"Error: {result.stderr}"
            
            return result.stdout.strip()
            
        except subprocess.TimeoutExpired:
            return "Error: Request timed out after 60 seconds"
        except Exception as e:
            return f"Error executing Hermes command: {str(e)}"
    
    def _send_message_to_hermes(self, agent_id: str, message: str, conversation_id: Optional[str] = None) -> str:
        """Send a message to Hermes agent"""
        # Use Hermes CLI to send message
        # Format: hermes chat <agent> <message>
        
        command = [
            "bash", "-c",
            f"source ~/.bashrc && hermes chat {agent_id} '{message.replace(\"'\", \"'\\\"'\\\"'\")}'"
        ]
        
        return self._run_hermes_command(command)
    
    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict
    ) -> str | Generator | Iterator:
        """
        Process a message through Hermes agent pipeline
        
        Args:
            user_message: The user's message
            model_id: Selected model/agent ID (e.g., "hermes-klimashift")
            messages: Full conversation history
            body: Request body with additional parameters
            
        Returns:
            Agent response (string or generator for streaming)
        """
        # Extract agent name from model_id
        agent_name = model_id.replace("hermes-", "") if model_id.startswith("hermes-") else self.valves.DEFAULT_AGENT
        
        # Get conversation context (last 5 messages)
        context = []
        for msg in messages[-5:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content and content != user_message:  # Don't duplicate current message
                context.append(f"{role}: {content}")
        
        # Build full prompt with context
        full_prompt = ""
        if context:
            full_prompt = "Previous conversation:\n" + "\n".join(context) + "\n\nCurrent message:\n"
        full_prompt += user_message
        
        # Send to Hermes
        response = self._send_message_to_hermes(agent_name, full_prompt)
        
        if self.valves.ENABLE_STREAMING:
            # Simulate streaming for better UX
            words = response.split()
            def generate_stream():
                for i, word in enumerate(words):
                    yield word + (" " if i < len(words) - 1 else "")
            return generate_stream()
        else:
            return response


# Alternative approach: Use Hermes as an OpenAI-compatible endpoint
class HermesOpenAICompatible(Pipeline):
    """
    OpenAI-compatible wrapper for Hermes
    This approach makes Hermes look like an OpenAI API to Open WebUI
    """
    
    def __init__(self):
        super().__init__()
        self.type = "filter"  # Act as a filter/middleware
    
    def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Pre-process request before sending to LLM"""
        # Add Hermes-specific context or routing
        messages = body.get("messages", [])
        
        # Inject system message about Hermes capabilities
        system_msg = {
            "role": "system",
            "content": (
                "You are a KlimaShift Assistant powered by Hermes Agent Framework. "
                "You can help with energy management, smart meter operations, "
                "site installations, and building specialized AI agents."
            )
        }
        
        # Insert system message if not present
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, system_msg)
        
        body["messages"] = messages
        return body
    
    def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Post-process response from LLM"""
        # Could add logging, analytics, etc.
        return body
