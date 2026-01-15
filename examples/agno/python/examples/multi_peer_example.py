"""
Multi-Peer Honcho + Agno Example

A realistic multi-agent scenario using Agno's patterns:
- A coordinator agent routes questions to specialists
- Each specialist has its own HonchoTools (identity)
- All share the same session for conversation continuity
- The coordinator uses specialists as tools

Environment Variables:
    OPENAI_API_KEY or LLM_OPENAI_API_KEY: OpenAI API key
    HONCHO_ENVIRONMENT: 'local' or 'production' (default: production)
    HONCHO_API_KEY: Required for production environment
"""

import os
import uuid

from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool

from honcho import Honcho
from honcho_agno import HonchoTools

load_dotenv()

if not os.getenv("OPENAI_API_KEY") and os.getenv("LLM_OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("LLM_OPENAI_API_KEY")


def create_advisor_system(honcho_env: str, session_id: str):
    """
    Creates a multi-agent advisory system where:
    - Each specialist agent has its own identity (HonchoTools)
    - A coordinator routes to specialists and synthesizes responses
    - All agents share the same conversation session
    """
    model_id = os.getenv("OPENAI_MODEL", "gpt-4o")

    # Shared Honcho client and session
    honcho = Honcho(workspace_id="advisory-system", environment=honcho_env)
    session = honcho.session(session_id)
    user_peer = honcho.peer("user")

    # === SPECIALIST AGENTS ===
    # Each has its own identity via HonchoTools

    tech_tools = HonchoTools(
        app_id="advisory-system",
        peer_id="tech-specialist",
        session_id=session_id,
        environment=honcho_env,
        honcho_client=honcho,
    )

    tech_agent = Agent(
        name="Tech Specialist",
        model=OpenAIChat(id=model_id),
        tools=[tech_tools],
        description="Technical advisor for architecture, implementation, and technology choices.",
        instructions=[
            "Focus on technical feasibility and implementation details.",
            "Use get_context to understand what's been discussed.",
            "Save key technical recommendations with add_message.",
            "Be concise - you're part of a team.",
        ],
    )

    business_tools = HonchoTools(
        app_id="advisory-system",
        peer_id="business-specialist",
        session_id=session_id,
        environment=honcho_env,
        honcho_client=honcho,
    )

    business_agent = Agent(
        name="Business Specialist",
        model=OpenAIChat(id=model_id),
        tools=[business_tools],
        description="Business advisor for strategy, market fit, and ROI.",
        instructions=[
            "Focus on business viability and market considerations.",
            "Use get_context to understand what's been discussed.",
            "Save key business insights with add_message.",
            "Be concise - you're part of a team.",
        ],
    )

    # === COORDINATOR TOOLS ===
    # Wrap specialists as tools the coordinator can invoke

    @tool
    def consult_tech_specialist(question: str) -> str:
        """
        Consult the technical specialist for architecture, implementation,
        or technology-related questions.

        Args:
            question: The technical question to ask.

        Returns:
            Technical specialist's response.
        """
        response = tech_agent.run(question)
        return response.content

    @tool
    def consult_business_specialist(question: str) -> str:
        """
        Consult the business specialist for strategy, market fit,
        or ROI-related questions.

        Args:
            question: The business question to ask.

        Returns:
            Business specialist's response.
        """
        response = business_agent.run(question)
        return response.content

    # Coordinator has its own identity too
    coordinator_tools = HonchoTools(
        app_id="advisory-system",
        peer_id="coordinator",
        session_id=session_id,
        environment=honcho_env,
        honcho_client=honcho,
    )

    coordinator = Agent(
        name="Advisory Coordinator",
        model=OpenAIChat(id=model_id),
        tools=[coordinator_tools, consult_tech_specialist, consult_business_specialist],
        description="Coordinates between specialists to provide comprehensive advice.",
        instructions=[
            "Use get_context to understand the full conversation history.",
            "Route technical questions to the tech specialist.",
            "Route business questions to the business specialist.",
            "Synthesize specialist inputs into actionable recommendations.",
            "Save your final synthesis with add_message.",
        ],
    )

    return coordinator, session, user_peer


def main(test_mode: bool = False):
    honcho_env = os.getenv("HONCHO_ENVIRONMENT", "production")
    session_id = f"advisory-{uuid.uuid4().hex[:8]}"

    print(f"Session: {session_id}")
    print("=" * 60)

    coordinator, session, user_peer = create_advisor_system(honcho_env, session_id)

    if test_mode:
        # Non-interactive test
        test_question = "I want to build a SaaS product for small businesses. What should I consider?"
        print(f"\n[TEST MODE] User: {test_question}\n")

        session.add_messages([user_peer.message(test_question)])
        response = coordinator.run(test_question)
        print(f"Advisor: {response.content}\n")
        print("=" * 60)
        print("Test completed successfully!")
        return

    # Interactive chat loop
    print("\nAdvisory System Ready")
    print("Ask questions about building a product. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        # Save user message to session
        session.add_messages([user_peer.message(user_input)])

        # Coordinator handles routing and synthesis
        response = coordinator.run(user_input)
        print(f"\nAdvisor: {response.content}\n")


if __name__ == "__main__":
    import sys
    test_mode = "--test" in sys.argv
    main(test_mode=test_mode)
