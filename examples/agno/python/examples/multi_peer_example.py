"""
Multi-Peer Honcho + Agno Example

A three-way conversation between:
- User: asking questions about life, work, and meaning
- Tech Bro Advisor: startup culture, hustle, optimization mindset
- Philosophy Guru: mindfulness, ancient wisdom, inner peace

All three peers observe each other and build representations on each other,
creating a rich understanding of each participant's perspective over time.

Environment Variables:
    LLM_OPENAI_API_KEY: OpenAI API key (matches honcho .env)
    HONCHO_API_KEY: Required for Honcho API access
"""

import os
import uuid

from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIChat

from honcho import Honcho
from honcho.session import SessionPeerConfig
from honcho_agno import HonchoTools

load_dotenv()

# Use LLM_OPENAI_API_KEY from honcho .env
if llm_key := os.getenv("LLM_OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = llm_key


def create_advisory_session(session_id: str):
    """
    Creates a three-peer advisory system where:
    - User asks questions
    - Tech Bro gives startup/optimization perspective
    - Philosophy Guru gives mindfulness/wisdom perspective
    - All three observe each other and build representations
    """
    model_id = os.getenv("OPENAI_MODEL", "gpt-4o")

    # Shared Honcho client
    honcho = Honcho(workspace_id="advisory-trio")

    # === TECH BRO ADVISOR ===
    tech_bro_tools = HonchoTools(
        peer_id="tech-bro",
        session_id=session_id,
        honcho_client=honcho,
    )

    tech_bro_agent = Agent(
        name="Tech Bro Advisor",
        model=OpenAIChat(id=model_id),
        tools=[tech_bro_tools],
        description="Startup founder vibes, optimization mindset, hustle culture perspective.",
        instructions=[
            "You're a successful tech entrepreneur who's been through YC and raised Series B.",
            "Everything is an opportunity to optimize, scale, or disrupt.",
            "Use the honcho_chat tool to understand what the user is dealing with and what they care about.",
            "Give advice through the lens of productivity, systems thinking, and growth hacking.",
            "Reference things like morning routines, cold plunges, biohacking, and 10x thinking.",
            "Be enthusiastic but genuine - you really believe this stuff works.",
            "Keep responses conversational and punchy.",
        ],
    )

    # === PHILOSOPHY MEDITATION GURU ===
    guru_tools = HonchoTools(
        peer_id="philosophy-guru",
        session_id=session_id,
        honcho_client=honcho,
    )

    guru_agent = Agent(
        name="Philosophy Guru",
        model=OpenAIChat(id=model_id),
        tools=[guru_tools],
        description="Meditation teacher, draws on Stoicism, Buddhism, and Taoism.",
        instructions=[
            "You're a calm, wise meditation teacher who's spent years studying ancient philosophy.",
            "Draw on Stoicism, Buddhism, Taoism, and other contemplative traditions.",
            "Use the honcho_chat tool to understand the user's inner state and what they truly seek.",
            "Gently guide toward presence, acceptance, and inner peace.",
            "Reference concepts like impermanence, the present moment, letting go, and wu wei.",
            "Offer a counterbalance to hustle culture - not everything needs to be optimized.",
            "Speak slowly and thoughtfully. Use metaphors from nature.",
        ],
    )

    # Create user peer and configure session observation
    user_peer = honcho.peer("user")
    session = tech_bro_tools.session  # Use session from toolkit

    # Add all peers to session and configure observation
    session.add_peers([user_peer, tech_bro_tools.peer, guru_tools.peer])

    full_observation = SessionPeerConfig(
        observe_me=True,
        observe_others=True
    )
    session.set_peer_config(user_peer, full_observation)
    session.set_peer_config(tech_bro_tools.peer, full_observation)
    session.set_peer_config(guru_tools.peer, full_observation)

    return session, user_peer, tech_bro_tools, guru_tools, tech_bro_agent, guru_agent


def main():
    session_id = f"trio-{uuid.uuid4().hex[:8]}"

    print(f"Session: {session_id}")
    print("=" * 60)

    session, user_peer, tech_bro_tools, guru_tools, tech_bro_agent, guru_agent = (
        create_advisory_session(session_id)
    )

    print("\nAdvisory Trio Ready")
    print("Ask about life, work, meaning - get two very different perspectives.")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        # Save user message
        session.add_messages([user_peer.message(user_input)])

        # Tech Bro responds
        print()
        print("-" * 40)
        tech_response = tech_bro_agent.run(user_input)
        tech_content = str(tech_response.content) if tech_response.content else ""
        session.add_messages([tech_bro_tools.peer.message(tech_content)])
        print(f"Tech Bro: {tech_content}\n")

        # Guru responds
        print("-" * 40)
        guru_response = guru_agent.run(user_input)
        guru_content = str(guru_response.content) if guru_response.content else ""
        session.add_messages([guru_tools.peer.message(guru_content)])
        print(f"Guru: {guru_content}\n")


if __name__ == "__main__":
    main()
