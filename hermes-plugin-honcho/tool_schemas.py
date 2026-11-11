"""Honcho tool schemas sent to the model — byte-stable (prompt-cache parity).

Chat / search / profile / conclude, exposed via HonchoMemoryProvider.get_tool_schemas(). Every read is the
peer's own view of itself.
"""

_PEER = {"type": "string",
         "description": "Peer to query. Built-in aliases: 'user' (default), 'ai'. Or pass any peer ID from this workspace."}


def _tool(name: str, description: str, properties: dict, required: list) -> dict:
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}


PROFILE_SCHEMA = _tool(
    "honcho_profile",
    "Read or write a peer's CARD — a short, curated list of standing facts "
    "about that peer (name, role, preferences, communication style, recurring "
    "patterns). This is the cheapest, fastest Honcho call: no query, no LLM, "
    "just the current card. Pass `card` to overwrite it; omit `card` to read. "
    "An empty read returns a `hint` explaining why (fresh peer, representation "
    "still warming up) — that is NOT an error; the card "
    "accumulates over time from observed conversation. "
    "Related tools: honcho_search to find specific things that were actually "
    "said; honcho_chat for a synthesized answer to a question.",
    {"peer": _PEER,
     "card": {"type": "array", "items": {"type": "string"},
              "description": "New peer card as a list of fact strings. Omit to read the current card."}},
    [],
)

SEARCH_SCHEMA = _tool(
    "honcho_search",
    "Hybrid (semantic + keyword) search over a peer's actual message "
    "history across ALL past sessions they took part in — not just the "
    "current one. Returns RRF-ranked raw message excerpts (what was "
    "literally said, including the assistant's own messages about the "
    "peer), no LLM synthesis. Cheaper and faster than honcho_chat. "
    "Use this to recall specific past facts — 'what did I say about X', "
    "'what was the regimen/decision/config we settled on' — and reason "
    "over the excerpts yourself. For nuanced questions needing synthesis, "
    "use honcho_chat instead.",
    {"query": {"type": "string",
               "description": "What to look for — a topic, keyword, name, or natural-language description of the fact you're trying to recall."},
     "max_tokens": {"type": "integer",
                    "description": "Approximate budget for returned excerpts (default 800, max 2000). Larger budgets return more/longer ranked snippets."},
     "peer": {"type": "string",
              "description": "Whose history to search. Built-in aliases: 'user' (default), 'ai'. Or pass any peer ID from this workspace. Spans every session that peer took part in."}},
    ["query"],
)

CHAT_SCHEMA = _tool(
    "honcho_chat",
    "Ask Honcho a natural-language question about the user and get back a SYNTHESIZED "
    "answer from their memory: Honcho searches raw messages and derived conclusions, "
    "reasons over them and writes a prose answer. This runs an LLM, so it is slower "
    "and costlier than honcho_search. Reach for it whenever the user's preferences, "
    "past decisions or history could shape your response, and when they ask to catch "
    "up or resume earlier work. For a specific fact that was stated, honcho_search is "
    "cheaper.",
    {"query": {"type": "string", "description": "Natural language question"},
     "scope": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}],
               "description": "Scope name, or several, to limit the answer to (e.g. a repository or machine name)."},
     "workspace": {"type": "boolean",
                   "description": "Ask across the whole workspace instead of about the user. Combine with `scope` to limit it.",
                   "default": False},
     "reasoning_level": {"type": "string", "enum": ["minimal", "low", "medium", "high", "max"],
                         "description": "Reasoning budget for this query. Use 'low' for simple lookups, 'medium' for "
                                        "general questions, 'high'/'max' for complex reasoning. Defaults to 'medium'."}},
    ["query"],
)

CONCLUDE_SCHEMA = _tool(
    "honcho_conclude",
    "Write, delete, or list CONCLUSIONS — persistent, derived facts about a peer that "
    "feeds their long-term profile (card + representation). Use this to record "
    "something durable you've learned about the peer (a stable preference, a "
    "correction, a standing constraint) so future sessions carry it forward. "
    "You MUST pass exactly one of `conclusion` (to create), `delete_id` (to "
    "delete), or `list` (to list/search); any other combination is an error. "
    "A deletion ID is an opaque server-generated string: first call with `list=true` "
    "and optionally `query`, then pass the returned ID as `delete_id`. "
    "Deletion exists only for "
    "PII removal — for merely wrong facts, write a corrected conclusion instead; "
    "Honcho self-heals contradictions over time. Listing returns the conclusions "
    "Honcho derived, not the facts stated outright. To read the profile use "
    "honcho_profile, and to search what was said use honcho_search.",
    {"conclusion": {"type": "string",
                    "description": "A factual statement to persist. Provide this when creating a conclusion. Do not send it together with delete_id or list."},
     "delete_id": {"type": "string",
                   "description": "Conclusion ID to delete for PII removal. Provide this when deleting a conclusion. Do not send it together with conclusion or list. Get this id from a prior `list` call — never guess it."},
     "list": {"type": "boolean",
              "description": "Set to true to list or search stored conclusions (with their ids) instead of creating or deleting one. Do not send together with conclusion or delete_id."},
     "query": {"type": "string",
               "description": "Optional semantic search query, used only when `list` is true. Omit to list the most recent conclusions instead of searching."},
     "peer": {"type": "string",
              "description": "The peer the conclusion is ABOUT. Built-in aliases: 'user' (default), 'ai'. Or pass any peer ID from this workspace."}},
    [],
)


ALL_TOOL_SCHEMAS = [CHAT_SCHEMA, SEARCH_SCHEMA, PROFILE_SCHEMA, CONCLUDE_SCHEMA]
