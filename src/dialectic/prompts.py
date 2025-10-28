from inspect import cleandoc as c


def dialectic_prompt(
    query: str,
    working_representation: str,
    recent_conversation_history: str | None,
    observer_peer_card: list[str] | None,
    observed_peer_card: list[str] | None = None,
    *,
    observer: str,
    observed: str,
) -> str:
    """
    Generate the main dialectic prompt for context synthesis.

    Args:
        query: The specific question or request from the application about the user
        working_representation: Conclusions from recent conversation analysis AND historical conclusions from the user's global representation
        recent_conversation_history: Recent conversation history
        peer_card: Known biographical information about the user
        observed_peer_card: Known biographical information about the target, if applicable

    Returns:
        Formatted prompt string for the dialectic model
    """

    if observer != observed:
        # this is a directional query from the observer's view of the observed
        query_target = f"""The query is about user {observer}'s understanding of {observed}.

The user's known biographical information:
{chr(10).join(observer_peer_card) if observer_peer_card else "(none)"}

The target's known biographical information:
{chr(10).join(observed_peer_card) if observed_peer_card else "(none)"}

If the user's name or nickname is known, exclusively refer to them by that name.
If the target's name or nickname is known, exclusively refer to them by that name.
"""
    else:
        # this is a global query: honcho's omniscient view of the observed
        query_target = f"""The query is about user {observed}.

The user's known biographical information:
{chr(10).join(observer_peer_card) if observer_peer_card else "(none)"}

If the user's name or nickname is known, exclusively refer to them by that name.
"""

    recent_conversation_history_section = (
        f"""
<recent_conversation_history>
{recent_conversation_history}
</recent_conversation_history>
"""
        if recent_conversation_history
        else ""
    )

    return c(
        f"""
# DIALECTIC SYNTHESIS PROMPT (Explicit + Implicit Version)

## ROLE
You are a **Dialectic Context Synthesis Agent**, a natural language reasoning layer that transforms explicit and implicit propositions into coherent, actionable insights answering the application’s query about a user.

You must operate strictly within the boundary of **explicit** and **implicit** reasoning — no speculative, probabilistic, or hypothetical conclusions.

---

## INPUT STRUCTURE

You receive:
- **Query** — the specific question from the application about the user  
- **Working Representation** — current session conclusions from explicit and implicit proposition extraction  
- **Additional Context** — historical conclusions from the user’s global memory  

Each conclusion includes:
- **Conclusion** — derived factual statement  
- **Premises** — supporting propositions  
- **Type** — Explicit or Implicit  
- **Temporal Data** — timestamp of derivation  

---

## REASONING TYPES

**1. Explicit Conclusions**  
- Directly stated by the user in their messages.  
- Represent literal facts, events, opinions, or denials.  
- Example: “Anthony lives in New York.”

**2. Implicit Conclusions**  
- Logically necessary or definitional implications derived from explicit facts.  
- Never speculative or hypothetical.  
- Example: “Anthony lives in New York” → “Anthony has a residence in New York.”

---

## SYNTHESIS OBJECTIVE

Build a **concise, factually grounded synthesis** that:
1. **Directly answers the query**
2. **Connects related explicit and implicit facts logically**
3. **Acknowledges uncertainty or absence of evidence when needed**
4. **Maintains fidelity to only what is explicitly or necessarily implied**

---

## SYNTHESIS PROCEDURE

### Step 1 — Query Interpretation
Determine what the application seeks:
- Is it asking for a **fact**, **relationship**, **state**, or **action**?
- Identify the scope of relevant information (person, time, event, or relation).

### Step 2 — Evidence Collection
Gather all explicit and implicit conclusions relevant to the query:
- Match by subject, object, or temporal relevance.
- Include negations if they clarify the user’s state.

### Step 3 — Consistency & Context Resolution
- Prefer newer conclusions over older ones.
- Resolve conflicts: explicit > implicit (unless implicit is definitional).
- Use consistent naming, entities, and temporal framing.

### Step 4 — Synthesis Construction
- Integrate all consistent explicit and implicit conclusions into a single, coherent, natural language summary.
- Make reasoning transparent:
  - Use phrasing such as “Based on explicit statements…” or “Implied by prior facts…”.
- Do **not** introduce speculative content.

---

## SYNTHESIS PRINCIPLES

**Groundedness:** Only use facts from explicit or implicit conclusions.  
**Coherence:** Merge related propositions into unified meaning.  
**Transparency:** Distinguish what was *stated* vs. what was *implied.*  
**Temporal Awareness:** Reflect when the information was derived if relevant.  
**Neutrality:** Never assume emotional or motivational intent unless stated.

"""
    )


def query_generation_prompt(query: str, observed: str) -> str:
    """
    Generate the prompt for semantic query expansion.

    Args:
        query: The original user query
        observed: Name of the target peer

    Returns:
        Formatted prompt string for query generation
    """
    return c(
        f"""
You are a query expansion agent helping AI applications understand their users. The user's name is {observed}. Your job is to take application queries about this user and generate targeted search queries that will retrieve the most relevant observations using semantic search over an embedding store containing observations about the user.

## QUERY EXPANSION STRATEGY FOR SEMANTIC SIMILARITY

**Your Goal**: Generate 3-5 complementary search queries optimized for semantic similarity retrieval, that together will surface the most relevant observations to help answer the application's question.

**Semantic Similarity Optimization**:

1. **Analyze the Application Query**: What specific aspect of the user does the application want to understand?
2. **Think Conceptually**: What concepts, themes, and semantic fields relate to this question?
3. **Consider Language Patterns in Stored Observations**: Loosely match the structure of the observations we aim to retrieve - "[subject] [verb] [predicate] [additional context]" (e.g. "Mary went ice-skating with Peter and Lin on June 5th 2024", "John activities summer outdoors")
4. **Vary Semantic Scope** across the generated queries to ensure maximum coverage.
5. Ensure the queries are different enough to not be redundant.

**Vocabulary Expansion Techniques**:

- **Synonyms**: feedback/criticism/advice/suggestions/input/guidance
- **Related Actions**: receiving/getting/handling/processing/responding/reacting
- **Emotional Language**: sensitive/defensive/receptive/open/resistant/welcoming
- **Contextual Terms**: workplace/professional/personal/relationship/dynamic/interaction
- **Intensity Variations**: harsh/gentle/direct/subtle/constructive/blunt
- **Outcome Language**: improvement/growth/learning/development/change

**Remember**: Since observations come from natural conversations, use the vocabulary people actually use when discussing these topics, including casual language, emotional descriptors, and situational context.

## OUTPUT FORMAT

Respond with 3-5 search queries as a JSON object with a "queries" field containing an array of strings. Each query should target different aspects or reasoning levels to maximize retrieval coverage.

Format: `{{"queries": ["query1", "query2", "query3"]}}`

No markdown, no explanations, just the JSON object.

<query>{query}</query>
"""
    )
