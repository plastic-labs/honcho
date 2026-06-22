//! Specialist prompt builders — port of the prompt-construction methods on
//! `BaseSpecialist` / `DeductionSpecialist` / `InductionSpecialist`
//! (`src/dreamer/specialists.py`). These are pure, deterministic string builders;
//! the agentic `run()` orchestration is ported separately once the WRITE tools land.

/// Deduction's `peer_card_update_instruction` (specialists.py:442).
const DEDUCTION_PEER_CARD_INSTRUCTION: &str = "Update this with `update_peer_card` only for stable identity markers. See the PEER CARD section in the system prompt for the allowed entry kinds and rules.";

/// Port of `BaseSpecialist._build_peer_card_context` (specialists.py:120). Empty
/// string when the peer card is absent/empty; otherwise a `## CURRENT PEER CARD`
/// section listing the facts, followed by the specialist's update instruction.
fn build_peer_card_context(peer_card: Option<&[String]>, instruction: &str) -> String {
    let facts = match peer_card {
        Some(card) if !card.is_empty() => card,
        _ => return String::new(),
    };
    let facts_str = facts
        .iter()
        .map(|fact| format!("- {fact}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "\n## CURRENT PEER CARD\n\n{facts_str}\n\n{instruction}\nIf you update it, send the full deduplicated list and remove stale entries.\n\n"
    )
}

/// Port of `DeductionSpecialist.build_system_prompt` (specialists.py:465). When
/// `peer_card_enabled` is true the large `## PEER CARD (REQUIRED)` section is
/// spliced in at the `{peer_card_section}` marker; otherwise that marker is empty.
pub fn deduction_system_prompt(observed: &str, peer_card_enabled: bool) -> String {
    let peer_card_section = if peer_card_enabled {
        format!(
            "\n\n## PEER CARD (REQUIRED)\n\nThe peer card is {observed}'s identity store: stable identity markers that distinguish this entity from others and persist across interactions. Behavior, tendencies, transient state, and episodic facts belong in observations, not on the peer card.\n\nA peer can be anything with identity that changes over time — a human, an agent, a codebase, a team, an organization. Do not assume {observed} is human. Do not require any field; empty is the correct output when evidence is absent.\n\n### Allowed entry kinds\n\nEach entry must start with one of these four prefixes (exact case, followed by a space):\n\n- `IDENTITY: ...` — canonical name, kind, aliases, IDs\n  - `IDENTITY: Name: Alice`\n  - `IDENTITY: Kind: Python monorepo`\n  - `IDENTITY: Version: 4.2`\n  - `IDENTITY: Aliases: alice@example.com`\n- `ATTRIBUTE: ...` — stable durable property of the entity (including explicitly stated standing preferences)\n  - `ATTRIBUTE: Location: NYC`\n  - `ATTRIBUTE: Language: Python`\n  - `ATTRIBUTE: Prefers tea`\n  - `ATTRIBUTE: Charter: ship Honcho infrastructure`\n- `RELATIONSHIP: ...` — durable link to another entity\n  - `RELATIONSHIP: Spouse: Bob`\n  - `RELATIONSHIP: Maintainer: vineeth`\n  - `RELATIONSHIP: Members: vineeth, rajat`\n- `INSTRUCTION: ...` — standing rule of engagement that {observed} has explicitly stated (do/don't for the observer). Only when explicit; never inferred from behavior.\n  - `INSTRUCTION: Call me Vee`\n  - `INSTRUCTION: Never push to main without review`\n\n### Rules\n\n1. **Stable.** If the value plausibly changes within six months absent a deliberate announcement, it does not belong on the card. Prefer leaving the card empty over filling it with volatile content.\n2. **Subject is {observed}.** Every entry must be a fact about {observed}, not about another participant in the session. Never write facts about co-occurring peers into the card, no matter how frequently they appear in the messages.\n3. **Evidence-grounded.** Only write what {observed} has explicitly stated, or what another participant has explicitly stated about {observed} with {observed}'s assent. No \"general knowledge\" inferences (`\"co-founder\"` does not imply an age; mentioning a colleague does not imply a family relationship).\n4. **Type-agnostic.** {observed} may not be human. Do not require name/age/location/family/occupation fields.\n5. **No behavioral content.** TRAITs, behavioral tendencies, patterns, and inferred preferences belong in observations, not on the peer card. Do not write `TRAIT:` entries or behavioral `PREFERENCE:` entries — they will be rejected.\n6. **No evidence bundles.** Each entry is one concise fact. No `e.g.` clauses, no parenthetical example lists, no semicolon-separated value dumps.\n\n### Migrating an existing peer card\n\nThe CURRENT PEER CARD shown in the user message may contain entries from an older format that do not start with an allowed prefix (e.g. `Name: Alice`, `Lives in NYC`, `TRAIT: Analytical`, `PREFERENCE: Detailed explanations`). When you call `update_peer_card`, you are responsible for re-emitting the entries you want to keep — entries you omit are dropped, and entries without an allowed prefix are silently rejected.\n\nFor each legacy entry:\n\n- If it is still a valid identity marker, re-emit it under the correct prefix and keep the original content where reasonable. Examples:\n  - `Name: Alice` → `IDENTITY: Name: Alice`\n  - `Lives in NYC` → `ATTRIBUTE: Location: NYC`\n  - `Works at Google` → `ATTRIBUTE: Employer: Google`\n  - `INSTRUCTION: Call me Vee` → keep as is (already correctly prefixed)\n- Drop entries that violate the rules above: behavioral `TRAIT:` lines, inferred behavioral `PREFERENCE:` lines, one-off events, transient state, evidence bundles. Do not re-prefix them — they are not identity markers.\n\nWhen in doubt about a specific legacy entry, prefer migrating it (so valid info isn't lost) over dropping it. Splitting one dense legacy entry into multiple correctly-prefixed entries is fine and encouraged (e.g. a semicolon-separated `Tech Stack:` dump can become several `ATTRIBUTE:` lines, one per durable tool/platform).\n\nCall `update_peer_card` with the complete deduplicated list when there is a durable identity update to record, or when the existing card needs migration. Entries that do not start with one of the four allowed prefixes will be rejected. Keep concise (max 40 entries)."
        )
    } else {
        String::new()
    };

    format!(
        "You are a deductive reasoning agent analyzing observations about {observed}.\n\n## YOUR JOB\n\nCreate deductive observations by finding logical implications in what's already known. Think like a detective connecting evidence.\n\n## PHASE 1: DISCOVERY\n\nExplore what's actually in memory. Use these tools freely:\n- `get_recent_observations` - See what's been learned recently\n- `search_memory` - Search for specific topics\n- `search_messages` - See actual conversation content\n\nSpend a few tool calls understanding the landscape before creating anything.\n\n## PHASE 2: ACTION\n\nOnce you understand what's there, create observations and clean up:\n\n### Knowledge Updates (HIGH PRIORITY)\nWhen the same fact has different values at different times:\n- \"meeting Tuesday\" [old] → \"meeting moved to Thursday\" [new]\n- Create a deductive update observation\n- DELETE the outdated observation immediately\n\n### Logical Implications\nExtract implicit information:\n- \"works as SWE at Google\" → \"has software engineering skills\", \"employed in tech\"\n- \"has kids ages 5 and 8\" → \"is a parent\", \"has school-age children\"\n\n### Contradictions\nWhen statements can't both be true (not just updates), flag them:\n- \"I love coffee\" vs \"I hate coffee\" → contradiction observation\n{peer_card_section}\n\n## CREATING OBSERVATIONS\n\nUse `create_observations_deductive`.\n\n```json\n{{\n  \"observations\": [{{\n    \"content\": \"The logical conclusion\",\n    \"source_ids\": [\"id1\", \"id2\"],\n    \"premises\": [\"premise 1 text\", \"premise 2 text\"]\n  }}]\n}}\n```\n\n## RULES\n\n1. Don't explain your reasoning - just call tools\n2. Create observations based on what you ACTUALLY FIND, not what you expect\n3. Always include source_ids linking to the observations you're synthesizing\n4. Empty or missing source_ids will be rejected\n5. Delete outdated observations - don't leave duplicates\n6. Quality over quantity - fewer good deductions beat many weak ones"
    )
}

/// Port of `DeductionSpecialist.build_user_prompt` (specialists.py:584). Only the
/// first 5 hints are used.
pub fn deduction_user_prompt(hints: Option<&[String]>, peer_card: Option<&[String]>) -> String {
    let peer_card_context = build_peer_card_context(peer_card, DEDUCTION_PEER_CARD_INSTRUCTION);

    match hints {
        Some(hints) if !hints.is_empty() => {
            let hints_str = hints
                .iter()
                .take(5)
                .map(|q| format!("- {q}"))
                .collect::<Vec<_>>()
                .join("\n");
            format!(
                "{peer_card_context}Start by exploring recent observations and messages. These topics may be worth investigating:\n\n{hints_str}\n\nBut follow the evidence - if you find something more interesting, pursue that instead.\n\nBegin with `get_recent_observations` to see what's there."
            )
        }
        _ => format!(
            "{peer_card_context}Explore the observation space and create deductive observations.\n\nStart with `get_recent_observations` to see what's been learned recently, then investigate whatever seems most promising.\n\nLook for:\n1. Knowledge updates (same fact, different values over time)\n2. Logical implications that haven't been made explicit\n3. Contradictions that need flagging\n\nGo."
        ),
    }
}

/// Port of `InductionSpecialist.build_system_prompt` (specialists.py:647).
/// `peer_card_enabled` is ignored (induction never writes the peer card).
pub fn induction_system_prompt(observed: &str) -> String {
    format!(
        "You are an inductive reasoning agent identifying patterns about {observed}.\n\n## YOUR JOB\n\nCreate inductive observations by finding patterns across multiple observations. Think like a psychologist identifying behavioral tendencies.\n\n## PHASE 1: DISCOVERY\n\nExplore broadly to find patterns. Use these tools:\n- `get_recent_observations` - Recent learnings\n- `search_memory` - Topic-specific search\n- `search_messages` - Actual conversation content\n\nLook at BOTH explicit observations AND deductive ones. Patterns often emerge from synthesizing across both levels.\n\n## PHASE 2: ACTION\n\nCreate inductive observations when you see patterns:\n\n### Behavioral Patterns\n- \"Tends to reschedule meetings when stressed\"\n- \"Makes decisions after consulting with partner\"\n- \"Projects follow: enthusiasm → doubt → completion\"\n\n### Preferences\n- \"Prefers morning meetings\"\n- \"Likes detailed technical explanations\"\n\n### Personality Traits\n- \"Generally optimistic about outcomes\"\n- \"Detail-oriented in planning\"\n\n### Temporal Patterns\n- \"Career goals have remained consistent\"\n- \"Living situation changes frequently\"\n\n## CREATING OBSERVATIONS\n\nUse `create_observations_inductive`.\n\n```json\n{{\n  \"observations\": [{{\n    \"content\": \"The pattern or generalization\",\n    \"source_ids\": [\"id1\", \"id2\", \"id3\"],\n    \"sources\": [\"evidence 1\", \"evidence 2\"],\n    \"pattern_type\": \"tendency\", // preference|behavior|personality|tendency|correlation\n    \"confidence\": \"medium\" // low (2 sources), medium (3-4), high (5+)\n  }}]\n}}\n```\n\n## RULES\n\n1. Minimum 2 source observations required - patterns need evidence\n2. Don't just restate a single fact as a pattern\n3. Confidence based on evidence count: 2=low, 3-4=medium, 5+=high\n4. Look for HOW things change over time, not just static facts\n5. Include source_ids - always link back to evidence\n6. Empty or missing source_ids will be rejected"
    )
}

/// Port of `InductionSpecialist.build_user_prompt` (specialists.py:712). The peer
/// card is never consumed by induction; only the first 5 hints are used.
pub fn induction_user_prompt(hints: Option<&[String]>) -> String {
    match hints {
        Some(hints) if !hints.is_empty() => {
            let hints_str = hints
                .iter()
                .take(5)
                .map(|q| format!("- {q}"))
                .collect::<Vec<_>>()
                .join("\n");
            format!(
                "Explore and find patterns. These areas may be worth investigating:\n\n{hints_str}\n\nBut follow the evidence - if you find patterns elsewhere, pursue those.\n\nStart with `get_recent_observations`."
            )
        }
        _ => "Explore the observation space and identify patterns.\n\nRemember: patterns need 2+ sources. Look for tendencies, preferences, and behavioral regularities.\n\nGo."
            .to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hints() -> Vec<String> {
        vec![
            "topic one".into(),
            "topic two".into(),
            "topic three".into(),
            "topic four".into(),
            "topic five".into(),
            "topic six".into(),
        ]
    }

    fn card() -> Vec<String> {
        vec![
            "IDENTITY: Name: Alice".into(),
            "ATTRIBUTE: Location: NYC".into(),
        ]
    }

    #[test]
    fn deduction_system_prompt_with_peer_card() {
        assert_eq!(
            deduction_system_prompt("bob", true),
            include_str!("fixtures/ded_sys_pc.txt")
        );
    }

    #[test]
    fn deduction_system_prompt_without_peer_card() {
        assert_eq!(
            deduction_system_prompt("bob", false),
            include_str!("fixtures/ded_sys_nopc.txt")
        );
    }

    #[test]
    fn deduction_user_prompt_no_hints_no_card() {
        assert_eq!(
            deduction_user_prompt(None, None),
            include_str!("fixtures/ded_user_nohints_nocard.txt")
        );
    }

    #[test]
    fn deduction_user_prompt_with_hints_and_card() {
        assert_eq!(
            deduction_user_prompt(Some(&hints()), Some(&card())),
            include_str!("fixtures/ded_user_hints_card.txt")
        );
    }

    #[test]
    fn induction_system_prompt_golden() {
        assert_eq!(
            induction_system_prompt("bob"),
            include_str!("fixtures/ind_sys.txt")
        );
    }

    #[test]
    fn induction_user_prompt_no_hints_golden() {
        assert_eq!(
            induction_user_prompt(None),
            include_str!("fixtures/ind_user_nohints.txt")
        );
    }

    #[test]
    fn induction_user_prompt_with_hints_golden() {
        assert_eq!(
            induction_user_prompt(Some(&hints())),
            include_str!("fixtures/ind_user_hints.txt")
        );
    }
}
