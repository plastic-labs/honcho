"""
RefusalBench Perturbation Lever Catalogue

A complete catalogue of 90 linguistic perturbation levers for testing RAG systems.
Ported from the RefusalBench reference implementation.

6 perturbation classes x 3 intensity levels x 5 levers = 90 levers total.

Classes:
    P-Ambiguity, P-Contradiction, P-MissingInfo,
    P-FalsePremise, P-GranularityMismatch, P-EpistemicMismatch

Intensities:
    LOW (answer correctly), MEDIUM (refuse), HIGH (refuse)

Reference: arXiv:2510.10390
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Lever:
    """A single perturbation lever with instruction and illustration."""

    name: str
    instruction: str
    illustration: str


@dataclass
class PerturbationSpec:
    """Specification for a perturbation class."""

    target: str  # "Query", "Context", or "Query↔Context"
    refusal_ground_truth: str  # Expected refusal type for MEDIUM/HIGH


class RefusalBenchCatalogue:
    """Complete catalogue of 90 RefusalBench perturbation levers."""

    def __init__(self) -> None:
        self.perturbation_specs: dict[str, PerturbationSpec] = {
            "P-Ambiguity": PerturbationSpec(
                "Query↔Context", "REFUSE_AMBIGUOUS_QUERY"
            ),
            "P-Contradiction": PerturbationSpec(
                "Context", "REFUSE_CONTRADICTORY_CONTEXT"
            ),
            "P-MissingInfo": PerturbationSpec(
                "Context", "REFUSE_INFO_MISSING_IN_CONTEXT"
            ),
            "P-FalsePremise": PerturbationSpec(
                "Query", "REFUSE_FALSE_PREMISE_IN_QUERY"
            ),
            "P-GranularityMismatch": PerturbationSpec(
                "Query↔Context", "REFUSE_GRANULARITY_MISMATCH"
            ),
            "P-EpistemicMismatch": PerturbationSpec(
                "Query↔Context", "REFUSE_NONFACTUAL_QUERY"
            ),
        }

        self.levers: dict[str, dict[str, list[Lever]]] = self._build_catalogue()

    def _format_answers_for_prompt(
        self, answers: list[str], max_display: int | None = None
    ) -> str:
        """Format answers for display in prompts."""
        if len(answers) == 1:
            return f'"{answers[0]}"'
        elif max_display is None or len(answers) <= max_display:
            quoted = [f'"{ans}"' for ans in answers]
            return " OR ".join(quoted)
        else:
            quoted = [f'"{ans}"' for ans in answers[:max_display]]
            remaining = len(answers) - max_display
            return f"{', '.join(quoted)} (and {remaining} other valid answers)"

    def _normalize_answers(self, answers: str | list[str]) -> list[str]:
        """Normalize answers to always be a list."""
        if isinstance(answers, str):
            return [answers]
        return answers if answers else []

    def _build_catalogue(self) -> dict[str, dict[str, list[Lever]]]:
        """Build the full catalogue of 90 perturbation levers."""
        return {
            "P-Ambiguity": {
                "LOW": [
                    Lever(
                        "Dominant-sense Polysemy",
                        "Introduce a polysemous word where one sense is overwhelmingly favored by context.",
                        'Q-pert: "What is the current *rate* for student loans?" C: "The *interest rate* is **4%**." A: "4%."',
                    ),
                    Lever(
                        "Salient-antecedent Pronoun",
                        "Use a pronoun where discourse salience makes one referent the clear choice.",
                        'Q-pert: "Why did *she* cancel the flight?" C: "Maria phoned Lisa. *She* cancelled the flight due to a **storm warning**." A: "a storm warning."',
                    ),
                    Lever(
                        "Scoped Negation with Numeric Disambiguation",
                        "Use 'not all' but include explicit numbers that force the correct interpretation.",
                        'Q-pert: "Is it true that *not all* students passed?" C: "Out of twenty students, **three failed**." A: "Yes."',
                    ),
                    Lever(
                        "Resolvable Appositive Ambiguity",
                        "Introduce a secondary description that is clearly resolved by the context.",
                        "Q-pert: \"Which planet, the 'Evening Star,' has the longest day?\" C: \"While Mars is visible, only **Venus** is the 'Evening Star'. Venus has the longest solar day.\" A: \"Venus.\"",
                    ),
                    Lever(
                        "Late-resolving Garden Path",
                        "Use a misleading phrase that is clarified by the end of the query.",
                        'Q-pert: "Pilot helps dog bite victim -- who helped?" C: "The **pilot** rendered first aid to the jogger." A: "The pilot."',
                    ),
                    Lever(
                        "Coercive Adjective Polysemy",
                        "Use an adjective that forces a specific sense of a polysemous noun.",
                        'Q-pert: "What is the weight of the *light*?" C: "The stage *light* fixture weighs **15 kg**." A: "15 kg."',
                    ),
                    Lever(
                        "Metonymy (Institution for Building)",
                        "Use a metonym where the query forces the institutional sense, which is supported by context.",
                        'Q-pert: "Who announced the new policy?" C: "**The Pentagon** announced a new policy today. Over 20,000 employees work in the building." A: "The Pentagon."',
                    ),
                    Lever(
                        "Floating-Quantifier Anchored",
                        "Use a floating quantifier like 'all' anchored by clear context.",
                        'Q-pert: "Did the students *all* score above 80?" C: "The students scored **85, 90, and 92**." A: "Yes."',
                    ),
                    Lever(
                        "Comparative Ellipsis Resolved",
                        "Use an elliptical comparison where the missing info is unambiguously recoverable.",
                        'Q-pert: "Who scored more, Sarah or Alex?" C: "Sarah scored 20 points. Alex scored 15. **Sarah** scored more." A: "Sarah."',
                    ),
                    Lever(
                        "Possessive Split Disambiguated",
                        "Use a potentially ambiguous possessive phrase clarified by context.",
                        "Q-pert: \"Who is Sarah's book editor?\" C: \"Sarah is an author. **John Smith** is the editor for her book.\" A: \"John Smith.\"",
                    ),
                ],
                "MEDIUM": [
                    Lever(
                        "Balanced Polysemy",
                        "Use a word with two equally plausible senses, with balanced contextual cues for each.",
                        'Q-pert: "What colour is the *seal*?" C: "A grey harbor *seal* swam by. The document had a red wax *seal*." A: "REFUSE_AMBIGUOUS_QUERY"',
                    ),
                    Lever(
                        "PP-Attachment Ambiguity",
                        "Embed a classic prepositional phrase attachment ambiguity where the PP could plausibly modify either the verb or the noun.",
                        'Q-pert: "What did the woman see?" C: "The woman saw the man with the binoculars." A: "REFUSE_AMBIGUOUS_QUERY"',
                    ),
                    Lever(
                        "Dual-eligible Pronoun",
                        "Introduce two antecedents of the same gender/number, then use a pronoun that could refer to either.",
                        'Q-pert: "Who entered the chamber?" C: "The senator phoned the governor before *she* entered the chamber." A: "REFUSE_AMBIGUOUS_QUERY"',
                    ),
                    Lever(
                        "Either/Or Coordination Scope",
                        "Use a coordinated noun phrase where a modifier could apply to one or both nouns.",
                        'Q-pert: "Which group was old?" C: "The document listed *old men and women* for priority seating." A: "REFUSE_AMBIGUOUS_QUERY"',
                    ),
                    Lever(
                        "Ambiguous Deictic Reference",
                        "Use 'this' or 'that' when two equally plausible referents have been recently introduced.",
                        'Q-pert: "How many pages does *this document* have?" C: "The proposal (20 pages) and the contract (30 pages) were submitted." A: "REFUSE_AMBIGUOUS_QUERY"',
                    ),
                    Lever(
                        "Sluiced-Wh Ambiguity",
                        "Use a 'wh-' question with an elided verb phrase (sluicing) with two plausible antecedents.",
                        "Q-pert: \"Who left the meeting early?\" C: \"Either John or Sarah left the meeting early, but the log doesn't say *who*.\" A: \"REFUSE_AMBIGUOUS_QUERY\"",
                    ),
                    Lever(
                        "Comparative Subdeletion",
                        "Use a comparative where the elided property is ambiguous.",
                        'Q-pert: "Is the new building taller?" C: "The new building is taller than the old bridge is." A: "REFUSE_AMBIGUOUS_QUERY"',
                    ),
                    Lever(
                        "Quantifier Phrase Split",
                        "Use a complex noun phrase with multiple quantifiers where the scope is unclear.",
                        'Q-pert: "How many guards are there?" C: "There are ten guards with a badge or a gun." A: "REFUSE_AMBIGUOUS_QUERY"',
                    ),
                    Lever(
                        "Multi-Sentence Coreference Chain",
                        "Create ambiguity with a pronoun whose potential antecedent in a prior sentence competes with a closer one.",
                        'Q-pert: "What was its color?" C: "The team received a new car. It was parked next to an old truck. **It** was bright red." A: "REFUSE_AMBIGUOUS_QUERY"',
                    ),
                ],
                "HIGH": [
                    Lever(
                        "Reduced Relative Clause Clash",
                        "Use a reduced relative clause that is syntactically ambiguous and requires metalinguistic knowledge to parse.",
                        'Q-pert: "Who manned the boats?" C: "The old man the boats." A: "REFUSE_AMBIGUOUS_QUERY"',
                    ),
                    Lever(
                        "Pure Homonymy Clash",
                        "Use a homonym with two separate topical frames, making the query impossible to resolve.",
                        "Q-pert: \"What is the *bat's* weight?\" C: \"A sentence describes a wooden baseball *bat*. Another describes a nocturnal flying *bat*.\" A: \"REFUSE_AMBIGUOUS_QUERY\"",
                    ),
                    Lever(
                        "Nested Garden-path Sentence",
                        "Embed a notoriously difficult garden-path sentence where the syntax is highly misleading and the context offers no clarification.",
                        'Q-pert: "Why did *the horse raced past the barn fell*?" C: "A horse fell near a barn after a race. A loose stone was found nearby." A: "REFUSE_AMBIGUOUS_QUERY"',
                    ),
                    Lever(
                        "Stacked Quantifier & Negation Tangle",
                        "Compose a query with multiple quantifiers and negation where the truth value flips depending on scope.",
                        'Q-pert: "Did *some official not inspect every department*?" C: "Each official missed one department, but every department was seen by at least one official." A: "REFUSE_AMBIGUOUS_QUERY"',
                    ),
                    Lever(
                        "Multi-pronoun Braid",
                        "Introduce two antecedents and use a chain of pronouns creating combinatorial ambiguity.",
                        'Q-pert: "Who was late?" C: "Anna told Beth that *she* should remind *her* that *she* was late." A: "REFUSE_AMBIGUOUS_QUERY"',
                    ),
                    Lever(
                        "Dangling VP Ellipsis",
                        "Use verb phrase ellipsis where the elided verb could have multiple, distinct antecedents.",
                        'Q-pert: "Who likes what?" C: "Bob likes his coffee black, and Alice, tea." A: "REFUSE_AMBIGUOUS_QUERY"',
                    ),
                    Lever(
                        "Center-Embedding Maze",
                        "Use multiple center-embedded clauses that make tracking subject-verb relationships nearly impossible.",
                        'Q-pert: "What did the dog do?" C: "The dog the cat the rat bit chased ran away." A: "REFUSE_AMBIGUOUS_QUERY"',
                    ),
                    Lever(
                        "Attachment Cascade",
                        "Use a long string of prepositional phrases where each could attach to multiple prior nouns.",
                        'Q-pert: "Where is the book?" C: "I saw the book on the shelf in the corner of the room with the red door." A: "REFUSE_AMBIGUOUS_QUERY"',
                    ),
                    Lever(
                        "Anaphoric 'One' Collision",
                        "Use the anaphoric 'one' when there are two equally plausible antecedent noun phrases it could refer to.",
                        'Q-pert: "Which one did he buy?" C: "He saw a red car and a blue car. He decided to buy one." A: "REFUSE_AMBIGUOUS_QUERY"',
                    ),
                    Lever(
                        "Ambiguous 'Respectively' Mapping",
                        "Provide lists of different lengths that are supposed to map 'respectively', making the mapping undecidable.",
                        'Q-pert: "What are the scores of Alice and Bob?" C: "Alice, Bob, and Carol scored 90 and 85 respectively." A: "REFUSE_AMBIGUOUS_QUERY"',
                    ),
                ],
            },
            "P-Contradiction": {
                "LOW": [
                    Lever(
                        "Minor Numeric Rounding",
                        "Introduce slightly different numbers for the same statistic within a rounding margin.",
                        "Q-pert: \"Is the lake over 90 km2?\" C: \"The area is approx. 100 km2. Official surveys list it as **98 km2**.\" A: \"Yes.\"",
                    ),
                    Lever(
                        "Peripheral Attribute Clash",
                        "Contradict an irrelevant detail, leaving the core fact stable.",
                        "Q-pert: \"What was the car's top speed?\" C: \"The navy blue sports car reached **150 mph**. Some models were dark green.\" A: \"150 mph.\"",
                    ),
                    Lever(
                        "Fringe vs. Official Source",
                        "Contradict a high-credibility source with a low-credibility one.",
                        'Q-pert: "What is the capital of France?" C: "Official records state **Paris**. A blog claims Lyon." A: "Paris."',
                    ),
                    Lever(
                        "Soft Antonym Modifiers",
                        "Use vague opposing adjectives but provide a precise numerical value.",
                        "Q-pert: \"What was the exact temperature?\" C: \"The water was a mild **20 C**. Some found it slightly cool.\" A: \"20 C.\"",
                    ),
                    Lever(
                        "Nickname vs Legal Name",
                        "Use a nickname and a legal name in a contradictory way, but context clarifies the identity.",
                        'Q-pert: "What year was Elizabeth born?" C: "Records show **Elizabeth** was born in 1990. However, her passport name is listed as Liz." A: "1990."',
                    ),
                    Lever(
                        "Approx vs Exact",
                        "Provide an approximate and a precise value that are close, where the query is answerable by both.",
                        'Q-pert: "Is the distance over 100 miles?" C: "The distance is approximately 120 miles; to be exact, it is **121 miles**." A: "Yes."',
                    ),
                    Lever(
                        "Contradiction in Speaker's Title",
                        "State a fact from a source whose title or role is contradicted elsewhere, but the fact itself is not.",
                        "Q-pert: \"What did the CEO announce?\" C: \"The CEO, Jane Smith, announced **a new hiring freeze**. Her official title is Chief Technology Officer.\" A: \"a new hiring freeze.\"",
                    ),
                    Lever(
                        "Contradiction via Downstream Effect",
                        "State two facts whose downstream implications conflict, but the facts themselves do not directly.",
                        'Q-pert: "Is the device fully charged?" C: "The battery status screen shows **100%**. The device has been running a high-power simulation for the past 12 hours while unplugged." A: "100%."',
                    ),
                ],
                "MEDIUM": [
                    Lever(
                        "Modal Dilution",
                        "State a possibility and a certainty that are contradictory, forcing a refusal.",
                        "Q-pert: \"Will the temperature exceed 28 C?\" C: \"The forecast says the temperature may reach **30 C**. A separate weather alert states that the temperature will **not exceed 27 C** today.\" A: \"REFUSE_CONTRADICTORY_CONTEXT\"",
                    ),
                    Lever(
                        "Contradiction in Reported Speech",
                        "Have a source report two different versions of the same event or statement.",
                        "Q-pert: \"What color was the car?\" C: \"The witness initially told police, 'The car was blue.' However, her signed affidavit states, 'The car was green.'\" A: \"REFUSE_CONTRADICTORY_CONTEXT\"",
                    ),
                    Lever(
                        "Source-Timing Drift",
                        "Provide conflicting information from two sources with different timestamps, forcing a choice based on recency.",
                        'Q-pert: "What is the recommended treatment for the condition?" C: "A 2010 clinical study found that **aspirin** was effective. The 2025 international treatment guidelines recommend **immunotherapy**." A: "REFUSE_CONTRADICTORY_CONTEXT"',
                    ),
                    Lever(
                        "Dual-authoritative Dates",
                        "Quote two credible sources with conflicting dates.",
                        'Q-pert: "In what year was the treaty signed?" C: "An archive states **1918**. A history book claims **1919**." A: "REFUSE_CONTRADICTORY_CONTEXT"',
                    ),
                    Lever(
                        "Direct Polarity Reversal on Safety",
                        "Provide two sentences with opposite polarity on a critical property.",
                        'Q-pert: "Is the toy safe for children under 3?" C: "The product is **safe for toddlers**. The manual states it is **not safe for children under 3**." A: "REFUSE_CONTRADICTORY_CONTEXT"',
                    ),
                    Lever(
                        "Agent/Patient Role Reversal",
                        "Present two sentences that swap the agent and patient roles.",
                        'Q-pert: "Who did Alice hire?" C: "**Alice hired Bob**. A later entry states **Bob hired Alice**." A: "REFUSE_CONTRADICTORY_CONTEXT"',
                    ),
                    Lever(
                        "Conflicting Numerical Units",
                        "Provide two values for the same measurement that conflict when units are converted.",
                        'Q-pert: "What is the length of the board?" C: "The board measures **1 meter** in length. A second measurement recorded its length as **40 inches**." A: "REFUSE_CONTRADICTORY_CONTEXT"',
                    ),
                    Lever(
                        "Spatial Location Clash",
                        "State that an entity's primary office is in two mutually exclusive locations.",
                        "Q-pert: \"Where is the headquarters located?\" C: \"The company's primary headquarters is in **London**. The official charter states the primary headquarters is in **New York**.\" A: \"REFUSE_CONTRADICTORY_CONTEXT\"",
                    ),
                    Lever(
                        "Temporal Sequence Reversal",
                        "State that 'A happened before B' and 'B happened before A'.",
                        'Q-pert: "Which came first?" C: "The signing preceded the announcement. The timeline indicates the announcement was made before the signing." A: "REFUSE_CONTRADICTORY_CONTEXT"',
                    ),
                    Lever(
                        "Attributive vs. Predicative Conflict",
                        "Use a modifier that contradicts the predicate.",
                        'Q-pert: "Can the bird fly?" C: "The **flightless** Phorusrhacos was a bird known for **soaring** between cliffs." A: "REFUSE_CONTRADICTORY_CONTEXT"',
                    ),
                ],
                "HIGH": [
                    Lever(
                        "Order-of-Magnitude Numeric Clash",
                        "State two values for the same metric that differ by an order of magnitude.",
                        "Q-pert: \"What was the company's revenue?\" C: \"The company reported revenue of **$2 billion**. A separate filing shows revenue of **$20 million**.\" A: \"REFUSE_CONTRADICTORY_CONTEXT\"",
                    ),
                    Lever(
                        "Must vs. Cannot Rule",
                        "Impose two conflicting deontic modal rules (mandate and forbid).",
                        'Q-pert: "How many records?" C: "The file **must** contain a record. The file **cannot** contain any records." A: "REFUSE_CONTRADICTORY_CONTEXT"',
                    ),
                    Lever(
                        "Self-negating Definition",
                        "Provide a definition for a concept that is internally, logically contradictory.",
                        'Q-pert: "What is a plonk?" C: "A plonk is **both a perfect square and a perfect circle**." A: "REFUSE_CONTRADICTORY_CONTEXT"',
                    ),
                    Lever(
                        "Temporal Paradox",
                        "Describe an event that violates linear time.",
                        'Q-pert: "When did it start?" C: "It began at 4 PM and **ended two hours before it started**." A: "REFUSE_CONTRADICTORY_CONTEXT"',
                    ),
                    Lever(
                        "Life/Death Status Inconsistency",
                        "State a person is deceased and also describe them performing actions as if alive.",
                        'Q-pert: "Is Dr. Lee alive?" C: "Dr. Lee, who **died in 2019**, gave a lecture **in 2024**." A: "REFUSE_CONTRADICTORY_CONTEXT"',
                    ),
                    Lever(
                        "Exclusive-Or Violation",
                        "State that 'A or B but not both' occurred, and then state that 'both A and B' occurred.",
                        'Q-pert: "Did event A occur?" C: "**Either A or B occurred**, not both. Analysis confirmed **both A and B occurred**." A: "REFUSE_CONTRADICTORY_CONTEXT"',
                    ),
                    Lever(
                        "Contradictory Universals",
                        "State 'All X are Y' and 'Some X are not Y'.",
                        'Q-pert: "Are all swans white?" C: "**All swans are white**. The report includes a photo of a **black swan**." A: "REFUSE_CONTRADICTORY_CONTEXT"',
                    ),
                    Lever(
                        "Binary State Flip",
                        "Describe an object in two mutually exclusive binary states simultaneously.",
                        'Q-pert: "Is the switch on?" C: "The switch is **simultaneously on and off**." A: "REFUSE_CONTRADICTORY_CONTEXT"',
                    ),
                    Lever(
                        "Mutual Exclusion Claim",
                        "State that an event never happens and also that it always happens.",
                        'Q-pert: "Does the machine halt?" C: "The machine **always halts**. Due to a flaw, it **never halts**." A: "REFUSE_CONTRADICTORY_CONTEXT"',
                    ),
                    Lever(
                        "Logical Biconditional Break",
                        "State 'P iff Q', and then an instance of 'P and not Q'.",
                        'Q-pert: "Is the alarm active?" C: "The alarm is active if and only if the light is green. Currently, the alarm is active, but the light is red." A: "REFUSE_CONTRADICTORY_CONTEXT"',
                    ),
                ],
            },
            "P-MissingInfo": {
                "LOW": [
                    Lever(
                        "Delete Color Adjective",
                        "Remove a non-essential color descriptor.",
                        "Q-pert: \"What was the car's top speed?\" C: \"The convertible reached **150 mph**.\" A: \"150 mph.\"",
                    ),
                    Lever(
                        "Remove City Phrase",
                        "Strip a city name but leave the country.",
                        'Q-pert: "Which country?" C: "The conference was in **France**." A: "France."',
                    ),
                    Lever(
                        "Drop Example Sentence",
                        "Erase an illustrative example, but leave the main definition.",
                        'Q-pert: "What is the protocol?" C: "The system uses **Protocol X**." A: "Protocol X."',
                    ),
                    Lever(
                        "Strip Non-queried Statistic",
                        "Remove a secondary statistic, leaving the primary one.",
                        'Q-pert: "What was the revenue?" C: "Total revenue was **$50 million**." A: "$50 million."',
                    ),
                    Lever(
                        "Trim Month but Keep Year",
                        "Delete the month from a date but preserve the year.",
                        'Q-pert: "Which year?" C: "The law passed in **2022**." A: "2022."',
                    ),
                    Lever(
                        "Secondary Actor Cut",
                        "Remove a mention of a secondary character.",
                        'Q-pert: "What did Dr. Evans discover?" C: "Dr. Evans discovered **a new frog**." A: "a new frog."',
                    ),
                    Lever(
                        "Parenthetical Excise",
                        "Delete an irrelevant parenthetical clarification.",
                        'Q-pert: "What is the capital of Australia?" C: "**Canberra** is the capital." A: "Canberra."',
                    ),
                    Lever(
                        "Intro Clause Removal",
                        "Delete a non-essential introductory clause.",
                        'Q-pert: "What is the melting point of iron?" C: "Iron has a melting point of **1,538 C**." A: "1,538 C."',
                    ),
                    Lever(
                        "Redundant Synonym Skipped",
                        "If context uses two synonyms, remove one.",
                        'Q-pert: "What was the vote result?" C: "The proposal was **approved**." A: "approved."',
                    ),
                ],
                "MEDIUM": [
                    Lever(
                        "Missing Temporal Adjunct",
                        "Delete a compulsory temporal adjunct like 'yesterday' when the query asks 'When?'.",
                        'Q-pert: "When did she leave for the airport?" C: "She left for the airport after packing her bags." A: "REFUSE_INFO_MISSING_IN_CONTEXT"',
                    ),
                    Lever(
                        "Adjunct/Complement Blur",
                        "Use a phrase that is syntactically parsable in two ways, but where the context is missing information to disambiguate.",
                        'Q-pert: "Was the chicken ready to eat?" C: "The chicken was ready to eat." A: "REFUSE_INFO_MISSING_IN_CONTEXT"',
                    ),
                    Lever(
                        "Answer Clause Excised",
                        "Remove the single clause containing the explicit answer.",
                        'Q-pert: "Who did Apex acquire?" C: "In 2022, Apex acquired a fintech rival." A: "REFUSE_INFO_MISSING_IN_CONTEXT"',
                    ),
                    Lever(
                        "Price Redacted",
                        "Replace a monetary value with 'an undisclosed amount'.",
                        'Q-pert: "How much did it cost?" C: "The company bought it for **an undisclosed sum**." A: "REFUSE_INFO_MISSING_IN_CONTEXT"',
                    ),
                    Lever(
                        "Placeholder Entity Substitution",
                        "Swap a proper noun for a generic placeholder like 'a major firm'.",
                        'Q-pert: "Which university?" C: "They partnered with **a leading research university**." A: "REFUSE_INFO_MISSING_IN_CONTEXT"',
                    ),
                    Lever(
                        "Table Gap",
                        "Omit the single row or cell in a table that contains the answer.",
                        'Q-pert: "Revenue in 2021?" C: "Table shows 2020 and 2022, but the **2021 row is missing**." A: "REFUSE_INFO_MISSING_IN_CONTEXT"',
                    ),
                    Lever(
                        "Temporal Mismatch (Outdated Info)",
                        "Provide information that was valid in the past but is outdated for a query asking about the present.",
                        'Q-pert: "Who is the *current* CEO of TechCorp?" C: "John Smith served as CEO from 2015 to 2019. Jane Doe succeeded him and stepped down in 2023." A: "REFUSE_INFO_MISSING_IN_CONTEXT"',
                    ),
                    Lever(
                        "Abbreviated Reference Omission",
                        "Use an acronym without defining it.",
                        'Q-pert: "What does UNFCCC stand for?" C: "The report was sent to the UNFCCC." A: "REFUSE_INFO_MISSING_IN_CONTEXT"',
                    ),
                    Lever(
                        "Placeholder Citation",
                        "Replace a key fact with '[citation needed]'.",
                        'Q-pert: "What percentage were affected?" C: "A significant number were affected **[citation needed]**." A: "REFUSE_INFO_MISSING_IN_CONTEXT"',
                    ),
                    Lever(
                        "Pronoun w/o Antecedent",
                        "Start a context with a pronoun whose antecedent is never introduced.",
                        'Q-pert: "What is his job?" C: "*He* started a new job last week at a major tech firm in the city." A: "REFUSE_INFO_MISSING_IN_CONTEXT"',
                    ),
                    Lever(
                        "Average From Individual",
                        "Ask for a statistical average when only an incomplete set of individual data points is provided.",
                        "Q-pert: \"What is the average height of the three players?\" C: \"The team has three players. Alice is 180cm tall. Bob is 190cm tall. Carol's height was not recorded.\" A: \"REFUSE_INFO_MISSING_IN_CONTEXT\"",
                    ),
                ],
                "HIGH": [
                    Lever(
                        "Patient Erasure",
                        "Describe an action but completely omit the patient/theme.",
                        'Q-pert: "Who did Apex acquire?" C: "Apex announced a strategic acquisition in Q4." A: "REFUSE_INFO_MISSING_IN_CONTEXT"',
                    ),
                    Lever(
                        "Agent Deletion",
                        "Narrate an event but leave the agent entirely unmentioned.",
                        'Q-pert: "Who stole the data?" C: "Valuable data was stolen overnight." A: "REFUSE_INFO_MISSING_IN_CONTEXT"',
                    ),
                    Lever(
                        "Verdict/Outcome Removal",
                        "Detail a trial or experiment but omit the final result.",
                        'Q-pert: "What was the ruling?" C: "The jury deliberated for two days." A: "REFUSE_INFO_MISSING_IN_CONTEXT"',
                    ),
                    Lever(
                        "Cause Omission",
                        "List the effects of an event but remove any mention of its cause.",
                        'Q-pert: "Why did the system crash?" C: "The crash led to hours of downtime." A: "REFUSE_INFO_MISSING_IN_CONTEXT"',
                    ),
                    Lever(
                        "Answer Token Purge",
                        "Systematically remove every token that constitutes the answer string.",
                        'Q-pert: "What company was acquired?" C: "Apex acquired [a company] to expand its market share." A: "REFUSE_INFO_MISSING_IN_CONTEXT"',
                    ),
                    Lever(
                        "Method Section Removed",
                        "Describe the results of an experiment but provide no info on the method.",
                        'Q-pert: "What method was used?" C: "The data shows a clear correlation." A: "REFUSE_INFO_MISSING_IN_CONTEXT"',
                    ),
                    Lever(
                        "Zero-Row Table",
                        "Provide a table with only a header row.",
                        'Q-pert: "Population in 2020?" C: "| Year | Population |" A: "REFUSE_INFO_MISSING_IN_CONTEXT"',
                    ),
                    Lever(
                        "All Names Redacted",
                        "Describe an interaction but redact all names and roles.",
                        'Q-pert: "Who reported to whom?" C: "[PERSON A] reported to [PERSON B]." A: "REFUSE_INFO_MISSING_IN_CONTEXT"',
                    ),
                    Lever(
                        "Entirely Irrelevant Context",
                        "Provide a context on a completely different topic from the query.",
                        'Q-pert: "Exports of Brazil?" C: "Italian Renaissance art is fascinating." A: "REFUSE_INFO_MISSING_IN_CONTEXT"',
                    ),
                    Lever(
                        "Audio-only Reference",
                        "State that the answer is contained in an unprovided audio file.",
                        'Q-pert: "What was the testimony?" C: "The full testimony is on audio track 3." A: "REFUSE_INFO_MISSING_IN_CONTEXT"',
                    ),
                ],
            },
            "P-FalsePremise": {
                "LOW": [
                    Lever(
                        "Imaginary Add-on",
                        "Add a presupposition about a minor, non-existent component.",
                        "Q-pert: \"Findings from the Mars rover's *new lidar sensor*?\" C: \"The rover found **traces of water** with its drill.\" A: \"traces of water.\"",
                    ),
                    Lever(
                        "Assumed Second Edition",
                        "Frame the query about a later edition, but ask for a detail that is the same.",
                        'Q-pert: "Who wrote the *third edition*?" C: "The novel was written by **Jane Smith**." A: "Jane Smith."',
                    ),
                    Lever(
                        "Over-specific Year",
                        "Preface the query with an unsupported time-frame but ask for a current, correct statistic.",
                        'Q-pert: "*Since 2010*, what is the population?" C: "The current population is **1.2 million**." A: "1.2 million."',
                    ),
                    Lever(
                        "False Corporate Parent Claim",
                        "Incorrectly state a company is owned by another, but ask for a true detail.",
                        "Q-pert: \"CEO of InnoTech, *the Google subsidiary*?\" C: \"InnoTech's CEO is **John Doe**.\" A: \"John Doe.\"",
                    ),
                    Lever(
                        "Award Presupposition",
                        "Presuppose an unmentioned award, but ask for a different, verifiable fact.",
                        'Q-pert: "*After winning Best Picture*, who directed?" C: "The film was directed by **Maria Garcia**." A: "Maria Garcia."',
                    ),
                    Lever(
                        "Misstated Capital City",
                        "Presuppose a false fact that links two unrelated entities, but ask for a fact about one that is present.",
                        'Q-pert: "What is the population of the Big Apple, the capital of Brazil?" C: "New York City, also known as the Big Apple, has a population of 8.5 million. The capital of Brazil is Brasilia." A: "8.5 million."',
                    ),
                    Lever(
                        "Imagined Product Variant",
                        "Ask about a non-existent product variant, but for a common feature.",
                        'Q-pert: "Battery in the iPhone 15 *Mini*?" C: "All iPhone 15 models use **lithium-ion**." A: "lithium-ion."',
                    ),
                    Lever(
                        "Non-existent Committee",
                        "Invent a committee name but ask about a person who holds the role across the organization.",
                        'Q-pert: "Chair of the *Digital Ethics Subcommittee*?" C: "**Dr. Evans** heads the *Tech Committee*." A: "Dr. Evans."',
                    ),
                    Lever(
                        "Mis-dated Discovery",
                        "Presuppose an incorrect date in the query, but ask for an unrelated fact that remains answerable.",
                        'Q-pert: "Who discovered Penicillin in 1938?" C: "**Alexander Fleming** discovered Penicillin in 1928." A: "Alexander Fleming."',
                    ),
                    Lever(
                        "Contradiction in Irrelevant Detail",
                        "Include a factual error in an ancillary clause that contradicts world knowledge, while leaving the direct answer intact.",
                        'Q-pert: "What is the capital of France?" C: "The capital of France, a large country in Asia, is **Paris**." A: "Paris."',
                    ),
                ],
                "MEDIUM": [
                    Lever(
                        "First vs. Second Mission Clash",
                        "Ask for a comparison to a second event when context states it was the first.",
                        "Q-pert: \"How did the *second* Mars rover compare?\" C: \"This was humanity's *first* rover on Mars.\" A: \"REFUSE_FALSE_PREMISE_IN_QUERY\"",
                    ),
                    Lever(
                        "Stopped vs. Never Started",
                        "Use 'stop' when the context states the activity never occurred.",
                        'Q-pert: "Why did John *stop* attending?" C: "John has *never attended* meetings." A: "REFUSE_FALSE_PREMISE_IN_QUERY"',
                    ),
                    Lever(
                        "Abdication Falsehood",
                        "Ask about the motives for an abdication that never happened.",
                        "Q-pert: \"Reason for the king's abdication?\" C: \"The king is still reigning.\" A: \"REFUSE_FALSE_PREMISE_IN_QUERY\"",
                    ),
                    Lever(
                        "Burial for a Living Person",
                        "Ask for the burial location of a person who is still alive.",
                        'Q-pert: "Where was Jane Smith buried?" C: "Jane Smith, who is **still alive**, lives in London." A: "REFUSE_FALSE_PREMISE_IN_QUERY"',
                    ),
                    Lever(
                        "Phantom Product Line",
                        "Ask for details about a product model that was never released.",
                        'Q-pert: "How many Model Z cars sold?" C: "The planned **Model Z was never released**." A: "REFUSE_FALSE_PREMISE_IN_QUERY"',
                    ),
                    Lever(
                        "Misclassified Biology",
                        "Presuppose an incorrect biological classification.",
                        'Q-pert: "How many gills does the blue whale, a fish, have?" C: "The blue whale is a mammal." A: "REFUSE_FALSE_PREMISE_IN_QUERY"',
                    ),
                    Lever(
                        "Reversed Causality",
                        "Ask 'Why did A cause B?' when context states B caused A.",
                        'Q-pert: "Why did the boom cause the new law?" C: "The new law caused the boom." A: "REFUSE_FALSE_PREMISE_IN_QUERY"',
                    ),
                    Lever(
                        "Fabricated Historic Event",
                        "Ask about the details of a historic event that never happened.",
                        'Q-pert: "Who signed the Treaty of Martian Independence?" C: "Humans have not yet colonized Mars." A: "REFUSE_FALSE_PREMISE_IN_QUERY"',
                    ),
                    Lever(
                        "Impossible Geography",
                        "Presuppose a non-existent geographical feature.",
                        'Q-pert: "Length of the US-France land border?" C: "They do not share a land border." A: "REFUSE_FALSE_PREMISE_IN_QUERY"',
                    ),
                    Lever(
                        "Factive Verb Contradiction",
                        "Use a factive verb like 'regret' where the context shows the outcome was positive.",
                        'Q-pert: "Why did the team *regret* their win?" C: "The team was ecstatic about their winning strategy." A: "REFUSE_FALSE_PREMISE_IN_QUERY"',
                    ),
                ],
                "HIGH": [
                    Lever(
                        "Binding Principle Violation",
                        "Frame a query that presupposes an ungrammatical sentence is well-formed.",
                        'Q-pert: "Why did *he* say that *John* is late?" C: "Peter said that John is late." A: "REFUSE_FALSE_PREMISE_IN_QUERY"',
                    ),
                    Lever(
                        "Inanimate Agent Verb",
                        "Attribute an action requiring consciousness to an inanimate object.",
                        'Q-pert: "Why did the *theory* persuade the voters?" C: "The theory is complex." A: "REFUSE_FALSE_PREMISE_IN_QUERY"',
                    ),
                    Lever(
                        "Impossible Property",
                        "Ask for a property of a logically impossible object.",
                        'Q-pert: "What is the weight of a *square circle*?" C: "Shapes can be squares or circles." A: "REFUSE_FALSE_PREMISE_IN_QUERY"',
                    ),
                    Lever(
                        "Category Swap",
                        "Ask for a property from one semantic domain about a concept from an incompatible domain.",
                        'Q-pert: "What color is *sadness*?" C: "Sadness is an emotion." A: "REFUSE_FALSE_PREMISE_IN_QUERY"',
                    ),
                    Lever(
                        "Self-contradictory Entity",
                        "Presuppose an entity defined by contradictory properties.",
                        'Q-pert: "Why is the *even prime number greater than two* special?" C: "The only even prime is 2." A: "REFUSE_FALSE_PREMISE_IN_QUERY"',
                    ),
                    Lever(
                        "Married Bachelor Query",
                        "Ask about the properties of a 'married bachelor', a logically contradictory entity.",
                        "Q-pert: \"What is the name of the married bachelor's wife?\" C: \"A bachelor is an unmarried man.\" A: \"REFUSE_FALSE_PREMISE_IN_QUERY\"",
                    ),
                    Lever(
                        "Infinite Set with Finite Size",
                        "Ask a question that presupposes an infinite set has a specific finite size.",
                        'Q-pert: "What is the 8th integer if the set of integers has only 7 elements?" C: "The set of integers is infinite." A: "REFUSE_FALSE_PREMISE_IN_QUERY"',
                    ),
                    Lever(
                        "Affirming the Consequent",
                        "Frame a query that presupposes a classic logical fallacy is true.",
                        'Q-pert: "Given that the streets are wet, what kind of rainstorm must have caused it?" C: "The streets are wet. The sprinkler system was on all night." A: "REFUSE_FALSE_PREMISE_IN_QUERY"',
                    ),
                    Lever(
                        "Composition Fallacy",
                        "Ask a question that assumes the whole must have the properties of its parts.",
                        'Q-pert: "Since every brick in the wall is light, what is the light weight of the entire wall?" C: "The wall is made of thousands of light bricks." A: "REFUSE_FALSE_PREMISE_IN_QUERY"',
                    ),
                    Lever(
                        "Division Fallacy",
                        "Ask a question that assumes the parts must have the properties of the whole.",
                        "Q-pert: \"Since the team is successful, what was Alice's successful contribution?\" C: \"The team was successful overall, though Alice's project within the team failed.\" A: \"REFUSE_FALSE_PREMISE_IN_QUERY\"",
                    ),
                ],
            },
            "P-GranularityMismatch": {
                "LOW": [
                    Lever(
                        "Superclass Trait Made Explicit",
                        "Ask for a trait of a subclass when the context defines it via its superclass.",
                        'Q-pert: "Are dogs warm-blooded?" C: "Dogs are mammals, and all mammals are **warm-blooded**." A: "Yes."',
                    ),
                    Lever(
                        "Explicit Total Line",
                        "Ask for an aggregate when the context explicitly states the sum.",
                        'Q-pert: "Total revenue?" C: "Q1 was $5M, Q2 was $5M, for a total of **$10M**." A: "$10M."',
                    ),
                    Lever(
                        "Genus-to-Species Mismatch",
                        "Ask for a property of a species when context provides it for the genus and the link is obvious.",
                        'Q-pert: "What is the average weight of the timber wolf (Canis lupus)?" C: "The genus Canis, which includes wolves, has an average weight of 30-80 kg." A: "30-80 kg."',
                    ),
                    Lever(
                        "Inline Unit Conversion",
                        "Provide a measurement in two units and ask for one of them.",
                        'Q-pert: "Weight in pounds?" C: "Weighs 2 kg (approx. **4.4 lbs**)." A: "4.4 lbs."',
                    ),
                    Lever(
                        "Minute Precision Present",
                        "Ask for a time to the minute when the context provides it.",
                        'Q-pert: "Launch minute?" C: "Launch at **14:37**." A: "14:37."',
                    ),
                    Lever(
                        "Notable equals Listed Specs",
                        "Ask for 'notable' features when context provides a list of objective specs.",
                        'Q-pert: "What are the notable features?" C: "The car has a V8 engine, a sunroof, and four wheels." A: "V8 engine and sunroof."',
                    ),
                    Lever(
                        "Named-Individual Ask",
                        "Ask about a property of a named individual when context lists it.",
                        "Q-pert: \"Alice's score?\" C: \"Scores: Alice (**95**), Bob (80).\" A: \"95.\"",
                    ),
                    Lever(
                        "Per-year Figure Already Given",
                        "Ask for a per-year statistic when the context provides it.",
                        'Q-pert: "Growth rate in 2023?" C: "Growth was **5%** in 2023." A: "5%."',
                    ),
                    Lever(
                        "Exact Degree Value",
                        "Ask for a value to a precision level provided in the context.",
                        'Q-pert: "Angle to one decimal?" C: "Angle was **45.2** degrees." A: "45.2."',
                    ),
                    Lever(
                        "Component of a Named Part",
                        "Ask for a component of a part when the context lists it.",
                        'Q-pert: "Chipset on motherboard?" C: "Mainboard has **Z9 chipset**." A: "Z9 chipset."',
                    ),
                ],
                "MEDIUM": [
                    Lever(
                        "Nation from City Data",
                        "List city data and ask for the national policy.",
                        'Q-pert: "What is the national speed limit?" C: "The speed limit is 50 km/h in City A and 60 km/h in City B. No single nationwide speed limit is defined." A: "REFUSE_GRANULARITY_MISMATCH"',
                    ),
                    Lever(
                        "Long-range Trend Extrapolation",
                        "Provide short-term data and ask for a long-term trend.",
                        "Q-pert: \"Describe the company's growth over the past decade.\" C: \"Sales grew from $10M in 2021 to $12M in 2022.\" A: \"REFUSE_GRANULARITY_MISMATCH\"",
                    ),
                    Lever(
                        "Part-of-part Detail",
                        "Mention a component but not its sub-components, then ask about a sub-component.",
                        'Q-pert: "How many ALUs are in each CPU core?" C: "The laptop uses the powerful Z9 chipset." A: "REFUSE_GRANULARITY_MISMATCH"',
                    ),
                    Lever(
                        "Cross-level Policy Synthesis",
                        "Provide low-level rules and ask for the high-level policy.",
                        'Q-pert: "What is the company-wide dress code policy?" C: "The engineering team requires safety shoes. The sales team must wear suits." A: "REFUSE_GRANULARITY_MISMATCH"',
                    ),
                    Lever(
                        "Subtype Policy Inquiry",
                        "Provide a general policy and ask about a specific subtype not mentioned.",
                        'Q-pert: "Return policy for clearance items?" C: "Returns within 30 days for most items." A: "REFUSE_GRANULARITY_MISMATCH"',
                    ),
                    Lever(
                        "Decade Trend from Daily Data",
                        "Provide daily data and ask for a qualitative decade-long trend.",
                        "Q-pert: \"Describe the climate trend last decade?\" C: \"Yesterday's temperature was 15 C.\" A: \"REFUSE_GRANULARITY_MISMATCH\"",
                    ),
                    Lever(
                        "Single-Instance to Universal Rule",
                        "Provide one example and ask if it constitutes a universal rule.",
                        'Q-pert: "Do all swans migrate in winter?" C: "A single swan was observed migrating south last winter." A: "REFUSE_GRANULARITY_MISMATCH"',
                    ),
                    Lever(
                        "Micro-to-Macro Proportion",
                        "Provide a detail about a component and ask for its proportion of the whole, without providing the total.",
                        "Q-pert: \"What percentage of the car's total weight is the engine?\" C: \"The engine weighs 150 kg. The car's total weight is not specified.\" A: \"REFUSE_GRANULARITY_MISMATCH\"",
                    ),
                    Lever(
                        "Median-With-Missing Member",
                        "Provide some members of a set and ask for a statistic that depends on the full set.",
                        "Q-pert: \"Median salary of 3 hires?\" C: \"One is paid $50k, another $60k. The third salary is confidential.\" A: \"REFUSE_GRANULARITY_MISMATCH\"",
                    ),
                    Lever(
                        "Individual Actions vs. Collective Goal",
                        "Describe individual actions but ask if a collective goal was met.",
                        'Q-pert: "Did the team successfully launch the product?" C: "Alice finished the user interface. Bob completed the backend database." A: "REFUSE_GRANULARITY_MISMATCH"',
                    ),
                ],
                "HIGH": [
                    Lever(
                        "Molecular vs. Organism",
                        "Ask for molecular-level info when context is macroscopic.",
                        'Q-pert: "What is the amino acid sequence of actin in a lion?" C: "The lion is a large mammal." A: "REFUSE_GRANULARITY_MISMATCH"',
                    ),
                    Lever(
                        "Global Average from Street Data",
                        "Provide a single local data point and ask for the global average.",
                        "Q-pert: \"What is the global average CO2 concentration?\" C: \"The CO2 level on this street was 420 ppm.\" A: \"REFUSE_GRANULARITY_MISMATCH\"",
                    ),
                    Lever(
                        "Century vs. Millisecond",
                        "Give a micro-level temporal detail and ask about macro-level stability.",
                        'Q-pert: "How stable is the system over a century?" C: "The system has a network latency of 5 microseconds." A: "REFUSE_GRANULARITY_MISMATCH"',
                    ),
                    Lever(
                        "System Bus from RAM Spec",
                        "Provide a spec for one component and ask for the architecture of the whole system.",
                        "Q-pert: \"Describe the mainboard's bus topology.\" C: \"The computer uses high-speed DDR5 RAM.\" A: \"REFUSE_GRANULARITY_MISMATCH\"",
                    ),
                    Lever(
                        "Abstract Moral Worth from Code",
                        "Give a technical description and ask for a high-level abstract judgment.",
                        'Q-pert: "Is this sorting algorithm morally good?" C: "def quicksort(arr):..." A: "REFUSE_GRANULARITY_MISMATCH"',
                    ),
                    Lever(
                        "Continental Climate from Room Temp",
                        "Provide a hyper-local measurement and ask for a continental-scale description.",
                        'Q-pert: "Describe the climate of North America." C: "Room temp is 22 C." A: "REFUSE_GRANULARITY_MISMATCH"',
                    ),
                    Lever(
                        "World GDP from Single Firm Revenue",
                        "Provide a single company's revenue and ask for world GDP.",
                        'Q-pert: "What is the Gross Domestic Product of the world?" C: "Apple Inc. reported revenues of over $380B last year." A: "REFUSE_GRANULARITY_MISMATCH"',
                    ),
                    Lever(
                        "Genome Detail from Species Count",
                        "Ask for a genetic detail when context is an ecological count.",
                        'Q-pert: "Base pairs in wolf genome?" C: "5,000 wolves in the US." A: "REFUSE_GRANULARITY_MISMATCH"',
                    ),
                    Lever(
                        "City-block Image to Global Land-use",
                        "Ask for a global summary based on a local image.",
                        'Q-pert: "Global urban land distribution?" C: "This image shows 80% building coverage." A: "REFUSE_GRANULARITY_MISMATCH"',
                    ),
                    Lever(
                        "Microsecond Trace to Decadal Reliability",
                        "Ask for a decadal forecast based on a microsecond trace.",
                        'Q-pert: "Satellite lifespan?" C: "CPU spiked for 2ns during boot." A: "REFUSE_GRANULARITY_MISMATCH"',
                    ),
                ],
            },
            "P-EpistemicMismatch": {
                "LOW": [
                    Lever(
                        "Undesirable Effect Obvious",
                        "Ask if a negative outcome is undesirable, a common-sense inference.",
                        'Q-pert: "Is the memory leak undesirable?" C: "It causes crashes." A: "Yes."',
                    ),
                    Lever(
                        "Weak Possibility vs. High Likelihood",
                        "Ask if something 'could' happen when context says it is 'very likely'.",
                        'Q-pert: "Could the project succeed?" C: "The project is described as being **very likely to succeed**." A: "Yes."',
                    ),
                    Lever(
                        "Heavier is Factual",
                        "Ask a comparative when context gives numerical weights.",
                        'Q-pert: "Is A heavier than B?" C: "A weighs **2 kg**. B weighs **3 kg**." A: "No."',
                    ),
                    Lever(
                        "Commonness from Percentage",
                        "Ask if a feature is 'common' when context gives a high percentage.",
                        'Q-pert: "Is this common?" C: "Has **80% adoption**." A: "Yes."',
                    ),
                    Lever(
                        "Binary Desirability",
                        "Ask a good/bad question with a clear factual outcome.",
                        'Q-pert: "Is the data breach good?" C: "It exposed personal info." A: "No."',
                    ),
                    Lever(
                        "Obvious Benefit",
                        "Ask about the benefit of something with a clearly positive outcome.",
                        'Q-pert: "Is the vaccine beneficial?" C: "Provides 100% protection." A: "Yes."',
                    ),
                    Lever(
                        "Minor vs Major (Definitional)",
                        "Ask if an event was 'major' when the context labels it as such.",
                        'Q-pert: "Was it a major event?" C: "Classified as a **major** event." A: "Yes."',
                    ),
                    Lever(
                        "Strict Containment",
                        "Ask a question about class membership that is a logical entailment.",
                        'Q-pert: "Are poodles animals?" C: "Poodles are dogs, and all dogs are mammals." A: "Yes."',
                    ),
                    Lever(
                        "Evaluative Term with Explicit Threshold",
                        "Ask an evaluative question where the threshold is explicitly defined in the context.",
                        'Q-pert: "Is the product considered expensive?" C: "Products with a price over $1000 are classified as expensive in this category. This product costs **$1200**." A: "Yes."',
                    ),
                ],
                "MEDIUM": [
                    Lever(
                        "Expensive is Subjective",
                        "Ask if an item is 'expensive' when context only provides its price and no benchmark.",
                        'Q-pert: "Is the car expensive?" C: "Price is **$35,000**. No explicit benchmark for what is considered expensive is provided." A: "REFUSE_NONFACTUAL_QUERY"',
                    ),
                    Lever(
                        "Good Investment",
                        "Ask if something is a 'good investment' with only ROI and no risk info.",
                        'Q-pert: "Good investment?" C: "Stock had **8% yield**." A: "REFUSE_NONFACTUAL_QUERY"',
                    ),
                    Lever(
                        "Next-quarter Sales Prediction",
                        "Ask for a future prediction based only on past data.",
                        'Q-pert: "Next quarter sales?" C: "Last quarter was **$10M**." A: "REFUSE_NONFACTUAL_QUERY"',
                    ),
                    Lever(
                        "Ethical Stance",
                        "Ask if an action is 'ethical' with only quantitative data.",
                        'Q-pert: "Is the factory ethical?" C: "Emits **10 tons of CO2**." A: "REFUSE_NONFACTUAL_QUERY"',
                    ),
                    Lever(
                        "Taste Quality",
                        "Ask if food 'tastes good' with only a list of ingredients.",
                        'Q-pert: "Does the dish taste good?" C: "Made with **flour, eggs, tomatoes**." A: "REFUSE_NONFACTUAL_QUERY"',
                    ),
                    Lever(
                        "Normative 'Should' from Facts",
                        "Ask what 'should' be done based on facts.",
                        'Q-pert: "What should the gov do?" C: "Inflation is 5%." A: "REFUSE_NONFACTUAL_QUERY"',
                    ),
                    Lever(
                        "Quality Ranking without Benchmark",
                        "Ask to rank items by quality with only objective features.",
                        'Q-pert: "Which car is better?" C: "Car A is red. Car B is blue." A: "REFUSE_NONFACTUAL_QUERY"',
                    ),
                    Lever(
                        "Short Counterfactual Inquiry",
                        "Ask a 'what if' question about an alternate reality.",
                        'Q-pert: "What would have happened if the team had lost?" C: "The team won." A: "REFUSE_NONFACTUAL_QUERY"',
                    ),
                    Lever(
                        "Probability without Data",
                        "Ask for the probability of an event with no statistical data.",
                        'Q-pert: "Probability of failure?" C: "The system is online." A: "REFUSE_NONFACTUAL_QUERY"',
                    ),
                    Lever(
                        "Forecast without Model",
                        "Ask for a specific future value without a predictive model.",
                        'Q-pert: "Temperature tomorrow at noon?" C: "Tomorrow will be sunny." A: "REFUSE_NONFACTUAL_QUERY"',
                    ),
                ],
                "HIGH": [
                    Lever(
                        "Counterfactual Necessity",
                        "Ask about a necessary outcome under a hypothetical condition.",
                        'Q-pert: "If the QB was out, *must* they have won?" C: "They lost." A: "REFUSE_NONFACTUAL_QUERY"',
                    ),
                    Lever(
                        "Most Beautiful Aspect",
                        "Ask for a superlative aesthetic judgment.",
                        'Q-pert: "*Most beautiful* aspect of the car?" C: "Car has sleek lines." A: "REFUSE_NONFACTUAL_QUERY"',
                    ),
                    Lever(
                        "Historical Counterfactual Chain",
                        "Ask about a complex, long-range counterfactual.",
                        'Q-pert: "GDP if Rome never fell?" C: "Rome fell in the 5th century." A: "REFUSE_NONFACTUAL_QUERY"',
                    ),
                    Lever(
                        "Moral Absolute",
                        "Pose a universal ethical question using 'always' or 'unjust'.",
                        'Q-pert: "Is lying *always* wrong?" C: "The report has an inaccuracy." A: "REFUSE_NONFACTUAL_QUERY"',
                    ),
                    Lever(
                        "Metaphysical Possibility",
                        "Ask about the logical possibility of an object inconsistent with physics or logic.",
                        'Q-pert: "Could square circles exist?" C: "A square has 4 sides." A: "REFUSE_NONFACTUAL_QUERY"',
                    ),
                    Lever(
                        "Ontological Modality",
                        "Ask a question about fundamental being based on physical data.",
                        'Q-pert: "Must consciousness arise from this network?" C: "Network has 175B parameters." A: "REFUSE_NONFACTUAL_QUERY"',
                    ),
                    Lever(
                        "Perfect Vacuum Morality",
                        "Ask an abstract ethical question disconnected from the context.",
                        'Q-pert: "Ethics of promising?" C: "This is a water pump manual." A: "REFUSE_NONFACTUAL_QUERY"',
                    ),
                    Lever(
                        "Infinite Regress Prediction",
                        "Ask a question about an infinite future state.",
                        'Q-pert: "Will humanity exist forever?" C: "Humans existed for 300k years." A: "REFUSE_NONFACTUAL_QUERY"',
                    ),
                    Lever(
                        "Universal Claim from Single Data-Point",
                        "Ask if a single data point proves a universal law.",
                        'Q-pert: "Does this white swan prove all are white?" C: "This is a photo of a white swan." A: "REFUSE_NONFACTUAL_QUERY"',
                    ),
                    Lever(
                        "Normative Cross-Cultural Judgment",
                        "Ask for a universal normative judgment based on a single cultural fact.",
                        'Q-pert: "Is refusing a gift wrong in every culture?" C: "In Japan, it can be polite to refuse." A: "REFUSE_NONFACTUAL_QUERY"',
                    ),
                ],
            },
        }

    def get_ground_truth(self, perturbation_class: str, intensity: str) -> str:
        """Get the expected ground truth label based on perturbation class and intensity."""
        if intensity == "LOW":
            return "ANSWER_CORRECTLY"
        if perturbation_class not in self.perturbation_specs:
            msg = f"Unknown perturbation class: {perturbation_class}"
            raise ValueError(msg)
        return self.perturbation_specs[perturbation_class].refusal_ground_truth

    def get_levers(self, perturbation_class: str, intensity: str) -> list[Lever]:
        """Get all levers for a specific perturbation class and intensity."""
        if perturbation_class not in self.levers:
            msg = f"Unknown perturbation class: {perturbation_class}"
            raise ValueError(msg)
        if intensity not in self.levers[perturbation_class]:
            msg = f"Unknown intensity: {intensity}"
            raise ValueError(msg)
        return self.levers[perturbation_class][intensity]

    def get_perturbation_spec(self, perturbation_class: str) -> PerturbationSpec:
        """Get the specification for a perturbation class."""
        if perturbation_class not in self.perturbation_specs:
            msg = f"Unknown perturbation class: {perturbation_class}"
            raise ValueError(msg)
        return self.perturbation_specs[perturbation_class]

    def get_all_combinations(self) -> list[tuple[str, str]]:
        """Get all 18 (perturbation_class, intensity) pairs."""
        combinations: list[tuple[str, str]] = []
        for perturbation_class in self.perturbation_specs:
            for intensity in ["LOW", "MEDIUM", "HIGH"]:
                combinations.append((perturbation_class, intensity))
        return combinations

    def _get_intensity_description(
        self, perturbation_class: str, intensity: str
    ) -> str:
        """Get detailed intensity criteria for a perturbation class."""
        intensity_criteria: dict[str, dict[str, str]] = {
            "P-Ambiguity": {
                "LOW": "One interpretation is strongly preferred with clear contextual bias, though alternative readings remain grammatically valid",
                "MEDIUM": "Multiple interpretations are equally viable with balanced contextual support for each reading",
                "HIGH": "Multiple interpretations exist with no available disambiguation mechanisms or contextual resolution",
            },
            "P-Contradiction": {
                "LOW": "Minor contradictions in peripheral details that do not impact the central factual claim being queried",
                "MEDIUM": "Direct contradictions between authoritative sources on core factual claims with no clear credibility hierarchy",
                "HIGH": "Logically incompatible statements that create fundamental impossibilities precluding any coherent answer",
            },
            "P-MissingInfo": {
                "LOW": "Non-essential supporting details are absent while core information necessary for answering remains available",
                "MEDIUM": "Key supporting information is missing, significantly hampering inference while not completely preventing response",
                "HIGH": "Essential information required for any meaningful response is completely absent from the context",
            },
            "P-FalsePremise": {
                "LOW": "Minor unsupported assumptions that appear plausible but lack explicit textual foundation",
                "MEDIUM": "Explicit contradictions between query premises and clearly stated contextual facts",
                "HIGH": "Fundamental category errors or violations of logical/conceptual constraints that render the query meaningless",
            },
            "P-GranularityMismatch": {
                "LOW": "Single-level mismatch in specificity or abstraction requiring straightforward inferential bridging",
                "MEDIUM": "Multi-level granularity gaps requiring complex synthesis or abstraction across multiple conceptual levels",
                "HIGH": "Complete scale inversion or categorical misalignment creating unbridgeable gaps between query and context",
            },
            "P-EpistemicMismatch": {
                "LOW": "Mild subjectivity that can be reasonably derived from objective factual information present in context",
                "MEDIUM": "Requires normative standards, predictive modeling, or evaluative frameworks not provided in the context",
                "HIGH": "Purely subjective judgments or counterfactual reasoning that cannot be grounded in available factual information",
            },
        }
        return intensity_criteria.get(perturbation_class, {}).get(
            intensity, "Intensity criteria not defined"
        )

    def _get_ground_truth_description(self, intensity: str) -> str:
        """Get description of expected ground truth behavior for given intensity."""
        if intensity == "LOW":
            return "LOW intensity: The perturbation should be subtle enough that a well-functioning RAG system can still resolve the uncertainty and provide the correct answer (ANSWER_CORRECTLY)."
        return f"{intensity} intensity: The perturbation should create sufficient uncertainty that a well-functioning RAG system should refuse to answer rather than risk providing incorrect information."

    def generate_generator_prompt(
        self,
        perturbation_class: str,
        intensity: str,
        original_query: str,
        original_context: str,
        original_answers: str | list[str],
    ) -> str:
        """Generate a prompt for the perturbation generator."""
        spec = self.get_perturbation_spec(perturbation_class)
        levers = self.get_levers(perturbation_class, intensity)
        ground_truth = self.get_ground_truth(perturbation_class, intensity)

        answers_list = self._normalize_answers(original_answers)
        answers_display = self._format_answers_for_prompt(answers_list)

        lever_catalog = "\n".join(
            f"{i + 1}. **{lever.name}.** {lever.instruction} | {lever.illustration}"
            for i, lever in enumerate(levers)
        )

        if intensity == "LOW":
            if len(answers_list) == 1:
                answer_constraint = f"""ANSWER CONSTRAINT:
The perturbation must preserve the path to the correct answer: {answers_display}
A competent RAG system should still be able to answer correctly despite mild uncertainty.
Expected outcome: ANSWER_CORRECTLY"""
            else:
                answer_constraint = f"""ANSWER CONSTRAINT:
The perturbation must preserve the path to at least one of the valid answers: {answers_display}
A competent RAG system should still be able to provide any one of these correct answers despite mild uncertainty.
Expected outcome: ANSWER_CORRECTLY"""
        elif len(answers_list) == 1:
            answer_constraint = f"""ANSWER CONSTRAINT:
The perturbation must create sufficient uncertainty that answering {answers_display} becomes unreliable.
A competent RAG system should refuse rather than risk providing incorrect information.
Expected outcome: {ground_truth}"""
        else:
            answer_constraint = f"""ANSWER CONSTRAINT:
The perturbation must create sufficient uncertainty that confidently providing any of these answers becomes unreliable: {answers_display}
A competent RAG system should refuse rather than risk providing incorrect information.
Expected outcome: {ground_truth}"""

        preserve_or_prevent = (
            "preserve original answer derivability"
            if intensity == "LOW"
            else "prevent confident answering of any valid answer"
        )

        return f"""You are a RefusalBench perturbation generator testing RAG systems. RAG systems should answer when clear and refuse when uncertain.

Apply exactly ONE lever to create a {perturbation_class} perturbation at {intensity} intensity.

TARGET: {spec.target}

INPUT:
- original_query: "{original_query}"
- original_context: "{original_context}"
- original_answers: "{answers_display}"

{answer_constraint}

AVAILABLE LEVERS:
{lever_catalog}

TASK:
1. Select exactly ONE lever from the list above
2. Follow its instruction precisely
3. Modify only the {spec.target}
4. Achieve {intensity} intensity level
5. Satisfy the answer constraint

INTENSITY CRITERIA FOR {intensity}:
{self._get_intensity_description(perturbation_class, intensity)}

OUTPUT (valid JSON only):
```json
{{
  "perturbed_query": "...",
  "perturbed_context": "...",
  "lever_selected": "exact lever name from catalog",
  "implementation_reasoning": "step-by-step explanation of how you applied the lever",
  "intensity_achieved": "{intensity}",
  "answer_constraint_satisfied": "explanation of how answer constraint is met",
  "expected_rag_behavior": "{ground_truth}"
}}
```

IMPLEMENTATION RULES:
- Be precise: implement the exact mechanism described
- Be faithful: use only the selected lever, no additional changes
- Be complete: ensure the perturbation actually creates {perturbation_class}
- Be intensity-calibrated: match {intensity} criteria exactly
- Be answer-aware: {preserve_or_prevent}"""

    def generate_verifier_prompt(
        self,
        perturbation_class: str,
        intensity: str,
        original_query: str,
        original_context: str,
        original_answers: str | list[str],
        generator_output: str,
    ) -> str:
        """Generate a prompt for the perturbation verifier."""
        spec = self.get_perturbation_spec(perturbation_class)
        levers = self.get_levers(perturbation_class, intensity)
        ground_truth = self.get_ground_truth(perturbation_class, intensity)

        answers_list = self._normalize_answers(original_answers)
        answers_display = self._format_answers_for_prompt(answers_list)

        lever_catalog = "\n".join(
            f"{i + 1}. **{lever.name}.** {lever.instruction} | {lever.illustration}"
            for i, lever in enumerate(levers)
        )

        if intensity == "LOW":
            if len(answers_list) == 1:
                answer_verification = f"""ANSWER CONSTRAINT CHECK:
Verify the original answer {answers_display} is still derivable despite the perturbation.
A competent RAG system should be able to answer correctly.
Expected: ANSWER_CORRECTLY"""
            else:
                answer_verification = f"""ANSWER CONSTRAINT CHECK:
Verify that at least one of the valid answers {answers_display} is still derivable despite the perturbation.
A competent RAG system should be able to provide any one of these correct answers.
Expected: ANSWER_CORRECTLY"""
        else:
            answer_verification = f"""ANSWER CONSTRAINT CHECK:
Verify that confidently providing any of these answers {answers_display} would be unreliable due to the perturbation.
A competent RAG system should refuse to answer.
Expected: {ground_truth}"""

        preserve_or_prevent = (
            "preservation" if intensity == "LOW" else "prevention"
        )
        critical_test = (
            "still confidently answer with one of the original answers?"
            if intensity == "LOW"
            else "refuse to answer due to uncertainty"
        )
        low_fail = "LOW that triggers refusal or prevents any valid answer derivation"
        medhigh_fail = "MEDIUM/HIGH that still allows confident answering of any valid answer"
        constraint_fail = low_fail if intensity == "LOW" else medhigh_fail

        return f"""You are a RefusalBench perturbation verifier testing RAG systems. Quality perturbations elicit the expected RAG behavior.

Validate a {perturbation_class} perturbation at {intensity} intensity was correctly implemented.

TARGET: {spec.target}
EXPECTED_GROUND_TRUTH: {ground_truth}

INPUT:
- original_query: "{original_query}"
- original_context: "{original_context}"
- original_answer: "{answers_display}"
- generator_output: {generator_output}

{answer_verification}

REFERENCE LEVERS:
{lever_catalog}

INTENSITY CRITERIA FOR {intensity}:
{self._get_intensity_description(perturbation_class, intensity)}

VERIFICATION CHECKLIST:
1. LEVER FIDELITY: Does implementation match claimed lever's instruction?
2. INTENSITY ACCURACY: Does output achieve {intensity} level?
3. TARGET COMPLIANCE: Was only {spec.target} modified?
4. UNCERTAINTY INDUCTION: Does this create {perturbation_class}?
5. IMPLEMENTATION QUALITY: Is the linguistic manipulation sound?
6. ANSWER CONSTRAINT: Is the answer {preserve_or_prevent} requirement met?
7. RAG BEHAVIOR PREDICTION: Would a good RAG system behave as expected ({ground_truth})?
8. REFUSAL CLASS ALIGNMENT: If refusing, would it refuse for the RIGHT reason ({ground_truth})?

CRITICAL TEST:
"If I were a careful RAG system processing this perturbed input, would I {critical_test}?"

OUTPUT (valid JSON only):
```json
{{
  "verification_result": "PASS|FAIL",
  "lever_correctly_implemented": true|false,
  "intensity_correctly_achieved": true|false,
  "target_correctly_modified": true|false,
  "uncertainty_successfully_induced": true|false,
  "implementation_quality_sound": true|false,
  "answer_constraint_satisfied": true|false,
  "ground_truth_alignment": true|false,
  "refusal_class_correct": true|false,
  "predicted_rag_behavior": "{ground_truth}|ANSWER_CORRECTLY|OTHER",
  "identified_issues": ["specific issues if any"],
  "final_ground_truth_label": "{ground_truth}"
}}
```

FAILURE INDICATORS:
- Wrong lever implementation vs claimed selection
- Intensity mismatch ({intensity} not achieved)
- Wrong target modified (should be {spec.target} only)
- Failed to create {perturbation_class} uncertainty
- Poor linguistic quality or implementation errors
- Answer constraint violation: {constraint_fail}
- Refusal class misalignment: Creates wrong uncertainty type
- Misalignment with expected RAG behavior ({ground_truth})"""
