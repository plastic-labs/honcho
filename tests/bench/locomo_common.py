"""
Common utilities for LoCoMo benchmark test runners.

Shared functionality between the Honcho benchmark and baseline benchmark.

LoCoMo evaluates very long-term conversational memory of LLM agents across
five question categories:
1. Single-hop - Direct factual recall from conversations
2. Multi-hop - Reasoning across multiple pieces of information
3. Temporal - Understanding time-based relationships and sequences
4. Commonsense/World knowledge - Applying broader contextual understanding
5. Adversarial - Challenging questions that cannot be answered from the conversation

Reference: https://github.com/snap-research/locomo
Paper: https://arxiv.org/abs/2402.17753
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import tiktoken
from openai import AsyncOpenAI
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


# Category ID to name mapping
CATEGORY_NAMES: dict[int, str] = {
    1: "single_hop",
    2: "multi_hop",
    3: "temporal",
    4: "commonsense",
    5: "adversarial",  # Should be filtered out during evaluation
}


class QuestionResult(TypedDict):
    """Type definition for question evaluation results."""

    question_id: int
    question: str
    expected_answer: str
    actual_response: str
    category: int
    category_name: str
    evidence: list[str]
    judgment: dict[str, Any]
    passed: bool


class ConversationResult(TypedDict):
    """Type definition for conversation execution results."""

    sample_id: str
    speaker_a: str
    speaker_b: str
    total_sessions: int
    total_turns: int
    total_tokens: int
    question_results: list[QuestionResult]
    category_scores: dict[str, dict[str, Any]]
    overall_score: float
    error: str | None
    start_time: float
    end_time: float
    duration_seconds: float


def format_duration(total_seconds: float) -> str:
    """Format a duration in seconds into a human-readable string."""
    minutes = int(total_seconds // 60)
    if minutes > 0:
        seconds_rounded = int(round(total_seconds - minutes * 60))
        if seconds_rounded == 60:
            minutes += 1
            seconds_rounded = 0
        return f"{minutes}m{seconds_rounded:02d}s"
    return f"{total_seconds:.2f}s"


def calculate_tokens(text: str) -> int:
    """Calculate tokens for a given text."""
    tokenizer = tiktoken.get_encoding("o200k_base")
    try:
        return len(
            tokenizer.encode(
                text,
                disallowed_special=(tokenizer.special_tokens_set - {"<|endoftext|>"}),
            )
        )
    except Exception:
        return len(text) // 4


def load_locomo_data(data_file: Path) -> list[dict[str, Any]]:
    """
    Load LoCoMo data from a JSON file.

    Args:
        data_file: Path to the LoCoMo JSON file

    Returns:
        List of conversation dictionaries
    """
    with open(data_file) as f:
        return json.load(f)


def parse_locomo_date(date_str: str) -> datetime:
    """
    Parse LoCoMo date format to datetime.

    Args:
        date_str: Date string in format "H:MM am/pm on D Month, YYYY"
                  e.g., "1:56 pm on 8 May, 2023"

    Returns:
        Parsed datetime object
    """
    try:
        # Handle formats like "1:56 pm on 8 May, 2023"
        # Remove "on " and parse
        date_str = date_str.replace(" on ", " ")
        # Try parsing with different formats
        for fmt in ["%I:%M %p %d %B, %Y", "%I:%M %p %d %B %Y"]:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        # If all fail, return a default datetime
        logger.warning(f"Could not parse date '{date_str}', using current time")
        return datetime.now()
    except Exception as e:
        logger.warning(f"Error parsing date '{date_str}': {e}")
        return datetime.now()


def extract_sessions(
    conversation: dict[str, Any],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """
    Extract sessions from a LoCoMo conversation.

    Args:
        conversation: The conversation dict containing session_N and session_N_date_time

    Returns:
        List of tuples (date_time_str, messages) where messages contain
        'speaker', 'text', 'dia_id', and optional 'img_url', 'blip_caption', 'query'
    """
    sessions: list[tuple[str, list[dict[str, Any]]]] = []

    # Find all session keys
    session_keys = sorted(
        [k for k in conversation if re.match(r"session_\d+$", k)],
        key=lambda x: int(x.split("_")[1]),
    )

    for session_key in session_keys:
        session_num = session_key.split("_")[1]
        date_key = f"session_{session_num}_date_time"
        date_str = conversation.get(date_key, "")
        messages = conversation.get(session_key, [])
        sessions.append((date_str, messages))

    return sessions


def extract_all_messages(
    conversation: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Extract all messages from all sessions in a conversation.

    Args:
        conversation: The conversation dict

    Returns:
        List of message dicts with 'speaker', 'text', 'dia_id', and optional image fields
    """
    all_messages: list[dict[str, Any]] = []
    sessions = extract_sessions(conversation)

    for _date_str, messages in sessions:
        for msg in messages:
            message_dict: dict[str, Any] = {
                "speaker": msg.get("speaker", ""),
                "text": msg.get("text", ""),
                "dia_id": msg.get("dia_id", ""),
            }
            # Preserve image fields if present
            if msg.get("blip_caption"):
                message_dict["blip_caption"] = msg["blip_caption"]
            if msg.get("img_url"):
                message_dict["img_url"] = msg["img_url"]
            if msg.get("query"):
                message_dict["query"] = msg["query"]

            all_messages.append(message_dict)

    return all_messages


def get_evidence_context(
    conversation: dict[str, Any],
    evidence_ids: list[str],
) -> str | None:
    """
    Extract evidence messages from a conversation based on dia_id references.

    Args:
        conversation: The conversation dict containing sessions
        evidence_ids: List of dia_id references (e.g., ["D1:3", "D2:8"])

    Returns:
        Formatted string of evidence messages, or None if no evidence found
    """
    if not evidence_ids:
        return None

    # Build a mapping of dia_id to message
    all_messages = extract_all_messages(conversation)
    dia_id_to_msg = {msg["dia_id"]: msg for msg in all_messages if msg.get("dia_id")}

    # Extract evidence messages
    evidence_messages: list[str] = []
    for eid in evidence_ids:
        if eid in dia_id_to_msg:
            msg = dia_id_to_msg[eid]
            text = msg["text"]
            # Include image caption if present
            if msg.get("blip_caption"):
                text = f"{text} [Image: {msg['blip_caption']}]"
            evidence_messages.append(f"[{eid}] {msg['speaker']}: {text}")

    if not evidence_messages:
        return None

    return "\n".join(evidence_messages)


def filter_questions(
    qa_list: list[dict[str, Any]],
    exclude_adversarial: bool = False,
    question_ids: list[int] | None = None,
    test_count: int | None = None,
) -> list[dict[str, Any]]:
    """
    Filter questions based on criteria.

    Args:
        qa_list: List of QA dictionaries
        exclude_adversarial: If True, exclude category 5 (adversarial) questions
        question_ids: Optional list of specific question indices to include
        test_count: Optional limit on number of questions

    Returns:
        Filtered list of questions
    """
    filtered = qa_list

    # Filter out adversarial questions (category 5)
    if exclude_adversarial:
        filtered = [q for q in filtered if q.get("category") != 5]

    # Filter by specific question IDs
    if question_ids is not None:
        filtered = [
            q
            for i, q in enumerate(filtered)
            if i in question_ids or (i + 1) in question_ids
        ]

    # Limit to first N questions
    if test_count is not None and test_count > 0:
        filtered = filtered[:test_count]

    return filtered


def _build_judge_system_prompt(context: str | None) -> str:
    """Build the system prompt for LoCoMo evaluation.

    Args:
        context: Optional evidence context from the conversation

    Returns:
        The system prompt for the judge model
    """
    return f"""You are evaluating whether a synthesized answer adequately addresses a query about a user based on available conclusions.
## EVIDENCE CONTEXT
{context if context else "No evidence provided."}
## EVALUATION CONTEXT
You will evaluate:
1. **Query**: The specific question asked about the user
2. **Synthesized Answer**: The response generated from available conclusions
3. **Gold Standard Answer**: The expected/correct answer
## EVALUATION CRITERIA
Judge the synthesized answer as SUFFICIENT or INSUFFICIENT based on:
### Content Completeness
- Does the answer address what the query is asking?
- Are all key aspects of the gold answer covered (even if phrased differently)?
- Is critical information missing that would change the answer's usefulness?
### Semantic Accuracy
- Are any factual errors or contradictions present?
## ACCEPTABLE DIFFERENCES
The following differences are ACCEPTABLE and should NOT result in INSUFFICIENT:
- Different phrasing or word choice that still conveys the same or very similar meaning, especially in cases where the question is tentative or open-ended.
- Additional relevant context beyond the gold answer (including evidence supplied above). This includes the case where the synthesized answer is longer and more detailed than the gold answer, potentially even including additional information that is not explicitly stated in the gold answer but is still broadly relevant to the query. Do NOT penalize the synthesized answer for including additional information that is not explicitly stated in the gold answer.
- **The synthesized answer explicitly includes the full gold answer text (even if surrounded by additional or unrelated details).  If the gold answer appears within the synthesized answer, you MUST mark the answer as SUFFICIENT.**
- More detailed explanations of reasoning or evidence
- Appropriate confidence qualifiers (e.g., "likely", "probably") when warranted
- Differences in length, with the synthesized answer being longer and even more circuitous or indirect in its addressing of the query, as long as it conveys the same meaning
- Minor format or structure variations
## EVIDENCE-GOLD ANSWER CONSISTENCY CHECK
It is possible for the gold answers to be wrong. Sometimes it may not be fully supported by or follow logically from the evidence messages, instead constituting a guess or assumption. Additionally, the gold answers are generated automatically based on the limited set of evidence messages provided above, whereas if additional context were to be taken into account, the answer might be different. In these cases, we must not penalize the synthesized answer for not being exactly the same as the gold answer.
Before deciding, verify whether the gold answer logically and necessarily follows from the supplied evidence context. If you identify a mismatch or missing logical link **and** the synthesized answer acknowledges this uncertainty or provides a more cautious, evidence-grounded explanation (optionally leveraging additional context beyond the ground truth evidence above), treat the synthesized answer as SUFFICIENT even when it diverges in wording or conclusion from the gold answer.  In short:
* If the gold answer over-claims beyond what the evidence shows, do **not** penalize a synthesized answer that appropriately qualifies the claim or offers a plausible alternative consistent with evidence.
* This includes the case where the synthesized answer is ambivalent or uncertain about the answer, as long as it provides sufficient evidence to support not providing a definitive, categorical answer.
* If the synthesized answer clearly explains the gap and gives a better-supported conclusion, mark it SUFFICIENT.
## UNACCEPTABLE DIFFERENCES
The following DO warrant an INSUFFICIENT rating:
- Irreconcilable errors or contradictions with the gold answer **and** the evidence context
- Missing information central to answering the query, such that its absence would change the meaning of the answer
- Does not address the question being asked
## YOUR TASK
First, analyze what the query is asking **and** how well both answers are supported by the evidence context.
Then, provide 2 brief 2-3 sentence arguments for both SUFFICIENT and INSUFFICIENT:
**Arguments for SUFFICIENT:**
- List reasons why the synthesized answer adequately addresses the query
- Note what key information from the gold answer is present or why deviations are justified by the evidence
- Note whether the gold answer is wrong or not necessarily true given the evidence above
**Arguments for INSUFFICIENT:**
- List reasons why the synthesized answer fails to address the question.

Based on weighing these arguments, provide 2-3 sentences to determine if the synthesized answer is sufficient. In your weighing, consider whether the synthesized answer might be a better answer than the gold answer given the evidence above.
Finally, set is_sufficient to true if sufficient or false if insufficient.
Your response MUST be a valid JSON object with EXACTLY these keys:
  - arguments_for_sufficient (string)
  - arguments_for_insufficient (string)
  - final_reasoning (string)
  - is_sufficient (boolean)
Return ONLY this JSON object and nothing else."""


def _build_judge_user_prompt(
    question: str,
    answer: str,
    response: str,
) -> str:
    """Build the user prompt for LoCoMo evaluation.

    Args:
        question: The question asked
        answer: Expected answer from the test
        response: Actual response from the system under test

    Returns:
        The user prompt for the judge model
    """
    return f"""Query: {question}
Gold Answer: {answer}
Synthesized Answer: {response}"""


async def judge_response(
    openai_client: AsyncOpenAI,
    question: str,
    expected_answer: str,
    actual_response: str,
    evidence_context: str | None = None,
) -> dict[str, Any]:
    """Use GPT-4o-mini to judge if the actual response matches the expected answer.

    This judge is designed to be lenient towards verbose answers that contain
    additional context, as long as they address the question. It also considers
    whether the gold answer is actually correct given the evidence.

    Args:
        openai_client: OpenAI client instance
        question: The question asked
        expected_answer: Expected answer from the test
        actual_response: Actual response from the system under test
        evidence_context: Optional evidence messages from the conversation

    Returns:
        Judgment result with pass/fail and reasoning
    """
    try:
        system_prompt = _build_judge_system_prompt(evidence_context)
        user_prompt = _build_judge_user_prompt(
            question, expected_answer, actual_response
        )

        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1024,
            temperature=0,
            n=1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        if not response.choices:
            raise ValueError("OpenAI returned empty response")

        eval_response = response.choices[0].message.content
        if eval_response is None:
            raise ValueError("No text content in response")

        # Parse JSON response
        try:
            # Strip any markdown code blocks if present
            json_str = eval_response.strip()
            if json_str.startswith("```"):
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
                json_str = json_str.strip()

            result = json.loads(json_str)
            passed = result.get("is_sufficient", False)
            reasoning = result.get("final_reasoning", eval_response.strip())

            return {
                "passed": passed,
                "reasoning": reasoning,
                "arguments_for_sufficient": result.get("arguments_for_sufficient", ""),
                "arguments_for_insufficient": result.get(
                    "arguments_for_insufficient", ""
                ),
            }
        except json.JSONDecodeError:
            # Fallback: check for "sufficient" in the response
            passed = 'is_sufficient": true' in eval_response.lower() or (
                "sufficient" in eval_response.lower()
                and "insufficient" not in eval_response.lower()
            )
            return {
                "passed": passed,
                "reasoning": eval_response.strip(),
            }

    except Exception as e:
        logger.error(f"Error judging response: {e}")
        # Fallback to simple string matching
        is_correct = expected_answer.lower() in actual_response.lower()
        return {
            "passed": is_correct,
            "reasoning": f"Fallback string matching due to error: {'Match found' if is_correct else 'No match found'}",
        }


def calculate_category_scores(
    question_results: list[QuestionResult],
) -> dict[str, dict[str, Any]]:
    """
    Calculate scores grouped by question category.

    Args:
        question_results: List of question results

    Returns:
        Dictionary mapping category name to statistics
    """
    category_stats: dict[str, dict[str, Any]] = {}

    for qr in question_results:
        cat_name = qr["category_name"]
        if cat_name not in category_stats:
            category_stats[cat_name] = {
                "total": 0,
                "passed": 0,
            }

        category_stats[cat_name]["total"] += 1
        if qr["passed"]:
            category_stats[cat_name]["passed"] += 1

    # Calculate success rates
    for cat_name in category_stats:
        stats = category_stats[cat_name]
        stats["success_rate"] = (
            (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        )

    return category_stats


def print_summary(
    results: list[ConversationResult], total_elapsed_seconds: float
) -> None:
    """Print a summary of all test results."""
    print(f"\n{'=' * 80}")
    print("LOCOMO BENCHMARK EXECUTION SUMMARY")
    print(f"{'=' * 80}")

    total_conversations = len(results)
    total_questions = sum(len(r["question_results"]) for r in results)

    print(f"Total Conversations: {total_conversations}")
    print(f"Total Questions: {total_questions}")
    print(f"Total Test Time: {format_duration(total_elapsed_seconds)}")

    # Aggregate category scores across all conversations
    category_totals: dict[str, dict[str, Any]] = {}
    for result in results:
        for cat_name, stats in result["category_scores"].items():
            if cat_name not in category_totals:
                category_totals[cat_name] = {
                    "total": 0,
                    "passed": 0,
                }
            category_totals[cat_name]["total"] += stats["total"]
            category_totals[cat_name]["passed"] += stats["passed"]

    print("\nScores by Question Category:")
    print(f"{'Category':<20} {'Total':<8} {'Passed':<8} {'Rate':<10}")
    print(f"{'-' * 20} {'-' * 8} {'-' * 8} {'-' * 10}")

    for cat_name in sorted(category_totals.keys()):
        stats = category_totals[cat_name]
        rate = (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f"{cat_name:<20} {stats['total']:<8} {stats['passed']:<8} {rate:<10.1f}%")

    # Overall averages
    overall_scores = [r["overall_score"] for r in results]
    overall_avg = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0

    print(f"\n{'Overall Average Score':<30}: {overall_avg:.3f}")

    print(f"{'=' * 80}")


def generate_json_summary(
    results: list[ConversationResult],
    total_elapsed_seconds: float,
    output_file: Path,
    metadata_extra: dict[str, Any] | None = None,
) -> None:
    """Generate a comprehensive JSON summary of test results."""
    total_conversations = len(results)
    total_questions = sum(len(r["question_results"]) for r in results)

    # Aggregate category scores
    category_totals: dict[str, dict[str, Any]] = {}
    for result in results:
        for cat_name, stats in result["category_scores"].items():
            if cat_name not in category_totals:
                category_totals[cat_name] = {
                    "total": 0,
                    "passed": 0,
                }
            category_totals[cat_name]["total"] += stats["total"]
            category_totals[cat_name]["passed"] += stats["passed"]

    category_averages = {
        cat: {
            "total": stats["total"],
            "passed": stats["passed"],
            "success_rate": (stats["passed"] / stats["total"]) * 100
            if stats["total"] > 0
            else 0,
        }
        for cat, stats in category_totals.items()
    }

    # Overall averages
    overall_scores = [r["overall_score"] for r in results]
    overall_avg = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0

    metadata = {
        "benchmark": "LoCoMo",
        "execution_timestamp": datetime.now().isoformat(),
        "runner_version": "1.0.0",
    }
    if metadata_extra:
        metadata.update(metadata_extra)

    summary = {
        "metadata": metadata,
        "summary_statistics": {
            "total_conversations": total_conversations,
            "total_questions": total_questions,
            "overall_average_score": overall_avg,
            "category_statistics": category_averages,
        },
        "timing": {
            "total_duration_seconds": total_elapsed_seconds,
        },
        "detailed_results": results,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nJSON summary written to: {output_file}")
