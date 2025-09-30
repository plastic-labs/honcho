import json
import logging
import re
from typing import Any

from json_repair import repair_json  # pyright: ignore

logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine.Engine").disabled = True


def comprehensive_json_repair(json_str: str) -> str:
    """Comprehensively repair malformed JSON with multiple strategies"""

    # Strategy 1: Handle truncated JSON by parsing what we can
    repaired = try_partial_parse_repair(json_str)
    if repaired:
        return repaired

    # Strategy 2: Smart bracket/brace matching with context awareness
    repaired = try_contextual_closure_repair(json_str)
    if repaired:
        return repaired

    # Strategy 3: Line-by-line reconstruction
    repaired = try_line_reconstruction_repair(json_str)
    if repaired:
        return repaired

    # Strategy 4: Regex-based common pattern fixes
    repaired = try_regex_pattern_repair(json_str)
    if repaired:
        return repaired

    # Fallback: Original simple method
    return simple_bracket_repair(json_str)


def try_partial_parse_repair(json_str: str) -> str | None:
    """Try to parse JSON incrementally and reconstruct from valid parts"""
    try:
        # First, try to find the last complete object/array
        lines = json_str.split("\n")

        for i in range(len(lines), 0, -1):
            partial = "\n".join(lines[:i])

            # Try different closure strategies
            for closure_attempt in generate_closure_attempts(partial):
                try:
                    json.loads(closure_attempt)
                    return closure_attempt
                except json.JSONDecodeError:
                    continue

        return None
    except Exception:
        return None


def generate_closure_attempts(partial_json: str) -> list[str]:
    """Generate different ways to close the JSON structure"""
    attempts: list[str] = []

    # Analyze the structure to understand what's open
    stack: list[tuple[str, int]] = []
    in_string = False
    escape_next = False

    for i, char in enumerate(partial_json):
        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char in "({[":
            stack.append((char, i))
        elif char in ")}]" and stack:
            opener, _ = stack.pop()
            # Verify matching pairs
            if not (
                (char == ")" and opener == "(")
                or (char == "}" and opener == "{")
                or (char == "]" and opener == "[")
            ):
                # Mismatched - this is likely where corruption started
                break

    # Generate closure attempts based on what's still open
    base = partial_json.rstrip()

    # Remove trailing comma if present
    if base.rstrip().endswith(","):
        base = base.rstrip()[:-1]
        attempts.append(base)

    # Close based on stack
    closures: list[str] = []
    for opener, _ in reversed(stack):
        if opener == "{":
            closures.append("}")
        elif opener == "[":
            closures.append("]")
        elif opener == "(":
            closures.append(")")

    # Try different combinations
    attempts.append(base + "".join(closures))

    # Try closing just objects/arrays (ignore parentheses)
    obj_closures = [c for c in closures if c in "]}"]
    attempts.append(base + "".join(obj_closures))

    # Try adding missing quotes if we're in a string
    if in_string:
        attempts.append(base + '"' + "".join(closures))

    return attempts


def try_contextual_closure_repair(json_str: str) -> str | None:
    """Smart closure repair based on JSON context"""
    try:
        # Find the last valid JSON token
        tokens: list[dict[str, Any]] = tokenize_json(json_str)

        # Look for patterns that indicate what should come next
        if not tokens:
            return None

        last_token: dict[str, Any] = tokens[-1]

        # If last token is a value, we might need to close objects/arrays
        if last_token["type"] in ["string", "number", "boolean", "null"]:
            return try_close_after_value(json_str, tokens)

        # If last token is a structural element, handle appropriately
        elif last_token["type"] in ["comma", "colon"]:
            return try_complete_structure(json_str, tokens)

        return None
    except Exception:
        return None


def tokenize_json(json_str: str) -> list[dict[str, Any]]:
    """Tokenize JSON string into meaningful components"""
    tokens: list[dict[str, Any]] = []
    i = 0

    while i < len(json_str):
        char = json_str[i]

        # Skip whitespace
        if char.isspace():
            i += 1
            continue

        # String literals
        if char == '"':
            start = i
            i += 1
            while i < len(json_str):
                if json_str[i] == '"' and json_str[i - 1] != "\\":
                    break
                i += 1
            tokens.append(
                {
                    "type": "string",
                    "value": json_str[start : i + 1],
                    "start": start,
                    "end": i,
                }
            )

        # Numbers
        elif char.isdigit() or char == "-":
            start = i
            while i < len(json_str) and (
                json_str[i].isdigit() or json_str[i] in ".-eE"
            ):
                i += 1
            tokens.append(
                {
                    "type": "number",
                    "value": json_str[start:i],
                    "start": start,
                    "end": i - 1,
                }
            )
            continue  # Don't increment i again

        # Structural characters
        elif char in "{}[],:":
            token_type = {
                "{": "object_start",
                "}": "object_end",
                "[": "array_start",
                "]": "array_end",
                ",": "comma",
                ":": "colon",
            }[char]

            tokens.append({"type": token_type, "value": char, "start": i, "end": i})

        # Boolean/null literals
        elif char in "tfn":
            if json_str[i : i + 4] == "true":
                tokens.append(
                    {"type": "boolean", "value": "true", "start": i, "end": i + 3}
                )
                i += 3
            elif json_str[i : i + 5] == "false":
                tokens.append(
                    {"type": "boolean", "value": "false", "start": i, "end": i + 4}
                )
                i += 4
            elif json_str[i : i + 4] == "null":
                tokens.append(
                    {"type": "null", "value": "null", "start": i, "end": i + 3}
                )
                i += 3

        i += 1

    return tokens


def try_close_after_value(json_str: str, tokens: list[dict[str, Any]]) -> str | None:
    """Try to close JSON after a value token"""
    # Analyze nesting to determine what needs to be closed
    nesting_stack: list[str] = []

    for token in tokens[:-1]:  # Exclude the last token (which is the value)
        if token["type"] == "object_start":
            nesting_stack.append("}")
        elif token["type"] == "array_start":
            nesting_stack.append("]")
        elif (
            token["type"] in ["object_end", "array_end"]
            and nesting_stack
            and nesting_stack[-1] == token["value"]
        ):
            nesting_stack.pop()

    # Close remaining open structures
    closure = "".join(reversed(nesting_stack))
    candidate = json_str + closure

    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        return None


def try_complete_structure(json_str: str, tokens: list[dict[str, Any]]) -> str | None:
    """Try to complete JSON ending with structural tokens like comma or colon"""
    last_token = tokens[-1]

    if last_token["type"] == "comma":
        # After comma, we might be missing a key-value pair or array element
        # Try removing the trailing comma first
        trimmed = json_str.rstrip().rstrip(",")
        return try_contextual_closure_repair(trimmed)

    elif last_token["type"] == "colon":
        # After colon, we're missing a value - try adding a placeholder
        candidates = [
            json_str + "null",
            json_str + '""',
            json_str + "[]",
            json_str + "{}",
        ]

        for candidate in candidates:
            try:
                repaired = try_contextual_closure_repair(candidate)
                if repaired:
                    return repaired
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

    return None


def try_line_reconstruction_repair(json_str: str) -> str | None:
    """Try to reconstruct JSON line by line"""
    lines = json_str.split("\n")

    # Find the last line that makes the JSON valid when truncated there
    for i in range(len(lines), 0, -1):
        partial_lines = lines[:i]
        partial_json = "\n".join(partial_lines)

        # Try to repair this partial JSON
        repaired = try_contextual_closure_repair(partial_json)
        if repaired:
            return repaired

    return None


def try_regex_pattern_repair(json_str: str) -> str | None:
    """Use regex to fix common JSON formatting issues"""
    fixed = json_str

    # Remove trailing commas before closing braces/brackets
    fixed = re.sub(r",(\s*[}\]])", r"\1", fixed)

    # Fix unescaped quotes in strings (basic attempt)
    fixed = re.sub(r'(?<!\\)"(?![,\]\}:\s]|$)', r'\\"', fixed)

    # Remove incomplete key-value pairs at the end
    fixed = re.sub(r',\s*"[^"]*"?\s*:?\s*$', "", fixed)

    # Try to parse the fixed version
    try:
        json.loads(fixed)
        return fixed
    except json.JSONDecodeError:
        pass

    # If that didn't work, try closing it
    return try_contextual_closure_repair(fixed)


def simple_bracket_repair(json_str: str) -> str:
    """Fallback: Original simple bracket counting method"""
    open_braces = json_str.count("{")
    close_braces = json_str.count("}")
    open_brackets = json_str.count("[")
    close_brackets = json_str.count("]")

    missing_brackets = open_brackets - close_brackets
    missing_braces = open_braces - close_braces

    repaired = json_str
    repaired += "]" * max(0, missing_brackets)
    repaired += "}" * max(0, missing_braces)

    return repaired


def validate_and_repair_json(json_str: str) -> str:
    """Main function with comprehensive repair strategies"""
    json_str = json_str.strip()

    # Try parsing with repair library
    good_json = repair_json(json_str)
    if good_json:
        return good_json

    # Try comprehensive repair
    try:
        repaired = comprehensive_json_repair(json_str)

        # Validate the repair
        json.loads(repaired)
        print("✅ JSON successfully repaired!")
        return repaired

    except json.JSONDecodeError as repair_error:
        print(f"❌ Repair failed: {repair_error}")
        raise ValueError(
            f"Could not repair JSON. Original error: {repair_error.msg}, "
            + f"Repair error: {repair_error.msg}"
        ) from repair_error
