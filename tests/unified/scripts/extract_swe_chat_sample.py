#!/usr/bin/env python3
"""
Extract a balanced sample of SWE-chat sessions for the meta-conversation test fixture.

Run once to refresh the fixture committed at:
    tests/unified/test_cases/data/swe_chat_meta_conversation_sample.json

Inputs (not committed; download from HuggingFace SALT-NLP/SWE-chat):
    SWE_CHAT_DIR / sessions.parquet
    SWE_CHAT_DIR / conversations.parquet

Default location:
    ~/claude-projects/test-case-development/swe-chat-dataset/

Selection criteria (documented in the fixture for reproducibility):
- Balanced across user_persona: 3 Expert Nitpicker, 3 Vague Requester, 3 Mind Changer, 1 Other
- Sessions must have prompt_count >= MIN_TURNS (drops trivially short sessions)
- Each session truncated to first TURNS_PER_SESSION conversational turns
- Random sample with fixed seed for reproducibility

Why this dataset: real-world coding-agent sessions provide subject-matter
diversity, ensuring tests don't accidentally couple to any one topic. The
prompt_pushback / user_persona annotations are particularly useful for
selecting input distributions that exercise the deriver's meta-conversational
extraction behavior.

Dependencies: pandas, pyarrow. Run with:
    uv run --with pandas --with pyarrow python tests/unified/scripts/extract_swe_chat_sample.py
"""

import json
import os
from pathlib import Path

import pandas as pd

SWE_CHAT_DIR = Path(
    os.environ.get(
        "SWE_CHAT_DIR",
        str(Path.home() / "claude-projects/test-case-development/swe-chat-dataset"),
    )
)

OUT_PATH = (
    Path(__file__).parent.parent
    / "test_cases"
    / "data"
    / "swe_chat_sample_persona_balanced.json"
)

PERSONA_COUNTS = {
    "Expert Nitpicker": 3,
    "Vague Requester": 3,
    "Mind Changer": 3,
    "Other": 1,
}
TURNS_PER_SESSION = 20
MIN_TURNS = 8
RANDOM_SEED = 42


def main() -> None:
    if not SWE_CHAT_DIR.is_dir():
        raise FileNotFoundError(
            f"SWE-chat dataset directory not found: {SWE_CHAT_DIR}\n"
            "Set SWE_CHAT_DIR env var or download from HuggingFace SALT-NLP/SWE-chat."
        )

    print(f"Loading sessions.parquet from {SWE_CHAT_DIR}...")
    sessions = pd.read_parquet(SWE_CHAT_DIR / "sessions.parquet")
    print(f"  total sessions: {len(sessions)}")

    sessions = sessions[sessions["user_persona"].notna()]
    sessions = sessions[sessions["prompt_count"] >= MIN_TURNS]
    print(f"  after persona/length filter: {len(sessions)}")

    selected_session_ids: list[str] = []
    for persona, n in PERSONA_COUNTS.items():
        cohort = sessions[sessions["user_persona"] == persona]
        if len(cohort) < n:
            print(f"  WARNING: only {len(cohort)} sessions for persona={persona!r}, need {n}")
            sampled = cohort
        else:
            sampled = cohort.sample(n=n, random_state=RANDOM_SEED)
        selected_session_ids.extend(sampled["session_id"].tolist())
        print(f"  persona={persona}: selected {len(sampled)}")

    print(f"Selected {len(selected_session_ids)} sessions total")

    print("Loading conversations.parquet (this may take a minute on first read)...")
    conversations = pd.read_parquet(SWE_CHAT_DIR / "conversations.parquet")
    print(f"  total conversation rows: {len(conversations)}")

    conv = conversations[
        (conversations["session_id"].isin(selected_session_ids))
        & (conversations["is_conversational"])
    ].copy()
    conv = conv.sort_values(["session_id", "conversation_turn_number"])
    print(f"  conversational rows for selected sessions: {len(conv)}")

    fixture_sessions = []
    for sid in selected_session_ids:
        session_meta = sessions[sessions["session_id"] == sid].iloc[0]
        session_conv = conv[conv["session_id"] == sid].head(TURNS_PER_SESSION)
        if len(session_conv) < MIN_TURNS:
            print(f"  Skipping {sid}: only {len(session_conv)} conversational turns")
            continue
        messages = []
        for _, row in session_conv.iterrows():
            ts = row["timestamp"]
            ts_iso = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            messages.append(
                {
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": ts_iso,
                }
            )
        fixture_sessions.append(
            {
                "swe_session_id": sid,
                "user_persona": session_meta["user_persona"],
                "agent": session_meta["agent"],
                "turn_count": len(messages),
                "messages": messages,
            }
        )

    fixture = {
        "dataset": "SALT-NLP/SWE-chat",
        "license": "ODC-BY",
        "paper": "https://arxiv.org/abs/2604.20779",
        "extracted_at": pd.Timestamp.utcnow().isoformat(),
        "selection_criteria": {
            "persona_balance": PERSONA_COUNTS,
            "turns_per_session": TURNS_PER_SESSION,
            "min_turns": MIN_TURNS,
            "random_seed": RANDOM_SEED,
            "filter": "is_conversational == True, user_persona not null, prompt_count >= min_turns",
        },
        "sessions": fixture_sessions,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"\nWrote {len(fixture_sessions)} sessions to {OUT_PATH}")
    print(f"Fixture size: {OUT_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
