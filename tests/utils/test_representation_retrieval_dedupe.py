from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

from src.utils.representation import Representation


def _doc(doc_id: str, level: str, content: str):
    return SimpleNamespace(
        id=doc_id,
        level=level,
        content=content,
        created_at=dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.UTC),
        internal_metadata={"message_ids": [1]},
        session_name="s1",
        source_ids=None,
    )


def test_representation_from_documents_dedupes_exact_normalized_content():
    representation = Representation.from_documents(
        [
            _doc("doc-a", "explicit", "User prefers concise responses."),
            _doc("doc-b", "explicit", " user   prefers CONCISE responses. "),
            _doc("doc-c", "deductive", "User prefers concise responses."),
        ]
    )

    assert [obs.id for obs in representation.explicit] == ["doc-a"]
    # Same normalized content in a different observation level is retained.
    assert [obs.id for obs in representation.deductive] == ["doc-c"]
