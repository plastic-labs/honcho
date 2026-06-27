"""Unit tests for graph memory schemas (no DB required)."""

import pytest
from pydantic import ValidationError

from src.schemas.graph_memory import (
    ContextCreate,
    EdgeCreate,
    EdgeListFilter,
    PinRequest,
    RecallRequest,
    ThreadBindingCreate,
)


class TestEdgeCreate:
    # 21-char nanoid-style test IDs
    SRC_ID = "abc123def456ghi789jk1"
    TGT_ID = "xyz789uvw456rst123ab2"

    def test_valid_edge(self):
        """Valid edge creation should succeed."""
        edge = EdgeCreate(
            collection_name="test-collection",
            source_obs_id=self.SRC_ID,
            target_obs_id=self.TGT_ID,
            edge_type="related",
        )
        assert edge.collection_name == "test-collection"
        assert edge.edge_type == "related"

    def test_invalid_edge_type(self):
        """Invalid edge type should be rejected."""
        with pytest.raises(ValidationError, match="edge_type"):
            EdgeCreate(
                collection_name="test",
                source_obs_id=self.SRC_ID,
                target_obs_id=self.TGT_ID,
                edge_type="invalid_type",
            )

    def test_self_edge(self):
        """Self-referencing edge should be allowed at schema level (DB constraint catches it)."""
        edge = EdgeCreate(
            collection_name="test",
            source_obs_id=self.SRC_ID,
            target_obs_id=self.SRC_ID,
            edge_type="related",
        )
        assert edge.source_obs_id == edge.target_obs_id

    def test_invalid_obs_id_length(self):
        """Observation IDs that aren't 21 chars should be rejected."""
        with pytest.raises(ValidationError, match="21 characters"):
            EdgeCreate(
                collection_name="test",
                source_obs_id="too-short",
                target_obs_id=self.TGT_ID,
                edge_type="related",
            )

    def test_all_edge_types(self):
        """All six edge types should be accepted."""
        for edge_type in ["related", "composes-with", "see-also", "refines", "supersedes", "contradicts"]:
            edge = EdgeCreate(
                collection_name="test",
                source_obs_id=self.SRC_ID,
                target_obs_id=self.TGT_ID,
                edge_type=edge_type,
            )
            assert edge.edge_type == edge_type

    def test_optional_metadata(self):
        """Optional metadata should default to empty dict."""
        edge = EdgeCreate(
            collection_name="test",
            source_obs_id=self.SRC_ID,
            target_obs_id=self.TGT_ID,
            edge_type="related",
        )
        assert edge.metadata == {}


class TestRecallRequest:
    def test_valid_recall(self):
        """Valid recall request should succeed."""
        req = RecallRequest(
            query="memory retrieval",
            collection_name="test-collection",
        )
        assert req.query == "memory retrieval"
        assert req.max_depth == 3  # default
        assert req.frontier_cap == 10  # default
        assert req.token_budget == 2000  # default
        assert req.include_pinned is True  # default

    def test_context_optional(self):
        """Context should be optional."""
        req = RecallRequest(query="test", collection_name="test")
        assert req.context is None

        req_with_context = RecallRequest(query="test", collection_name="test", context="my-context")
        assert req_with_context.context == "my-context"

    def test_max_depth_bounds(self):
        """Max depth should be between 1 and 10."""
        with pytest.raises(ValidationError):
            RecallRequest(query="test", collection_name="test", max_depth=0)
        with pytest.raises(ValidationError):
            RecallRequest(query="test", collection_name="test", max_depth=11)

    def test_frontier_cap_bounds(self):
        """Frontier cap should be between 1 and 100."""
        with pytest.raises(ValidationError):
            RecallRequest(query="test", collection_name="test", frontier_cap=0)
        with pytest.raises(ValidationError):
            RecallRequest(query="test", collection_name="test", frontier_cap=101)

    def test_token_budget_bounds(self):
        """Token budget should be between 100 and 10000."""
        with pytest.raises(ValidationError):
            RecallRequest(query="test", collection_name="test", token_budget=50)
        with pytest.raises(ValidationError):
            RecallRequest(query="test", collection_name="test", token_budget=20000)


class TestContextCreate:
    def test_valid_context_name(self):
        """Valid context names should succeed."""
        for name in ["my-context", "my_context", "context123", "a", "x" * 64]:
            ctx = ContextCreate(context_name=name)
            assert ctx.context_name == name

    def test_invalid_context_name(self):
        """Invalid context names should be rejected."""
        for name in ["", "has spaces", "has.dots", "has/slashes", "x" * 65]:
            with pytest.raises(ValidationError):
                ContextCreate(context_name=name)


class TestThreadBindingCreate:
    def test_valid_thread_id(self):
        """Valid Slack thread IDs should succeed."""
        binding = ThreadBindingCreate(
            thread_id="1234567890.123456",
            context_name="my-context",
        )
        assert binding.thread_id == "1234567890.123456"

    def test_invalid_thread_id(self):
        """Invalid thread IDs should be rejected."""
        with pytest.raises(ValidationError):
            ThreadBindingCreate(thread_id="not-a-thread-id", context_name="test")


class TestPinRequest:
    def test_no_cadence(self):
        """Null cadence should be accepted (default)."""
        pin = PinRequest()
        assert pin.verify_cadence_days is None

    def test_valid_cadence(self):
        """Valid cadence values should be accepted."""
        for days in [1, 7, 30, 365, 3650]:
            pin = PinRequest(verify_cadence_days=days)
            assert pin.verify_cadence_days == days

    def test_negative_cadence(self):
        """Negative cadence should be rejected."""
        with pytest.raises(ValidationError):
            PinRequest(verify_cadence_days=-1)

    def test_zero_cadence(self):
        """Zero cadence should be rejected (must be >= 1)."""
        with pytest.raises(ValidationError):
            PinRequest(verify_cadence_days=0)

    def test_excessive_cadence(self):
        """Cadence over 3650 should be rejected."""
        with pytest.raises(ValidationError):
            PinRequest(verify_cadence_days=3651)


class TestEdgeListFilter:
    def test_empty_filter(self):
        """Empty filter should accept all fields as None."""
        filt = EdgeListFilter()
        assert filt.source_obs_id is None
        assert filt.target_obs_id is None
        assert filt.edge_type is None
        assert filt.collection_name is None

    def test_partial_filter(self):
        """Partial filter should work."""
        filt = EdgeListFilter(source_obs_id="abc123def456ghi789jk1")
        assert filt.source_obs_id == "abc123def456ghi789jk1"
        assert filt.target_obs_id is None
