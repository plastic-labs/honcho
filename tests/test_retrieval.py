"""Unit tests for hybrid retrieval utilities (MMR, lexical rerank, RRF, thresholds)."""

from src.utils.search import (
    _apply_score_threshold,
    lexical_rerank,
    maximal_marginal_relevance,
    reciprocal_rank_fusion,
)


class FakeItem:
    """Fake item with content and id for testing lexical rerank."""

    id: str
    content: str

    def __init__(self, item_id: str, content: str):
        self.id = item_id
        self.content = content


class TestReciprocalRankFusion:
    def test_empty_lists(self):
        assert reciprocal_rank_fusion(limit=10) == []

    def test_single_list_passthrough(self):
        items = ["a", "b", "c"]
        result = reciprocal_rank_fusion(items, limit=10)
        assert result == items

    def test_two_lists_fusion(self):
        list1 = ["a", "b", "c"]
        list2 = ["b", "c", "d"]
        result = reciprocal_rank_fusion(list1, list2, k=60, limit=10)
        # b appears in both lists (rank 2 and 1), c appears in both (rank 3 and 2)
        # a appears only in list1 (rank 1), d appears only in list2 (rank 3)
        # Expected order: b > c > a > d
        assert result == ["b", "c", "a", "d"]

    def test_limit_respected(self):
        list1 = ["a", "b", "c", "d"]
        list2 = ["e", "f", "g", "h"]
        result = reciprocal_rank_fusion(list1, list2, k=60, limit=2)
        assert len(result) == 2


class TestApplyScoreThreshold:
    def test_no_threshold_returns_all(self):
        ranked = ["a", "b", "c"]
        result = _apply_score_threshold(
            ranked, ["a", "b", "c"], threshold=0.0, k=60
        )
        assert result == ranked

    def test_high_threshold_filters_all(self):
        ranked = ["a", "b", "c"]
        result = _apply_score_threshold(
            ranked, ["a"], ["b"], threshold=10.0, k=60
        )
        assert result == []

    def test_threshold_keeps_high_scorers(self):
        # a is rank 1 in both lists -> score = 1/61 + 1/61 ≈ 0.0328
        # b is rank 2 in one list only -> score = 1/62 ≈ 0.0161
        ranked = ["a", "b"]
        result = _apply_score_threshold(
            ranked, ["a", "b"], ["a", "b"], threshold=0.02, k=60
        )
        assert "a" in result


class TestLexicalRerank:
    def test_empty_terms_returns_original(self):
        items = [FakeItem("1", "hello")]
        assert lexical_rerank(items, "") == items

    def test_exact_match_boost(self):
        items = [
            FakeItem("1", "the quick brown fox"),
            FakeItem("2", "quick brown fox jumps over"),
            FakeItem("3", "exact phrase here"),
        ]
        result = lexical_rerank(items, "exact phrase")
        # Item 3 contains the exact phrase and should win
        assert result[0].id == "3"

    def test_term_overlap_ordering(self):
        items = [
            FakeItem("1", "apple"),
            FakeItem("2", "apple banana cherry date"),
            FakeItem("3", "cherry date"),
        ]
        result = lexical_rerank(items, "apple banana")
        # Item 2 has both terms plus exact phrase -> first
        assert result[0].id == "2"
        assert result[1].id == "1"


class TestMaximalMarginalRelevance:
    def test_empty_documents(self):
        assert maximal_marginal_relevance([], [1.0], {}, 0.5, 5) == []

    def test_lambda_one_returns_original_order(self):
        docs = [
            FakeItem("1", "a"),
            FakeItem("2", "b"),
        ]
        embeddings = {"1": [1.0, 0.0], "2": [0.0, 1.0]}
        query_emb = [1.0, 0.0]
        result = maximal_marginal_relevance(
            docs, query_emb, embeddings, lambda_param=1.0, top_k=2
        )
        assert result[0].id == "1"
        assert result[1].id == "2"

    def test_diversity_changes_order(self):
        # Three docs: two very similar, one different
        docs = [
            FakeItem("1", "a"),
            FakeItem("2", "a similar"),
            FakeItem("3", "completely different"),
        ]
        embeddings = {
            "1": [1.0, 0.0],
            "2": [0.99, 0.01],  # very close to 1
            "3": [0.0, 1.0],  # far from 1 and 2
        }
        query_emb = [1.0, 0.0]
        result = maximal_marginal_relevance(
            docs, query_emb, embeddings, lambda_param=0.3, top_k=2
        )
        # With low lambda (diversity priority), we should get 1 first (most relevant)
        # then 3 (most diverse from 1) instead of 2 (too similar to 1)
        ids = [d.id for d in result]
        assert ids[0] == "1"
        assert ids[1] == "3"

    def test_missing_embeddings_skipped(self):
        docs = [
            FakeItem("1", "a"),
            FakeItem("2", "b"),
        ]
        embeddings = {"1": [1.0, 0.0]}  # missing embedding for 2
        query_emb = [1.0, 0.0]
        result = maximal_marginal_relevance(
            docs, query_emb, embeddings, lambda_param=0.5, top_k=2
        )
        assert result[0].id == "1"
        # Item 2 has no embedding, so it can't be selected by MMR
        # but should be appended to fill the quota
        assert len(result) == 2
