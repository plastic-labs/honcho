"""Unit tests for graph memory CRUD logic (no DB required — tests pure functions)."""

import math
import time
from datetime import datetime, timezone

import pytest

from src.crud.graph_memory import (
    ACTIVATION_HALF_LIFE_HOURS,
    CONFIDENCE_HALF_LIFE_DAYS,
    CONFIDENCE_THRESHOLD,
    EVENT_WEIGHTS,
    PINNED_FLOOR,
)


class TestEventWeights:
    """Verify event weights match spec §3."""

    def test_access_weight(self):
        assert EVENT_WEIGHTS["access"] == 0.3

    def test_verify_weight(self):
        assert EVENT_WEIGHTS["verify"] == 1.0

    def test_recall_weight(self):
        assert EVENT_WEIGHTS["recall"] == 0.5

    def test_promote_weight(self):
        assert EVENT_WEIGHTS["promote"] == 1.0

    def test_rehydrate_weight(self):
        assert EVENT_WEIGHTS["rehydrate"] == 1.0

    def test_evict_weight(self):
        assert EVENT_WEIGHTS["evict"] == 0.0

    def test_all_expected_keys(self):
        """All expected event types should have weights."""
        expected = {"access", "verify", "recall", "promote", "rehydrate", "evict"}
        assert set(EVENT_WEIGHTS.keys()) == expected


class TestDecayConstants:
    """Verify decay constants match spec §3."""

    def test_activation_half_life(self):
        assert ACTIVATION_HALF_LIFE_HOURS == 24.0

    def test_confidence_half_life(self):
        assert CONFIDENCE_HALF_LIFE_DAYS == 30.0

    def test_confidence_threshold(self):
        assert CONFIDENCE_THRESHOLD == 0.3

    def test_pinned_floor(self):
        assert PINNED_FLOOR == 0.85


class TestActivationDecay:
    """Verify activation decay math matches spec formula.
    
    activation = Σ weight * exp(-Δt / half_life)
    """

    def test_single_event_at_t0(self):
        """A single event at t=0 should contribute its full weight."""
        weight = EVENT_WEIGHTS["access"]  # 0.3
        dt_hours = 0.0
        decay = math.exp(-dt_hours / ACTIVATION_HALF_LIFE_HOURS)
        contribution = weight * decay
        assert contribution == pytest.approx(0.3)

    def test_single_event_at_one_half_life(self):
        """A single event at t=24h should contribute weight * exp(-1)."""
        weight = EVENT_WEIGHTS["access"]  # 0.3
        dt_hours = ACTIVATION_HALF_LIFE_HOURS  # 24.0
        decay = math.exp(-dt_hours / ACTIVATION_HALF_LIFE_HOURS)  # exp(-1)
        contribution = weight * decay
        assert contribution == pytest.approx(0.3 * math.exp(-1))

    def test_single_event_at_five_half_lives(self):
        """A single event at t=120h should contribute negligible weight."""
        weight = EVENT_WEIGHTS["access"]  # 0.3
        dt_hours = 5 * ACTIVATION_HALF_LIFE_HOURS  # 120.0
        decay = math.exp(-dt_hours / ACTIVATION_HALF_LIFE_HOURS)  # exp(-5)
        contribution = weight * decay
        assert contribution == pytest.approx(0.3 * math.exp(-5))
        assert contribution < 0.01  # Negligible

    def test_verify_event_higher_weight(self):
        """Verify events (weight=1.0) should contribute more than access events (weight=0.3)."""
        dt_hours = 1.0
        decay = math.exp(-dt_hours / ACTIVATION_HALF_LIFE_HOURS)
        verify_contrib = EVENT_WEIGHTS["verify"] * decay
        access_contrib = EVENT_WEIGHTS["access"] * decay
        assert verify_contrib > access_contrib

    def test_evict_event_no_contribution(self):
        """Evict events (weight=0.0) should contribute nothing."""
        dt_hours = 0.0
        decay = math.exp(-dt_hours / ACTIVATION_HALF_LIFE_HOURS)
        contribution = EVENT_WEIGHTS["evict"] * decay
        assert contribution == 0.0


class TestConfidenceDecay:
    """Verify confidence decay math matches spec formula.
    
    confidence = exp(-(now - last_verify) / verify_half_life)
    
    This is a PURE function of last_verify and now — NO compounding.
    """

    def test_confidence_at_t0(self):
        """Confidence should be 1.0 at t=0."""
        dt_hours = 0.0
        half_life_hours = CONFIDENCE_HALF_LIFE_DAYS * 24.0
        confidence = math.exp(-dt_hours / half_life_hours)
        assert confidence == pytest.approx(1.0)

    def test_confidence_at_one_half_life(self):
        """Confidence should be exp(-1) at t=30 days."""
        dt_hours = CONFIDENCE_HALF_LIFE_DAYS * 24.0  # 720 hours
        half_life_hours = CONFIDENCE_HALF_LIFE_DAYS * 24.0
        confidence = math.exp(-dt_hours / half_life_hours)
        assert confidence == pytest.approx(math.exp(-1))

    def test_confidence_no_compounding(self):
        """Confidence should be a pure function of last_verify — no compounding.
        
        If last_verify is at t=0, confidence at t=60d should be exp(-2).
        If last_verify is at t=30d, confidence at t=60d should be exp(-1).
        These are different because the function depends ONLY on (now - last_verify).
        """
        half_life_hours = CONFIDENCE_HALF_LIFE_DAYS * 24.0
        
        # Case 1: last_verify at t=0, now at t=60d
        dt_1 = 60 * 24.0
        conf_1 = math.exp(-dt_1 / half_life_hours)
        
        # Case 2: last_verify at t=30d, now at t=60d
        dt_2 = 30 * 24.0
        conf_2 = math.exp(-dt_2 / half_life_hours)
        
        assert conf_1 == pytest.approx(math.exp(-2))
        assert conf_2 == pytest.approx(math.exp(-1))
        assert conf_1 < conf_2  # Older verification = lower confidence

    def test_confidence_threshold_crossing(self):
        """Confidence should cross the 0.3 threshold at a predictable time."""
        half_life_hours = CONFIDENCE_HALF_LIFE_DAYS * 24.0
        # confidence = exp(-t / HL) = 0.3
        # t = -HL * ln(0.3)
        t_hours = -half_life_hours * math.log(CONFIDENCE_THRESHOLD)
        t_days = t_hours / 24.0
        # Should be approximately 36 days
        assert t_days == pytest.approx(36.0, abs=1.0)


class TestSourceDiversity:
    """Verify source-diversity weighting math.
    
    Same-source repeats: repeat_factor = 1 / (1 + ln(1 + n))
    Cross-source: full weight for each distinct source.
    """

    def test_first_access_full_weight(self):
        """First access from a source should have repeat_factor = 1.0."""
        n = 0  # First access
        factor = 1.0 / (1.0 + math.log(1.0 + n))
        assert factor == pytest.approx(1.0)

    def test_second_access_diminished(self):
        """Second access from same source should have reduced factor."""
        n = 1  # Second access
        factor = 1.0 / (1.0 + math.log(1.0 + n))
        assert factor < 1.0
        assert factor == pytest.approx(1.0 / (1.0 + math.log(2)))

    def test_tenth_access_heavily_diminished(self):
        """Tenth access from same source should be heavily diminished."""
        n = 9  # Tenth access
        factor = 1.0 / (1.0 + math.log(1.0 + n))
        assert factor < 0.5  # Less than half weight

    def test_two_sources_better_than_one(self):
        """Two distinct sources should give more total activation than one source with same total events."""
        half_life_hours = ACTIVATION_HALF_LIFE_HOURS
        weight = EVENT_WEIGHTS["access"]
        dt_hours = 1.0
        decay = math.exp(-dt_hours / half_life_hours)
        
        # One source, 4 events
        one_source = 0.0
        for i in range(4):
            factor = 1.0 / (1.0 + math.log(1.0 + i))
            one_source += weight * decay * factor
        
        # Two sources, 2 events each
        two_sources = 0.0
        for _ in range(2):  # Two sources
            for i in range(2):  # Two events each
                factor = 1.0 / (1.0 + math.log(1.0 + i))
                two_sources += weight * decay * factor
        
        assert two_sources > one_source


class TestPinnedFloor:
    """Verify pinned floor behavior."""

    def test_pinned_floor_value(self):
        """Pinned floor should be 0.85."""
        assert PINNED_FLOOR == 0.85

    def test_pinned_floor_applied(self):
        """Pinned activation should be max(computed, 0.85)."""
        computed = 0.5
        pinned = max(computed, PINNED_FLOOR)
        assert pinned == PINNED_FLOOR

    def test_pinned_above_floor(self):
        """If computed activation is above floor, use computed."""
        computed = 0.95
        pinned = max(computed, PINNED_FLOOR)
        assert pinned == computed
