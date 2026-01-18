"""Example: Querying Reasoning Artifacts

This example demonstrates how to query hypotheses, predictions, traces, and inductions
generated during reasoning dreams using the Honcho SDK.

Reasoning artifacts are created exclusively by reasoning agents during dream processing
and cannot be created or modified via the API. This provides read-only access for
transparency and debugging.
"""

from honcho import Honcho

# Initialize the client
client = Honcho(workspace_id="my-workspace")

# =============================================================================
# Hypotheses - Explanatory theories about observed patterns
# =============================================================================

print("=" * 80)
print("HYPOTHESES")
print("=" * 80)

# List all active hypotheses for a peer
print("\n1. List active hypotheses for a peer:")
hypotheses = client.get_hypotheses(
    observer="user_123",
    observed="user_123",
    status="active"
)
print(f"Found {len(hypotheses)} active hypotheses")

# Get a specific hypothesis
if hypotheses:
    hypothesis_id = hypotheses[0].id
    print(f"\n2. Get hypothesis details:")
    hypothesis = client.get_hypothesis(hypothesis_id)
    print(f"Content: {hypothesis.content}")
    print(f"Confidence: {hypothesis.confidence}")
    print(f"Tier: {hypothesis.tier}")
    print(f"Status: {hypothesis.status}")

    # Get predictions for this hypothesis
    print(f"\n3. Get predictions for hypothesis:")
    predictions = client.get_hypothesis_predictions(
        hypothesis_id,
        status="unfalsified"
    )
    print(f"Found {len(predictions)} unfalsified predictions")

    # Get hypothesis genealogy (evolution tree)
    print(f"\n4. Get hypothesis genealogy:")
    genealogy = client.get_hypothesis_genealogy(hypothesis_id)
    print(f"Parents (superseded): {len(genealogy['parents'])}")
    print(f"Children (superseded by): {len(genealogy['children'])}")
    if genealogy['reasoning_metadata']:
        print(f"Evolution reason: {genealogy['reasoning_metadata'].get('reason', 'N/A')}")


# =============================================================================
# Predictions - Testable claims derived from hypotheses
# =============================================================================

print("\n" + "=" * 80)
print("PREDICTIONS")
print("=" * 80)

# List all predictions
print("\n5. List predictions by status:")
predictions = client.get_predictions(status="unfalsified")
print(f"Found {len(predictions)} unfalsified predictions")

# Get a specific prediction
if predictions:
    prediction_id = predictions[0].id
    print(f"\n6. Get prediction details:")
    prediction = client.get_prediction(prediction_id)
    print(f"Content: {prediction.content}")
    print(f"Status: {prediction.status}")
    print(f"Is Blind: {prediction.is_blind}")
    print(f"Hypothesis ID: {prediction.hypothesis_id}")

    # Get traces for this prediction
    print(f"\n7. Get falsification traces for prediction:")
    traces = client.get_prediction_traces(prediction_id)
    print(f"Found {len(traces)} traces")

# Semantic search for similar predictions
print("\n8. Search for similar predictions:")
similar = client.search_predictions(
    "prefers dark mode over light mode",
    hypothesis_id=hypothesis_id if hypotheses else None
)
print(f"Found {len(similar)} semantically similar predictions")
for pred in similar[:3]:  # Show top 3
    print(f"  - {pred.content} (confidence: {pred.confidence})")


# =============================================================================
# Falsification Traces - Records of falsification attempts
# =============================================================================

print("\n" + "=" * 80)
print("FALSIFICATION TRACES")
print("=" * 80)

# List all traces
print("\n9. List falsification traces:")
traces = client.get_traces(final_status="unfalsified")
print(f"Found {len(traces)} unfalsified traces")

# Get a specific trace
if traces:
    trace_id = traces[0].id
    print(f"\n10. Get trace details:")
    trace = client.get_trace(trace_id)
    print(f"Prediction ID: {trace.prediction_id}")
    print(f"Final Status: {trace.final_status}")
    print(f"Search Count: {trace.search_count}")
    print(f"Search Efficiency: {trace.search_efficiency_score}")
    print(f"Search Queries Executed:")
    for query in trace.search_queries or []:
        print(f"  - {query}")
    if trace.contradicting_premise_ids:
        print(f"Contradicting Premises Found: {len(trace.contradicting_premise_ids)}")


# =============================================================================
# Inductions - Patterns extracted from unfalsified predictions
# =============================================================================

print("\n" + "=" * 80)
print("INDUCTIONS")
print("=" * 80)

# List all inductions
print("\n11. List inductions by confidence:")
inductions = client.get_inductions(
    observer="user_123",
    observed="user_123",
    confidence="high"
)
print(f"Found {len(inductions)} high-confidence inductions")

# Get a specific induction
if inductions:
    induction_id = inductions[0].id
    print(f"\n12. Get induction details:")
    induction = client.get_induction(induction_id)
    print(f"Content: {induction.content}")
    print(f"Pattern Type: {induction.pattern_type}")
    print(f"Confidence: {induction.confidence}")
    print(f"Stability Score: {induction.stability_score}")

    # Get sources for this induction
    print(f"\n13. Get induction sources:")
    sources = client.get_induction_sources(induction_id)
    print(f"Based on {len(sources['source_predictions'])} predictions")
    print(f"From {len(sources['source_premises'])} observations")
    print(f"\nSource Predictions:")
    for pred in sources['source_predictions'][:3]:  # Show top 3
        print(f"  - {pred.content}")


# =============================================================================
# Filtering by Pattern Type
# =============================================================================

print("\n" + "=" * 80)
print("PATTERN TYPE FILTERING")
print("=" * 80)

# Get inductions by pattern type
pattern_types = ["preference", "behavior", "personality", "tendency"]
for pattern_type in pattern_types:
    inductions = client.get_inductions(
        observer="user_123",
        observed="user_123",
        pattern_type=pattern_type
    )
    if inductions:
        print(f"\n{pattern_type.capitalize()} patterns: {len(inductions)}")
        for ind in inductions[:2]:  # Show top 2
            print(f"  - {ind.content}")


# =============================================================================
# Cross-Referencing
# =============================================================================

print("\n" + "=" * 80)
print("CROSS-REFERENCING")
print("=" * 80)

# Example: Trace a pattern back to its origins
print("\n14. Trace pattern provenance:")
if inductions:
    induction = inductions[0]
    print(f"\nPattern: {induction.content}")

    # Get sources
    sources = client.get_induction_sources(induction.id)

    # Show predictions
    print(f"\n└─ Based on {len(sources['source_predictions'])} predictions:")
    for pred in sources['source_predictions'][:2]:
        print(f"   ├─ {pred.content}")

        # Get hypothesis for each prediction
        hypothesis = client.get_hypothesis(pred.hypothesis_id)
        print(f"   │  └─ From hypothesis: {hypothesis.content}")

        # Get traces for each prediction
        traces = client.get_prediction_traces(pred.id)
        print(f"   │     └─ Tested {len(traces)} times, all unfalsified")

print("\n" + "=" * 80)
print("PEER CONVENIENCE METHODS")
print("=" * 80)

# Peer objects have convenience methods that automatically scope to that peer
print("\n15. Using Peer convenience methods:")

# Get a peer object
peer = client.peer("user_123")

# Get hypotheses about self using Peer method
print("\nHypotheses about self (via peer.get_hypotheses()):")
peer_hypotheses = peer.get_hypotheses(status="active")
print(f"Found {len(peer_hypotheses.get('items', []))} active hypotheses")

# Get hypotheses about another peer
print("\nHypotheses about another peer:")
peer_hypotheses_target = peer.get_hypotheses(target="user_456", status="active")
print(f"Found {len(peer_hypotheses_target.get('items', []))} hypotheses about user_456")

# Get induction patterns
print("\nInduction patterns (via peer.get_inductions()):")
peer_inductions = peer.get_inductions(confidence="high")
print(f"Found {len(peer_inductions.get('items', []))} high-confidence patterns")

# Get behavioral patterns about another peer
print("\nBehavioral patterns about another peer:")
peer_behavioral = peer.get_inductions(
    target="user_456",
    pattern_type="behavioral"
)
print(f"Found {len(peer_behavioral.get('items', []))} behavioral patterns")


print("\n" + "=" * 80)
print("COMPLETE")
print("=" * 80)
