"""Tests for observation SDK methods."""

import pytest

from sdks.python.src.honcho.client import Honcho
from sdks.python.src.honcho.conclusions import (
    Conclusion,
    ConclusionCreateParams,
    ConclusionsView,
    WorkspaceConclusions,
)
from sdks.python.src.honcho.http import NotFoundError


@pytest.mark.asyncio
async def test_observation_create_single(
    client_fixture: tuple[Honcho, str],
):
    """
    Tests creating a single observation via the SDK.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        observer = await honcho_client.aio.peer(id="test-obs-create-single-observer")
        target = await honcho_client.aio.peer(id="test-obs-create-single-target")
        session = await honcho_client.aio.session(id="test-obs-create-single-session")

        # Ensure session and both peers exist by adding messages from both
        await session.aio.add_messages(
            [
                observer.message("Hello from observer"),
                target.message("Hello from target"),
            ]
        )

        # Get observation scope for observer -> target
        obs_scope = observer.conclusions_of(target)
        assert isinstance(obs_scope, ConclusionsView)

        # Create a single observation
        created = await obs_scope.aio.create(
            [
                ConclusionCreateParams(
                    content="User prefers dark mode",
                    session_id=session.id,
                )
            ]
        )

        assert len(created) == 1
        assert isinstance(created[0], Conclusion)
        assert created[0].content == "User prefers dark mode"
        assert created[0].observer_id == observer.id
        assert created[0].observed_id == target.id
        assert created[0].session_id == session.id
        assert created[0].id  # Has an ID
    else:
        observer = honcho_client.peer(id="test-obs-create-single-observer")
        target = honcho_client.peer(id="test-obs-create-single-target")
        session = honcho_client.session(id="test-obs-create-single-session")

        # Ensure session and both peers exist by adding messages from both
        session.add_messages(
            [
                observer.message("Hello from observer"),
                target.message("Hello from target"),
            ]
        )

        # Get observation scope for observer -> target
        obs_scope = observer.conclusions_of(target)
        assert isinstance(obs_scope, ConclusionsView)

        # Create a single observation
        created = obs_scope.create(
            [
                ConclusionCreateParams(
                    content="User prefers dark mode",
                    session_id=session.id,
                )
            ]
        )

        assert len(created) == 1
        assert isinstance(created[0], Conclusion)
        assert created[0].content == "User prefers dark mode"
        assert created[0].observer_id == observer.id
        assert created[0].observed_id == target.id
        assert created[0].session_id == session.id
        assert created[0].id  # Has an ID


@pytest.mark.asyncio
async def test_observation_create_batch(
    client_fixture: tuple[Honcho, str],
):
    """
    Tests creating multiple observations in a batch via the SDK.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        observer = await honcho_client.aio.peer(id="test-obs-create-batch-observer")
        target = await honcho_client.aio.peer(id="test-obs-create-batch-target")
        session = await honcho_client.aio.session(id="test-obs-create-batch-session")

        # Ensure session and both peers exist
        await session.aio.add_messages(
            [
                observer.message("Hello from observer"),
                target.message("Hello from target"),
            ]
        )

        # Get observation scope
        obs_scope = observer.conclusions_of(target)

        # Create multiple observations
        created = await obs_scope.aio.create(
            [
                ConclusionCreateParams(
                    content="User prefers dark mode",
                    session_id=session.id,
                ),
                ConclusionCreateParams(
                    content="User works late at night",
                    session_id=session.id,
                ),
                ConclusionCreateParams(
                    content="User enjoys programming",
                    session_id=session.id,
                ),
            ]
        )

        assert len(created) == 3
        contents = {obs.content for obs in created}
        assert "User prefers dark mode" in contents
        assert "User works late at night" in contents
        assert "User enjoys programming" in contents

        # All observations have correct observer/observed
        for obs in created:
            assert obs.observer_id == observer.id
            assert obs.observed_id == target.id
            assert obs.session_id == session.id
    else:
        observer = honcho_client.peer(id="test-obs-create-batch-observer")
        target = honcho_client.peer(id="test-obs-create-batch-target")
        session = honcho_client.session(id="test-obs-create-batch-session")

        # Ensure session and both peers exist
        session.add_messages(
            [
                observer.message("Hello from observer"),
                target.message("Hello from target"),
            ]
        )

        # Get observation scope
        obs_scope = observer.conclusions_of(target)

        # Create multiple observations
        created = obs_scope.create(
            [
                {"content": "User prefers dark mode", "session_id": session.id},
                {"content": "User works late at night", "session_id": session.id},
                {"content": "User enjoys programming", "session_id": session.id},
            ]
        )

        assert len(created) == 3
        contents = {obs.content for obs in created}
        assert "User prefers dark mode" in contents
        assert "User works late at night" in contents
        assert "User enjoys programming" in contents

        # All observations have correct observer/observed
        for obs in created:
            assert obs.observer_id == observer.id
            assert obs.observed_id == target.id
            assert obs.session_id == session.id


@pytest.mark.asyncio
async def test_observation_create_then_list(
    client_fixture: tuple[Honcho, str],
):
    """
    Tests that created observations can be listed.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        observer = await honcho_client.aio.peer(id="test-obs-create-list-observer")
        target = await honcho_client.aio.peer(id="test-obs-create-list-target")
        session = await honcho_client.aio.session(id="test-obs-create-list-session")

        # Ensure session and both peers exist
        await session.aio.add_messages(
            [
                observer.message("Hello from observer"),
                target.message("Hello from target"),
            ]
        )

        # Get observation scope
        obs_scope = observer.conclusions_of(target)

        # Create observations
        created = await obs_scope.aio.create(
            [
                {
                    "content": "Unique observation for list test",
                    "session_id": session.id,
                },
            ]
        )

        # List observations
        listed = await obs_scope.aio.list()

        # The created observation should be in the list
        listed_ids = {obs.id for obs in listed.items}
        assert created[0].id in listed_ids
    else:
        observer = honcho_client.peer(id="test-obs-create-list-observer")
        target = honcho_client.peer(id="test-obs-create-list-target")
        session = honcho_client.session(id="test-obs-create-list-session")

        # Ensure session and both peers exist
        session.add_messages(
            [
                observer.message("Hello from observer"),
                target.message("Hello from target"),
            ]
        )

        # Get observation scope
        obs_scope = observer.conclusions_of(target)

        # Create observations
        created = obs_scope.create(
            [
                {
                    "content": "Unique observation for list test",
                    "session_id": session.id,
                },
            ]
        )

        # List observations
        listed = obs_scope.list()

        # The created observation should be in the list
        listed_ids = {obs.id for obs in listed}
        assert created[0].id in listed_ids


@pytest.mark.asyncio
async def test_observation_create_then_query(
    client_fixture: tuple[Honcho, str],
):
    """
    Tests that created observations can be queried semantically.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        observer = await honcho_client.aio.peer(id="test-obs-create-query-observer")
        target = await honcho_client.aio.peer(id="test-obs-create-query-target")
        session = await honcho_client.aio.session(id="test-obs-create-query-session")

        # Ensure session and both peers exist
        await session.aio.add_messages(
            [
                observer.message("Hello from observer"),
                target.message("Hello from target"),
            ]
        )

        # Get observation scope
        obs_scope = observer.conclusions_of(target)

        # Create observation with specific content
        await obs_scope.aio.create(
            [
                {
                    "content": "User loves Italian cuisine especially pasta and pizza",
                    "session_id": session.id,
                },
            ]
        )

        # Query for food-related observations
        results = await obs_scope.aio.query("food preferences")

        assert len(results) >= 1
        # At least one result should mention Italian food
        contents = " ".join(obs.content for obs in results)
        assert "Italian" in contents or "pasta" in contents or "pizza" in contents
    else:
        observer = honcho_client.peer(id="test-obs-create-query-observer")
        target = honcho_client.peer(id="test-obs-create-query-target")
        session = honcho_client.session(id="test-obs-create-query-session")

        # Ensure session and both peers exist
        session.add_messages(
            [
                observer.message("Hello from observer"),
                target.message("Hello from target"),
            ]
        )

        # Get observation scope
        obs_scope = observer.conclusions_of(target)

        # Create observation with specific content
        obs_scope.create(
            [
                {
                    "content": "User loves Italian cuisine especially pasta and pizza",
                    "session_id": session.id,
                },
            ]
        )

        # Query for food-related observations
        results = obs_scope.query("food preferences")

        assert len(results) >= 1
        # At least one result should mention Italian food
        contents = " ".join(obs.content for obs in results)
        assert "Italian" in contents or "pasta" in contents or "pizza" in contents


@pytest.mark.asyncio
async def test_observation_create_then_delete(
    client_fixture: tuple[Honcho, str],
):
    """
    Tests that created observations can be deleted.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        observer = await honcho_client.aio.peer(id="test-obs-create-delete-observer")
        target = await honcho_client.aio.peer(id="test-obs-create-delete-target")
        session = await honcho_client.aio.session(id="test-obs-create-delete-session")

        # Ensure session and both peers exist
        await session.aio.add_messages(
            [
                observer.message("Hello from observer"),
                target.message("Hello from target"),
            ]
        )

        # Get observation scope
        obs_scope = observer.conclusions_of(target)

        # Create observations
        created = await obs_scope.aio.create(
            [{"content": "Observation to be deleted", "session_id": session.id}]
        )

        observation_id = created[0].id

        # Delete the observation
        await obs_scope.aio.delete(observation_id)

        # List observations - should not contain deleted one
        listed = await obs_scope.aio.list()
        listed_ids = {obs.id for obs in listed.items}
        assert observation_id not in listed_ids
    else:
        observer = honcho_client.peer(id="test-obs-create-delete-observer")
        target = honcho_client.peer(id="test-obs-create-delete-target")
        session = honcho_client.session(id="test-obs-create-delete-session")

        # Ensure session and both peers exist
        session.add_messages(
            [
                observer.message("Hello from observer"),
                target.message("Hello from target"),
            ]
        )

        # Get observation scope
        obs_scope = observer.conclusions_of(target)

        # Create observations
        created = obs_scope.create(
            [{"content": "Observation to be deleted", "session_id": session.id}]
        )

        observation_id = created[0].id

        # Delete the observation
        obs_scope.delete(observation_id)

        # List observations - should not contain deleted one
        listed = obs_scope.list()
        listed_ids = {obs.id for obs in listed}
        assert observation_id not in listed_ids


@pytest.mark.asyncio
async def test_observation_get_by_id(
    client_fixture: tuple[Honcho, str],
):
    """
    Tests fetching a single conclusion by ID, including attribution fields.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        observer = await honcho_client.aio.peer(id="test-obs-get-by-id-observer")
        target = await honcho_client.aio.peer(id="test-obs-get-by-id-target")
        session = await honcho_client.aio.session(id="test-obs-get-by-id-session")

        # Ensure session and both peers exist
        await session.aio.add_messages(
            [
                observer.message("Hello from observer"),
                target.message("Hello from target"),
            ]
        )

        obs_scope = observer.conclusions_of(target)
        created = await obs_scope.aio.create(
            [{"content": "Conclusion to fetch", "session_id": session.id}]
        )

        fetched = await obs_scope.aio.get(created[0].id)

        assert isinstance(fetched, Conclusion)
        assert fetched.id == created[0].id
        assert fetched.content == "Conclusion to fetch"
        assert fetched.observer_id == observer.id
        assert fetched.observed_id == target.id
        assert fetched.level == "explicit"
        # User-created conclusions are explicit: no premises, derived once
        assert fetched.source_ids is None
        assert fetched.times_derived == 1
    else:
        observer = honcho_client.peer(id="test-obs-get-by-id-observer")
        target = honcho_client.peer(id="test-obs-get-by-id-target")
        session = honcho_client.session(id="test-obs-get-by-id-session")

        # Ensure session and both peers exist
        session.add_messages(
            [
                observer.message("Hello from observer"),
                target.message("Hello from target"),
            ]
        )

        obs_scope = observer.conclusions_of(target)
        created = obs_scope.create(
            [{"content": "Conclusion to fetch", "session_id": session.id}]
        )

        fetched = obs_scope.get(created[0].id)

        assert isinstance(fetched, Conclusion)
        assert fetched.id == created[0].id
        assert fetched.content == "Conclusion to fetch"
        assert fetched.observer_id == observer.id
        assert fetched.observed_id == target.id
        assert fetched.level == "explicit"
        # User-created conclusions are explicit: no premises, derived once
        assert fetched.source_ids is None
        assert fetched.times_derived == 1


@pytest.mark.asyncio
async def test_observation_get_many(
    client_fixture: tuple[Honcho, str],
):
    """
    Tests batch-fetching conclusions by ID (the tree-walk helper for source_ids).
    """
    honcho_client, client_type = client_fixture

    contents = [
        {"content": "First batch conclusion"},
        {"content": "Second batch conclusion"},
        {"content": "Third batch conclusion"},
    ]

    if client_type == "async":
        observer = await honcho_client.aio.peer(id="test-obs-get-many-observer")
        target = await honcho_client.aio.peer(id="test-obs-get-many-target")
        session = await honcho_client.aio.session(id="test-obs-get-many-session")

        # Ensure session and both peers exist
        await session.aio.add_messages(
            [
                observer.message("Hello from observer"),
                target.message("Hello from target"),
            ]
        )

        obs_scope = observer.conclusions_of(target)
        created = await obs_scope.aio.create(
            [{**c, "session_id": session.id} for c in contents]
        )

        assert await obs_scope.aio.get_many([]) == []

        fetched = await obs_scope.aio.get_many([c.id for c in created])
    else:
        observer = honcho_client.peer(id="test-obs-get-many-observer")
        target = honcho_client.peer(id="test-obs-get-many-target")
        session = honcho_client.session(id="test-obs-get-many-session")

        # Ensure session and both peers exist
        session.add_messages(
            [
                observer.message("Hello from observer"),
                target.message("Hello from target"),
            ]
        )

        obs_scope = observer.conclusions_of(target)
        created = obs_scope.create([{**c, "session_id": session.id} for c in contents])

        assert obs_scope.get_many([]) == []

        fetched = obs_scope.get_many([c.id for c in created])

    assert all(isinstance(c, Conclusion) for c in fetched)
    assert {c.id for c in fetched} == {c.id for c in created}
    assert {c.content for c in fetched} == {c["content"] for c in contents}


@pytest.mark.asyncio
async def test_observation_derived_empty_for_leaf(
    client_fixture: tuple[Honcho, str],
):
    """
    Tests the derived() traversal: a user-created (explicit) conclusion has
    nothing derived from it, so the endpoint returns an empty page.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        observer = await honcho_client.aio.peer(id="test-obs-derived-observer")
        target = await honcho_client.aio.peer(id="test-obs-derived-target")
        session = await honcho_client.aio.session(id="test-obs-derived-session")

        # Ensure session and both peers exist
        await session.aio.add_messages(
            [
                observer.message("Hello from observer"),
                target.message("Hello from target"),
            ]
        )

        obs_scope = observer.conclusions_of(target)
        created = await obs_scope.aio.create(
            [{"content": "Leaf conclusion", "session_id": session.id}]
        )

        page = await obs_scope.aio.derived(created[0].id)
        items = page.items
    else:
        observer = honcho_client.peer(id="test-obs-derived-observer")
        target = honcho_client.peer(id="test-obs-derived-target")
        session = honcho_client.session(id="test-obs-derived-session")

        # Ensure session and both peers exist
        session.add_messages(
            [
                observer.message("Hello from observer"),
                target.message("Hello from target"),
            ]
        )

        obs_scope = observer.conclusions_of(target)
        created = obs_scope.create(
            [{"content": "Leaf conclusion", "session_id": session.id}]
        )

        page = obs_scope.derived(created[0].id)
        items = page.items

    assert items == []


@pytest.mark.asyncio
async def test_self_observation_create(
    client_fixture: tuple[Honcho, str],
):
    """
    Tests creating self-observations (observer == observed).
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        peer = await honcho_client.aio.peer(id="test-self-obs-create-peer")
        session = await honcho_client.aio.session(id="test-self-obs-create-session")

        # Ensure session exists
        await session.aio.add_messages([peer.message("Hello")])

        # Get self-observation scope
        obs_scope = peer.conclusions
        assert isinstance(obs_scope, ConclusionsView)
        assert obs_scope.observer == peer.id
        assert obs_scope.observed == peer.id

        # Create a self-observation
        created = await obs_scope.aio.create(
            [{"content": "I prefer morning workouts", "session_id": session.id}]
        )

        assert len(created) == 1
        assert created[0].observer_id == peer.id
        assert created[0].observed_id == peer.id
    else:
        peer = honcho_client.peer(id="test-self-obs-create-peer")
        session = honcho_client.session(id="test-self-obs-create-session")

        # Ensure session exists
        session.add_messages([peer.message("Hello")])

        # Get self-observation scope
        obs_scope = peer.conclusions
        assert isinstance(obs_scope, ConclusionsView)
        assert obs_scope.observer == peer.id
        assert obs_scope.observed == peer.id

        # Create a self-observation
        created = obs_scope.create(
            [{"content": "I prefer morning workouts", "session_id": session.id}]
        )

        assert len(created) == 1
        assert created[0].observer_id == peer.id
        assert created[0].observed_id == peer.id


@pytest.mark.asyncio
async def test_observation_create_with_session_filter(
    client_fixture: tuple[Honcho, str],
):
    """
    Tests creating observations and filtering list by session.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        observer = await honcho_client.aio.peer(id="test-obs-session-filter-observer")
        target = await honcho_client.aio.peer(id="test-obs-session-filter-target")
        session1 = await honcho_client.aio.session(id="test-obs-session-filter-s1")
        session2 = await honcho_client.aio.session(id="test-obs-session-filter-s2")

        # Ensure sessions and both peers exist
        await session1.aio.add_messages(
            [
                observer.message("Hello 1 from observer"),
                target.message("Hello 1 from target"),
            ]
        )
        await session2.aio.add_messages(
            [
                observer.message("Hello 2 from observer"),
                target.message("Hello 2 from target"),
            ]
        )

        # Get observation scope
        obs_scope = observer.conclusions_of(target)

        # Create observations in different sessions
        await obs_scope.aio.create(
            [{"content": "Session 1 observation", "session_id": session1.id}]
        )
        await obs_scope.aio.create(
            [{"content": "Session 2 observation", "session_id": session2.id}]
        )

        # List filtered by session1
        s1_obs = await obs_scope.aio.list(session=session1)
        s1_contents = [obs.content for obs in s1_obs.items]
        assert "Session 1 observation" in s1_contents
        assert "Session 2 observation" not in s1_contents

        # List filtered by session2
        s2_obs = await obs_scope.aio.list(session=session2)
        s2_contents = [obs.content for obs in s2_obs.items]
        assert "Session 2 observation" in s2_contents
        assert "Session 1 observation" not in s2_contents
    else:
        observer = honcho_client.peer(id="test-obs-session-filter-observer")
        target = honcho_client.peer(id="test-obs-session-filter-target")
        session1 = honcho_client.session(id="test-obs-session-filter-s1")
        session2 = honcho_client.session(id="test-obs-session-filter-s2")

        # Ensure sessions and both peers exist
        session1.add_messages(
            [
                observer.message("Hello 1 from observer"),
                target.message("Hello 1 from target"),
            ]
        )
        session2.add_messages(
            [
                observer.message("Hello 2 from observer"),
                target.message("Hello 2 from target"),
            ]
        )

        # Get observation scope
        obs_scope = observer.conclusions_of(target)

        # Create observations in different sessions
        obs_scope.create(
            [{"content": "Session 1 observation", "session_id": session1.id}]
        )
        obs_scope.create(
            [{"content": "Session 2 observation", "session_id": session2.id}]
        )

        # List filtered by session1
        s1_obs = obs_scope.list(session=session1)
        s1_contents = [obs.content for obs in s1_obs]
        assert "Session 1 observation" in s1_contents
        assert "Session 2 observation" not in s1_contents

        # List filtered by session2
        s2_obs = obs_scope.list(session=session2)
        s2_contents = [obs.content for obs in s2_obs]
        assert "Session 2 observation" in s2_contents
        assert "Session 1 observation" not in s2_contents


@pytest.mark.asyncio
async def test_observation_scope_via_peer_string(
    client_fixture: tuple[Honcho, str],
):
    """
    Tests creating observations via conclusions_of(string).
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        observer = await honcho_client.aio.peer(id="test-obs-string-target-observer")
        target = await honcho_client.aio.peer(id="test-obs-string-target-target")
        session = await honcho_client.aio.session(id="test-obs-string-target-session")

        # Ensure session and both peers exist
        await session.aio.add_messages(
            [
                observer.message("Hello from observer"),
                target.message("Hello from target"),
            ]
        )

        # Get observation scope using string ID
        obs_scope = observer.conclusions_of(target.id)
        assert obs_scope.observed == target.id

        # Create observation
        created = await obs_scope.aio.create(
            [{"content": "Created via string target", "session_id": session.id}]
        )

        assert len(created) == 1
        assert created[0].observed_id == target.id
    else:
        observer = honcho_client.peer(id="test-obs-string-target-observer")
        target = honcho_client.peer(id="test-obs-string-target-target")
        session = honcho_client.session(id="test-obs-string-target-session")

        # Ensure session and both peers exist
        session.add_messages(
            [
                observer.message("Hello from observer"),
                target.message("Hello from target"),
            ]
        )

        # Get observation scope using string ID
        obs_scope = observer.conclusions_of(target.id)
        assert obs_scope.observed == target.id

        # Create observation
        created = obs_scope.create(
            [{"content": "Created via string target", "session_id": session.id}]
        )

        assert len(created) == 1
        assert created[0].observed_id == target.id


@pytest.mark.asyncio
async def test_observation_create_without_session_id(
    client_fixture: tuple[Honcho, str],
):
    """
    Tests creating observations without a session_id (sessionless/global conclusions).
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        observer = await honcho_client.aio.peer(id="test-obs-no-session-observer")
        target = await honcho_client.aio.peer(id="test-obs-no-session-target")

        # Create a session just to ensure peers exist
        session = await honcho_client.aio.session(id="test-obs-no-session-session")
        await session.aio.add_messages(
            [
                observer.message("Hello from observer"),
                target.message("Hello from target"),
            ]
        )

        # Get observation scope for observer -> target
        obs_scope = observer.conclusions_of(target)

        # Create observation WITHOUT session_id
        created = await obs_scope.aio.create(
            [
                ConclusionCreateParams(
                    content="Global observation without session",
                    # No session_id - this is the key test
                )
            ]
        )

        assert len(created) == 1
        assert isinstance(created[0], Conclusion)
        assert created[0].content == "Global observation without session"
        assert created[0].observer_id == observer.id
        assert created[0].observed_id == target.id
        assert created[0].session_id is None  # Should be None
        assert created[0].id  # Has an ID
    else:
        observer = honcho_client.peer(id="test-obs-no-session-observer")
        target = honcho_client.peer(id="test-obs-no-session-target")

        # Create a session just to ensure peers exist
        session = honcho_client.session(id="test-obs-no-session-session")
        session.add_messages(
            [
                observer.message("Hello from observer"),
                target.message("Hello from target"),
            ]
        )

        # Get observation scope for observer -> target
        obs_scope = observer.conclusions_of(target)

        # Create observation WITHOUT session_id
        created = obs_scope.create(
            [
                ConclusionCreateParams(
                    content="Global observation without session",
                    # No session_id - this is the key test
                )
            ]
        )

        assert len(created) == 1
        assert isinstance(created[0], Conclusion)
        assert created[0].content == "Global observation without session"
        assert created[0].observer_id == observer.id
        assert created[0].observed_id == target.id
        assert created[0].session_id is None  # Should be None
        assert created[0].id  # Has an ID


@pytest.mark.asyncio
async def test_observation_create_mixed_session_and_sessionless(
    client_fixture: tuple[Honcho, str],
):
    """
    Tests creating a batch with both session-scoped and sessionless observations.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        observer = await honcho_client.aio.peer(id="test-obs-mixed-session-observer")
        target = await honcho_client.aio.peer(id="test-obs-mixed-session-target")
        session = await honcho_client.aio.session(id="test-obs-mixed-session-session")

        # Ensure session and both peers exist
        await session.aio.add_messages(
            [
                observer.message("Hello from observer"),
                target.message("Hello from target"),
            ]
        )

        # Get observation scope
        obs_scope = observer.conclusions_of(target)

        # Create mixed batch: one with session, one without
        created = await obs_scope.aio.create(
            [
                {"content": "Session-scoped observation", "session_id": session.id},
                {"content": "Global observation without session"},  # No session_id
            ]
        )

        assert len(created) == 2

        # Find observations by content
        session_obs = next(
            c for c in created if c.content == "Session-scoped observation"
        )
        global_obs = next(
            c for c in created if c.content == "Global observation without session"
        )

        assert session_obs.session_id == session.id
        assert global_obs.session_id is None
    else:
        observer = honcho_client.peer(id="test-obs-mixed-session-observer")
        target = honcho_client.peer(id="test-obs-mixed-session-target")
        session = honcho_client.session(id="test-obs-mixed-session-session")

        # Ensure session and both peers exist
        session.add_messages(
            [
                observer.message("Hello from observer"),
                target.message("Hello from target"),
            ]
        )

        # Get observation scope
        obs_scope = observer.conclusions_of(target)

        # Create mixed batch: one with session, one without
        created = obs_scope.create(
            [
                {"content": "Session-scoped observation", "session_id": session.id},
                {"content": "Global observation without session"},  # No session_id
            ]
        )

        assert len(created) == 2

        # Find observations by content
        session_obs = next(
            c for c in created if c.content == "Session-scoped observation"
        )
        global_obs = next(
            c for c in created if c.content == "Global observation without session"
        )

        assert session_obs.session_id == session.id
        assert global_obs.session_id is None


@pytest.mark.asyncio
async def test_list_rejects_reserved_scope_filter_keys(
    client_fixture: tuple[Honcho, str],
):
    """`list` rejects observer/observed/session filter keys managed by the scope.

    These keys are fixed by the scope (observer/observed) or by the dedicated
    ``session=`` parameter, so passing them in ``filters`` would silently return
    data from a different scope. The guard raises before any HTTP call.
    """
    honcho_client, client_type = client_fixture
    reserved = [
        "observer",
        "observed",
        "observer_id",
        "observed_id",
        "session_id",
        "session",
    ]

    if client_type == "async":
        observer = await honcho_client.aio.peer(id="test-obs-reserved-list-observer")
        target = await honcho_client.aio.peer(id="test-obs-reserved-list-target")
        obs_scope = observer.conclusions_of(target)
        for key in reserved:
            with pytest.raises(ValueError, match="managed by this conclusions view"):
                await obs_scope.aio.list(filters={key: "someone-else"})
        # A non-reserved filter (level) is allowed through.
        await obs_scope.aio.list(filters={"level": "explicit"})
    else:
        observer = honcho_client.peer(id="test-obs-reserved-list-observer")
        target = honcho_client.peer(id="test-obs-reserved-list-target")
        obs_scope = observer.conclusions_of(target)
        for key in reserved:
            with pytest.raises(ValueError, match="managed by this conclusions view"):
                obs_scope.list(filters={key: "someone-else"})
        obs_scope.list(filters={"level": "explicit"})


@pytest.mark.asyncio
async def test_query_rejects_reserved_scope_filter_keys(
    client_fixture: tuple[Honcho, str],
):
    """`query` rejects observer/observed filter keys but allows session_id.

    Unlike ``list``, ``query`` has no dedicated session parameter, so
    ``session_id`` remains a normal filter and must NOT be rejected.
    """
    honcho_client, client_type = client_fixture
    reserved = ["observer", "observed", "observer_id", "observed_id"]

    if client_type == "async":
        observer = await honcho_client.aio.peer(id="test-obs-reserved-query-observer")
        target = await honcho_client.aio.peer(id="test-obs-reserved-query-target")
        obs_scope = observer.conclusions_of(target)
        for key in reserved:
            with pytest.raises(ValueError, match="managed by this conclusions view"):
                await obs_scope.aio.query("q", filters={key: "someone-else"})
        # session_id is a normal filter for query (no dedicated param) — allowed.
        await obs_scope.aio.query("q", filters={"session_id": "some-session"})
    else:
        observer = honcho_client.peer(id="test-obs-reserved-query-observer")
        target = honcho_client.peer(id="test-obs-reserved-query-target")
        obs_scope = observer.conclusions_of(target)
        for key in reserved:
            with pytest.raises(ValueError, match="managed by this conclusions view"):
                obs_scope.query("q", filters={key: "someone-else"})
        obs_scope.query("q", filters={"session_id": "some-session"})


@pytest.mark.asyncio
async def test_workspace_conclusions_list_and_get(
    client_fixture: tuple[Honcho, str],
):
    """honcho.conclusions lists and fetches across observer/observed pairs."""
    honcho_client, client_type = client_fixture

    if client_type == "async":
        alice = await honcho_client.aio.peer(id="test-ws-conc-alice")
        bob = await honcho_client.aio.peer(id="test-ws-conc-bob")
        session = await honcho_client.aio.session(id="test-ws-conc-session")
        await session.aio.add_messages(
            [alice.message("hi from alice"), bob.message("hi from bob")]
        )
        alice_conc = (
            await alice.conclusions.aio.create(
                [{"content": "Alice self conclusion", "session_id": session.id}]
            )
        )[0]
        about_bob = (
            await alice.conclusions_of(bob).aio.create(
                [{"content": "Alice about Bob", "session_id": session.id}]
            )
        )[0]

        assert isinstance(honcho_client.conclusions, WorkspaceConclusions)

        page = await honcho_client.aio.conclusions.list(size=100)
        ids = {c.id for c in page.items}
        assert alice_conc.id in ids
        assert about_bob.id in ids

        about_bob_only = await honcho_client.aio.conclusions.list(
            filters={"observed_id": bob.id}, size=100
        )
        assert {c.id for c in about_bob_only.items} >= {about_bob.id}
        assert all(c.observed_id == bob.id for c in about_bob_only.items)

        session_page = await honcho_client.aio.conclusions.list(
            filters={"session_id": session.id}, size=100
        )
        assert {c.id for c in session_page.items} >= {alice_conc.id, about_bob.id}

        fetched = await honcho_client.aio.conclusions.get(about_bob.id)
        assert fetched.id == about_bob.id
        assert fetched.observer_id == alice.id
        assert fetched.observed_id == bob.id

        batch = await honcho_client.aio.conclusions.get_many(
            [alice_conc.id, about_bob.id]
        )
        assert {c.id for c in batch} == {alice_conc.id, about_bob.id}

        # Scoped get cannot see a conclusion from a different pair.
        with pytest.raises(NotFoundError):
            await alice.conclusions.aio.get(about_bob.id)
    else:
        alice = honcho_client.peer(id="test-ws-conc-alice")
        bob = honcho_client.peer(id="test-ws-conc-bob")
        session = honcho_client.session(id="test-ws-conc-session")
        session.add_messages(
            [alice.message("hi from alice"), bob.message("hi from bob")]
        )
        alice_conc = alice.conclusions.create(
            [{"content": "Alice self conclusion", "session_id": session.id}]
        )[0]
        about_bob = alice.conclusions_of(bob).create(
            [{"content": "Alice about Bob", "session_id": session.id}]
        )[0]

        assert isinstance(honcho_client.conclusions, WorkspaceConclusions)

        page = honcho_client.conclusions.list(size=100)
        ids = {c.id for c in page.items}
        assert alice_conc.id in ids
        assert about_bob.id in ids

        about_bob_only = honcho_client.conclusions.list(
            filters={"observed_id": bob.id}, size=100
        )
        assert {c.id for c in about_bob_only.items} >= {about_bob.id}
        assert all(c.observed_id == bob.id for c in about_bob_only.items)

        session_page = honcho_client.conclusions.list(
            filters={"session_id": session.id}, size=100
        )
        assert {c.id for c in session_page.items} >= {alice_conc.id, about_bob.id}

        fetched = honcho_client.conclusions.get(about_bob.id)
        assert fetched.id == about_bob.id
        assert fetched.observer_id == alice.id
        assert fetched.observed_id == bob.id

        batch = honcho_client.conclusions.get_many([alice_conc.id, about_bob.id])
        assert {c.id for c in batch} == {alice_conc.id, about_bob.id}

        with pytest.raises(NotFoundError):
            alice.conclusions.get(about_bob.id)
