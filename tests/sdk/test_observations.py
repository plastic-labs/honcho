"""Tests for observation SDK methods."""

import pytest

from sdks.python.src.honcho.async_client.client import AsyncHoncho
from sdks.python.src.honcho.client import Honcho
from sdks.python.src.honcho.observations import (
    AsyncObservationScope,
    Observation,
    ObservationScope,
)


@pytest.mark.asyncio
async def test_observation_create_single(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
):
    """
    Tests creating a single observation via the SDK.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        observer = await honcho_client.peer(id="test-obs-create-single-observer")
        target = await honcho_client.peer(id="test-obs-create-single-target")
        session = await honcho_client.session(id="test-obs-create-single-session")

        # Ensure session and both peers exist by adding messages from both
        await session.add_messages(
            [
                observer.message("Hello from observer"),
                target.message("Hello from target"),
            ]
        )

        # Get observation scope for observer -> target
        obs_scope = observer.observations_of(target)
        assert isinstance(obs_scope, AsyncObservationScope)

        # Create a single observation
        created = await obs_scope.create(
            {"content": "User prefers dark mode", "session_id": session.id}
        )

        assert len(created) == 1
        assert isinstance(created[0], Observation)
        assert created[0].content == "User prefers dark mode"
        assert created[0].observer_id == observer.id
        assert created[0].observed_id == target.id
        assert created[0].session_id == session.id
        assert created[0].id  # Has an ID
    else:
        assert isinstance(honcho_client, Honcho)
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
        obs_scope = observer.observations_of(target)
        assert isinstance(obs_scope, ObservationScope)

        # Create a single observation
        created = obs_scope.create(
            {"content": "User prefers dark mode", "session_id": session.id}
        )

        assert len(created) == 1
        assert isinstance(created[0], Observation)
        assert created[0].content == "User prefers dark mode"
        assert created[0].observer_id == observer.id
        assert created[0].observed_id == target.id
        assert created[0].session_id == session.id
        assert created[0].id  # Has an ID


@pytest.mark.asyncio
async def test_observation_create_batch(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
):
    """
    Tests creating multiple observations in a batch via the SDK.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        observer = await honcho_client.peer(id="test-obs-create-batch-observer")
        target = await honcho_client.peer(id="test-obs-create-batch-target")
        session = await honcho_client.session(id="test-obs-create-batch-session")

        # Ensure session and both peers exist
        await session.add_messages(
            [
                observer.message("Hello from observer"),
                target.message("Hello from target"),
            ]
        )

        # Get observation scope
        obs_scope = observer.observations_of(target)

        # Create multiple observations
        created = await obs_scope.create(
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
    else:
        assert isinstance(honcho_client, Honcho)
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
        obs_scope = observer.observations_of(target)

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
    client_fixture: tuple[Honcho | AsyncHoncho, str],
):
    """
    Tests that created observations can be listed.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        observer = await honcho_client.peer(id="test-obs-create-list-observer")
        target = await honcho_client.peer(id="test-obs-create-list-target")
        session = await honcho_client.session(id="test-obs-create-list-session")

        # Ensure session and both peers exist
        await session.add_messages(
            [
                observer.message("Hello from observer"),
                target.message("Hello from target"),
            ]
        )

        # Get observation scope
        obs_scope = observer.observations_of(target)

        # Create observations
        created = await obs_scope.create(
            [
                {
                    "content": "Unique observation for list test",
                    "session_id": session.id,
                },
            ]
        )

        # List observations
        listed = await obs_scope.list()

        # The created observation should be in the list
        listed_ids = {obs.id for obs in listed}
        assert created[0].id in listed_ids
    else:
        assert isinstance(honcho_client, Honcho)
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
        obs_scope = observer.observations_of(target)

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
    client_fixture: tuple[Honcho | AsyncHoncho, str],
):
    """
    Tests that created observations can be queried semantically.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        observer = await honcho_client.peer(id="test-obs-create-query-observer")
        target = await honcho_client.peer(id="test-obs-create-query-target")
        session = await honcho_client.session(id="test-obs-create-query-session")

        # Ensure session and both peers exist
        await session.add_messages(
            [
                observer.message("Hello from observer"),
                target.message("Hello from target"),
            ]
        )

        # Get observation scope
        obs_scope = observer.observations_of(target)

        # Create observation with specific content
        await obs_scope.create(
            [
                {
                    "content": "User loves Italian cuisine especially pasta and pizza",
                    "session_id": session.id,
                },
            ]
        )

        # Query for food-related observations
        results = await obs_scope.query("food preferences")

        assert len(results) >= 1
        # At least one result should mention Italian food
        contents = " ".join(obs.content for obs in results)
        assert "Italian" in contents or "pasta" in contents or "pizza" in contents
    else:
        assert isinstance(honcho_client, Honcho)
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
        obs_scope = observer.observations_of(target)

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
    client_fixture: tuple[Honcho | AsyncHoncho, str],
):
    """
    Tests that created observations can be deleted.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        observer = await honcho_client.peer(id="test-obs-create-delete-observer")
        target = await honcho_client.peer(id="test-obs-create-delete-target")
        session = await honcho_client.session(id="test-obs-create-delete-session")

        # Ensure session and both peers exist
        await session.add_messages(
            [
                observer.message("Hello from observer"),
                target.message("Hello from target"),
            ]
        )

        # Get observation scope
        obs_scope = observer.observations_of(target)

        # Create observations
        created = await obs_scope.create(
            [
                {"content": "Observation to be deleted", "session_id": session.id},
            ]
        )

        observation_id = created[0].id

        # Delete the observation
        await obs_scope.delete(observation_id)

        # List observations - should not contain deleted one
        listed = await obs_scope.list()
        listed_ids = {obs.id for obs in listed}
        assert observation_id not in listed_ids
    else:
        assert isinstance(honcho_client, Honcho)
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
        obs_scope = observer.observations_of(target)

        # Create observations
        created = obs_scope.create(
            [
                {"content": "Observation to be deleted", "session_id": session.id},
            ]
        )

        observation_id = created[0].id

        # Delete the observation
        obs_scope.delete(observation_id)

        # List observations - should not contain deleted one
        listed = obs_scope.list()
        listed_ids = {obs.id for obs in listed}
        assert observation_id not in listed_ids


@pytest.mark.asyncio
async def test_self_observation_create(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
):
    """
    Tests creating self-observations (observer == observed).
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        peer = await honcho_client.peer(id="test-self-obs-create-peer")
        session = await honcho_client.session(id="test-self-obs-create-session")

        # Ensure session exists
        await session.add_messages([peer.message("Hello")])

        # Get self-observation scope
        obs_scope = peer.observations
        assert isinstance(obs_scope, AsyncObservationScope)
        assert obs_scope.observer == peer.id
        assert obs_scope.observed == peer.id

        # Create a self-observation
        created = await obs_scope.create(
            {"content": "I prefer morning workouts", "session_id": session.id}
        )

        assert len(created) == 1
        assert created[0].observer_id == peer.id
        assert created[0].observed_id == peer.id
    else:
        assert isinstance(honcho_client, Honcho)
        peer = honcho_client.peer(id="test-self-obs-create-peer")
        session = honcho_client.session(id="test-self-obs-create-session")

        # Ensure session exists
        session.add_messages([peer.message("Hello")])

        # Get self-observation scope
        obs_scope = peer.observations
        assert isinstance(obs_scope, ObservationScope)
        assert obs_scope.observer == peer.id
        assert obs_scope.observed == peer.id

        # Create a self-observation
        created = obs_scope.create(
            {"content": "I prefer morning workouts", "session_id": session.id}
        )

        assert len(created) == 1
        assert created[0].observer_id == peer.id
        assert created[0].observed_id == peer.id


@pytest.mark.asyncio
async def test_observation_create_with_session_filter(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
):
    """
    Tests creating observations and filtering list by session.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        observer = await honcho_client.peer(id="test-obs-session-filter-observer")
        target = await honcho_client.peer(id="test-obs-session-filter-target")
        session1 = await honcho_client.session(id="test-obs-session-filter-s1")
        session2 = await honcho_client.session(id="test-obs-session-filter-s2")

        # Ensure sessions and both peers exist
        await session1.add_messages(
            [
                observer.message("Hello 1 from observer"),
                target.message("Hello 1 from target"),
            ]
        )
        await session2.add_messages(
            [
                observer.message("Hello 2 from observer"),
                target.message("Hello 2 from target"),
            ]
        )

        # Get observation scope
        obs_scope = observer.observations_of(target)

        # Create observations in different sessions
        await obs_scope.create(
            [
                {"content": "Session 1 observation", "session_id": session1.id},
            ]
        )
        await obs_scope.create(
            [
                {"content": "Session 2 observation", "session_id": session2.id},
            ]
        )

        # List filtered by session1
        s1_obs = await obs_scope.list(session=session1)
        s1_contents = [obs.content for obs in s1_obs]
        assert "Session 1 observation" in s1_contents
        assert "Session 2 observation" not in s1_contents

        # List filtered by session2
        s2_obs = await obs_scope.list(session=session2)
        s2_contents = [obs.content for obs in s2_obs]
        assert "Session 2 observation" in s2_contents
        assert "Session 1 observation" not in s2_contents
    else:
        assert isinstance(honcho_client, Honcho)
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
        obs_scope = observer.observations_of(target)

        # Create observations in different sessions
        obs_scope.create(
            [
                {"content": "Session 1 observation", "session_id": session1.id},
            ]
        )
        obs_scope.create(
            [
                {"content": "Session 2 observation", "session_id": session2.id},
            ]
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
    client_fixture: tuple[Honcho | AsyncHoncho, str],
):
    """
    Tests creating observations via observations_of(string).
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        observer = await honcho_client.peer(id="test-obs-string-target-observer")
        target = await honcho_client.peer(id="test-obs-string-target-target")
        session = await honcho_client.session(id="test-obs-string-target-session")

        # Ensure session and both peers exist
        await session.add_messages(
            [
                observer.message("Hello from observer"),
                target.message("Hello from target"),
            ]
        )

        # Get observation scope using string ID
        obs_scope = observer.observations_of(target.id)
        assert obs_scope.observed == target.id

        # Create observation
        created = await obs_scope.create(
            {"content": "Created via string target", "session_id": session.id}
        )

        assert len(created) == 1
        assert created[0].observed_id == target.id
    else:
        assert isinstance(honcho_client, Honcho)
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
        obs_scope = observer.observations_of(target.id)
        assert obs_scope.observed == target.id

        # Create observation
        created = obs_scope.create(
            {"content": "Created via string target", "session_id": session.id}
        )

        assert len(created) == 1
        assert created[0].observed_id == target.id
