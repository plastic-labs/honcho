# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownLambdaType=false
"""Every /v3 operation answers schema-valid input without a 5xx.

Schemathesis generates requests from the OpenAPI schema. Path IDs are random,
so most requests name a workspace, peer, or session that does not exist, which
is exactly the case users hit and report as a 500.
"""

import os
from typing import Any

import pytest
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from schemathesis import GenerationMode
from schemathesis.core.result import Ok

from tests.property.conftest import API_SCHEMA

# Raise locally to dig for rarer failures, e.g. HONCHO_FUZZ_EXAMPLES=1000
MAX_EXAMPLES = int(os.environ.get("HONCHO_FUZZ_EXAMPLES", "50"))
# A file to append one "<mode> <METHOD> <path> <status>" line per request to.
# A route that never reaches a 2xx is only being fuzzed on its error paths.
STATS_PATH = os.environ.get("HONCHO_FUZZ_STATS")

# Body fields that name an existing resource, mapped to the seeded ID to use.
BODY_ID_FIELDS = {
    "peer_id": "peer_id",
    "observer_id": "peer_id",
    "observed_id": "peer_id",
    "target": "peer_id",
    "session_id": "session_id",
}

OPERATIONS = [
    result.ok()
    for result in API_SCHEMA.get_all_operations()
    if isinstance(result, Ok) and result.ok().path.startswith("/v3")
]


def _point_at_seeded(
    value: Any, seeded_ids: dict[str, str], data: st.DataObject
) -> None:
    """Swap ID fields anywhere in a JSON body for seeded IDs, most of the time."""
    if isinstance(value, list):
        for item in value:
            _point_at_seeded(item, seeded_ids, data)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in BODY_ID_FIELDS and isinstance(item, str):
                if data.draw(st.integers(0, 3)):
                    value[key] = seeded_ids[BODY_ID_FIELDS[key]]
            else:
                _point_at_seeded(item, seeded_ids, data)


@pytest.mark.parametrize("mode", [GenerationMode.POSITIVE, GenerationMode.NEGATIVE])
@pytest.mark.parametrize("operation", OPERATIONS, ids=lambda op: op.label)
def test_operation_does_not_return_5xx(
    operation, mode: GenerationMode, fuzz_client: TestClient, seeded_ids: dict[str, str]
):
    @settings(
        max_examples=MAX_EXAMPLES,
        deadline=None,
        database=None,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.too_slow,
            HealthCheck.filter_too_much,
            HealthCheck.data_too_large,
        ],
    )
    @given(data=st.data())
    def run(data: st.DataObject) -> None:
        case = data.draw(operation.as_strategy(generation_mode=mode))
        # Point each path parameter at a seeded resource most of the time, so
        # requests get past the parent lookup. Negative mode keeps its own
        # (deliberately invalid) values.
        if mode is GenerationMode.POSITIVE:
            for name in case.path_parameters:
                if name in seeded_ids and data.draw(st.integers(0, 3)):
                    case.path_parameters[name] = seeded_ids[name]
            _point_at_seeded(case.body, seeded_ids, data)
        kwargs = case.as_transport_kwargs(base_url="http://testserver")
        try:
            response = fuzz_client.request(
                method=kwargs["method"],
                url=kwargs["url"],
                params=kwargs.get("params"),
                headers=kwargs.get("headers"),
                json=kwargs.get("json"),
            )
        except Exception as exc:
            # TestClient re-raises unhandled server exceptions; report them
            # the same way as a 5xx so the failure names the request.
            first_line = str(exc).splitlines()[0] if str(exc) else ""
            pytest.fail(
                f"{type(exc).__name__}: {first_line}\n{case.as_curl_command()}",
                pytrace=False,
            )
        if STATS_PATH:
            with open(STATS_PATH, "a") as stats:
                stats.write(f"{mode.value} {operation.label} {response.status_code}\n")
        assert response.status_code < 500, (
            f"{response.status_code} from {case.as_curl_command()}\n{response.text}"
        )

    run()
