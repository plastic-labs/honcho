"""Thin raw-HTTP client for one Honcho API container.

The matrix talks to the /v3 contract directly rather than through an SDK,
because the SDK version would pin one side of the comparison.
"""

from typing import Any

import httpx


class APIError(RuntimeError):
    """A /v3 call returned an unexpected status."""


class HonchoAPI:
    """One running API container, labelled by the image it runs."""

    def __init__(self, label: str, base_url: str) -> None:
        self.label: str = label
        self._http: httpx.Client = httpx.Client(base_url=base_url, timeout=30.0)

    def __repr__(self) -> str:
        return f"HonchoAPI({self.label})"

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: dict[str, Any] | None = None,
        ok: tuple[int, ...] = (200, 201, 202, 204),
    ) -> httpx.Response:
        response = self._http.request(method, path, json=json, params=params)
        if response.status_code not in ok:
            raise APIError(
                f"[{self.label}] {method} {path} -> {response.status_code}: "
                + response.text[:500]
            )
        return response

    def _all_pages(self, path: str, body: dict[str, Any] | None = None) -> list[Any]:
        items: list[Any] = []
        page = 1
        while True:
            data = self.request(
                "POST", path, json=body or {}, params={"page": page, "size": 100}
            ).json()
            items.extend(data["items"])
            if page >= data.get("pages", 1):
                return items
            page += 1

    # Workspaces, peers, sessions

    def workspace(self, name: str, metadata: dict[str, Any] | None = None) -> None:
        body: dict[str, Any] = {"id": name}
        if metadata is not None:
            body["metadata"] = metadata
        self.request("POST", "/v3/workspaces", json=body)

    def update_workspace(self, name: str, metadata: dict[str, Any]) -> None:
        self.request("PUT", f"/v3/workspaces/{name}", json={"metadata": metadata})

    def list_workspaces(self) -> list[dict[str, Any]]:
        return self._all_pages("/v3/workspaces/list")

    def peer(
        self, workspace: str, name: str, metadata: dict[str, Any] | None = None
    ) -> None:
        body: dict[str, Any] = {"id": name}
        if metadata is not None:
            body["metadata"] = metadata
        self.request("POST", f"/v3/workspaces/{workspace}/peers", json=body)

    def list_peers(self, workspace: str) -> list[dict[str, Any]]:
        return self._all_pages(f"/v3/workspaces/{workspace}/peers/list")

    def session(
        self,
        workspace: str,
        name: str,
        peers: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        body: dict[str, Any] = {"id": name, "peers": {p: {} for p in peers}}
        if metadata is not None:
            body["metadata"] = metadata
        self.request("POST", f"/v3/workspaces/{workspace}/sessions", json=body)

    def list_sessions(self, workspace: str) -> list[dict[str, Any]]:
        return self._all_pages(f"/v3/workspaces/{workspace}/sessions/list")

    def delete_session(self, workspace: str, session: str) -> None:
        self.request("DELETE", f"/v3/workspaces/{workspace}/sessions/{session}")

    def summaries(self, workspace: str, session: str) -> dict[str, Any]:
        return self.request(
            "GET", f"/v3/workspaces/{workspace}/sessions/{session}/summaries"
        ).json()

    # Messages

    def add_messages(
        self, workspace: str, session: str, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return self.request(
            "POST",
            f"/v3/workspaces/{workspace}/sessions/{session}/messages",
            json={"messages": messages},
        ).json()

    def get_message(
        self, workspace: str, session: str, message_id: str
    ) -> dict[str, Any] | None:
        response = self.request(
            "GET",
            f"/v3/workspaces/{workspace}/sessions/{session}/messages/{message_id}",
            ok=(200, 404),
        )
        return response.json() if response.status_code == 200 else None

    # Conclusions

    def create_conclusions(
        self, workspace: str, conclusions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return self.request(
            "POST",
            f"/v3/workspaces/{workspace}/conclusions",
            json={"conclusions": conclusions},
        ).json()

    def list_conclusions(self, workspace: str) -> list[dict[str, Any]]:
        return self._all_pages(f"/v3/workspaces/{workspace}/conclusions/list")

    def get_conclusion(
        self, workspace: str, conclusion_id: str
    ) -> dict[str, Any] | None:
        response = self.request(
            "GET",
            f"/v3/workspaces/{workspace}/conclusions/{conclusion_id}",
            ok=(200, 404),
        )
        return response.json() if response.status_code == 200 else None

    def delete_conclusion(self, workspace: str, conclusion_id: str) -> None:
        self.request(
            "DELETE", f"/v3/workspaces/{workspace}/conclusions/{conclusion_id}"
        )

    # Queued work

    def schedule_dream(self, workspace: str, observer: str, observed: str) -> None:
        self.request(
            "POST",
            f"/v3/workspaces/{workspace}/schedule_dream",
            json={"observer": observer, "observed": observed, "dream_type": "omni"},
        )
