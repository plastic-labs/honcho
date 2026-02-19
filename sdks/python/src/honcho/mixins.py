"""Mixins for common SDK functionality."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import Field, validate_call


class MetadataConfigMixin(ABC):
    """
    Mixin providing get/set/refresh methods for metadata and configuration.

    Classes using this mixin must implement:
    - _get_http_client() -> HTTPClientProtocol
    - _get_fetch_route() -> str
    - _get_update_route() -> str
    - _get_fetch_body() -> dict[str, Any]
    - _parse_response(data: dict[str, Any]) -> tuple[dict, dict]

    And must have these attributes:
    - _metadata: dict[str, object] | None
    - _configuration: dict[str, object] | None
    """

    _metadata: dict[str, object] | None
    _configuration: dict[str, object] | None

    @abstractmethod
    def _get_http_client(self) -> Any:
        """Get the HTTP client for making requests."""
        ...

    @abstractmethod
    def _get_fetch_route(self) -> str:
        """Get the route for fetching metadata/configuration."""
        ...

    @abstractmethod
    def _get_update_route(self) -> str:
        """Get the route for updating metadata/configuration."""
        ...

    @abstractmethod
    def _get_fetch_body(self) -> dict[str, Any]:
        """Get the body for fetching metadata/configuration."""
        ...

    @abstractmethod
    def _parse_response(
        self, data: dict[str, Any]
    ) -> tuple[dict[str, object], dict[str, object]]:
        """Parse the response to extract metadata and configuration."""
        ...

    def get_metadata(self) -> dict[str, object]:
        """
        Get metadata from the server and update the cache.

        Returns:
            A dictionary containing the metadata. Returns an empty dictionary
            if no metadata is set.
        """
        data = self._get_http_client().post(
            self._get_fetch_route(), body=self._get_fetch_body()
        )
        metadata, configuration = self._parse_response(data)
        self._metadata = metadata
        self._configuration = configuration
        return self._metadata

    @validate_call
    def set_metadata(
        self,
        metadata: dict[str, object] = Field(
            ..., description="Metadata dictionary to set"
        ),
    ) -> None:
        """
        Set metadata on the server and update the cache.

        Args:
            metadata: A dictionary of metadata to set.
                      Keys must be strings, values can be any JSON-serializable type.
        """
        self._get_http_client().put(
            self._get_update_route(),
            body={"metadata": metadata},
        )
        self._metadata = metadata

    def get_configuration(self) -> dict[str, object]:
        """
        Get configuration from the server and update the cache.

        Returns:
            A dictionary containing the configuration. Returns an empty dictionary
            if no configuration is set.
        """
        data = self._get_http_client().post(
            self._get_fetch_route(), body=self._get_fetch_body()
        )
        metadata, configuration = self._parse_response(data)
        self._metadata = metadata
        self._configuration = configuration
        return self._configuration

    @validate_call
    def set_configuration(
        self,
        configuration: dict[str, object] = Field(
            ..., description="Configuration dictionary to set"
        ),
    ) -> None:
        """
        Set configuration on the server and update the cache.

        Args:
            configuration: A dictionary of configuration to set.
                    Keys must be strings, values can be any JSON-serializable type.
        """
        self._get_http_client().put(
            self._get_update_route(),
            body={"configuration": configuration},
        )
        self._configuration = configuration

    def refresh(self) -> None:
        """
        Refresh cached metadata and configuration from the server.

        Makes a single API call to retrieve the latest metadata and configuration
        and updates the cached attributes.
        """
        data = self._get_http_client().post(
            self._get_fetch_route(), body=self._get_fetch_body()
        )
        metadata, configuration = self._parse_response(data)
        self._metadata = metadata
        self._configuration = configuration


class AsyncMetadataConfigMixin(ABC):
    """
    Async mixin providing get/set/refresh methods for metadata and configuration.

    Classes using this mixin must implement:
    - _get_async_http_client() -> AsyncHTTPClientProtocol
    - _get_fetch_route() -> str
    - _get_update_route() -> str
    - _get_fetch_body() -> dict[str, Any]
    - _parse_response(data: dict[str, Any]) -> tuple[dict, dict]
    - _set_metadata(metadata: dict[str, object]) -> None
    - _set_configuration(configuration: dict[str, object]) -> None
    """

    @abstractmethod
    def _get_async_http_client(self) -> Any:
        """Get the async HTTP client for making requests."""
        ...

    @abstractmethod
    def _get_fetch_route(self) -> str:
        """Get the route for fetching metadata/configuration."""
        ...

    @abstractmethod
    def _get_update_route(self) -> str:
        """Get the route for updating metadata/configuration."""
        ...

    @abstractmethod
    def _get_fetch_body(self) -> dict[str, Any]:
        """Get the body for fetching metadata/configuration."""
        ...

    @abstractmethod
    def _parse_response(
        self, data: dict[str, Any]
    ) -> tuple[dict[str, object], dict[str, object]]:
        """Parse the response to extract metadata and configuration."""
        ...

    @abstractmethod
    def _set_metadata(self, metadata: dict[str, object]) -> None:
        """Set metadata on the parent object."""
        ...

    @abstractmethod
    def _set_configuration(self, configuration: dict[str, object]) -> None:
        """Set configuration on the parent object."""
        ...

    @abstractmethod
    def _get_metadata(self) -> dict[str, object]:
        """Get cached metadata from the parent object."""
        ...

    @abstractmethod
    def _get_configuration(self) -> dict[str, object]:
        """Get cached configuration from the parent object."""
        ...

    async def get_metadata(self) -> dict[str, object]:
        """Get metadata from the server asynchronously."""
        data = await self._get_async_http_client().post(
            self._get_fetch_route(), body=self._get_fetch_body()
        )
        metadata, configuration = self._parse_response(data)
        self._set_metadata(metadata)
        self._set_configuration(configuration)
        return self._get_metadata()

    async def set_metadata(self, metadata: dict[str, object]) -> None:
        """Set metadata on the server asynchronously."""
        await self._get_async_http_client().put(
            self._get_update_route(),
            body={"metadata": metadata},
        )
        self._set_metadata(metadata)

    async def get_configuration(self) -> dict[str, object]:
        """Get configuration from the server asynchronously."""
        data = await self._get_async_http_client().post(
            self._get_fetch_route(), body=self._get_fetch_body()
        )
        metadata, configuration = self._parse_response(data)
        self._set_metadata(metadata)
        self._set_configuration(configuration)
        return self._get_configuration()

    async def set_configuration(self, configuration: dict[str, object]) -> None:
        """Set configuration on the server asynchronously."""
        await self._get_async_http_client().put(
            self._get_update_route(),
            body={"configuration": configuration},
        )
        self._set_configuration(configuration)

    async def refresh(self) -> None:
        """Refresh cached metadata and configuration asynchronously."""
        data = await self._get_async_http_client().post(
            self._get_fetch_route(), body=self._get_fetch_body()
        )
        metadata, configuration = self._parse_response(data)
        self._set_metadata(metadata)
        self._set_configuration(configuration)
