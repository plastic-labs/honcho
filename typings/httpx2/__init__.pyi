# Type-checking shim: starlette>=1.2 annotates its TestClient against `httpx2`
# under TYPE_CHECKING but falls back to `httpx` at runtime when `httpx2` is not
# installed. Honcho only installs `httpx`, so alias `httpx2` to it here to keep
# TestClient's signatures typed instead of resolving to Unknown.
from httpx import *  # noqa: F403
from httpx import _client as _client
from httpx import _types as _types
