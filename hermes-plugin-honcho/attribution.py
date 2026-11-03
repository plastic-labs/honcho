"""Marketing attribution for links the plugin shows or opens.

Every Honcho web link that leaves Hermes (the browser sign-in, the device-code page, "get a key at
app.honcho.dev", the docs link in the config panel) carries the same UTM parameters, so a visit that
starts in Hermes lands in web analytics as its own source instead of "Direct"::

    https://app.honcho.dev/?utm_source=hermes&utm_medium=plugin&utm_content=apikey-prompt

Only these three static parameters are added. Nothing user-, machine- or session-identifying goes
into a URL.

``config_schema.py`` can't import this module (Hermes loads that file standalone, outside the package),
so it carries its tagged docs URL as a literal; ``tests/test_attribution.py`` keeps the two in sync.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

HONCHO_APP_URL = "https://app.honcho.dev"
HONCHO_DOCS_URL = "https://docs.honcho.dev/v3/guides/integrations/hermes"

UTM_SOURCE = "hermes"
UTM_MEDIUM = "plugin"


def attribution_params(placement: str) -> dict[str, str]:
    """The UTM parameters for a link shown at ``placement`` (e.g. ``"oauth"``, ``"apikey-prompt"``)."""
    return {"utm_source": UTM_SOURCE, "utm_medium": UTM_MEDIUM, "utm_content": placement}


def attribution_url(url: str, placement: str) -> str:
    """``url`` with the Hermes UTM parameters appended.

    Existing query parameters are kept in order, and a ``utm_*`` key the URL already carries wins, so
    an already-tagged link is never re-attributed. Never raises: a URL that can't be parsed comes
    back unchanged, because a link with no attribution beats a setup step that crashes.
    """
    try:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            return url
        query = parse_qsl(parts.query, keep_blank_values=True)
        present = {key for key, _ in query}
        extra = [(k, v) for k, v in attribution_params(placement).items() if k not in present and v]
        if not extra:
            return url
        # Append to the original query bytes rather than re-encoding them: a server-supplied link
        # (the device verification URL) may be signed or otherwise byte-sensitive.
        tagged = f"{parts.query}&{urlencode(extra)}" if parts.query else urlencode(extra)
        return urlunsplit(parts._replace(path=parts.path or "/", query=tagged))
    except Exception:
        return url
