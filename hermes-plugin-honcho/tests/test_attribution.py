"""UTM attribution on every Honcho web link the plugin shows or opens."""

import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from hermes_plugin_honcho.attribution import HONCHO_APP_URL, HONCHO_DOCS_URL, attribution_url


def _q(url: str) -> dict[str, list[str]]:
    return parse_qs(urlparse(url).query)


def test_bare_app_link_gets_the_three_hermes_params():
    url = attribution_url(HONCHO_APP_URL, "apikey-prompt")
    assert url == "https://app.honcho.dev/?utm_source=hermes&utm_medium=plugin&utm_content=apikey-prompt"


def test_existing_query_is_kept_in_order():
    url = attribution_url("https://app.honcho.dev/device?user_code=ABCD-EFGH", "device")
    assert url.startswith("https://app.honcho.dev/device?user_code=ABCD-EFGH&utm_source=hermes")
    assert _q(url)["user_code"] == ["ABCD-EFGH"]


def test_existing_utm_values_win():
    url = attribution_url("https://app.honcho.dev/?utm_source=newsletter", "oauth")
    q = _q(url)
    assert q["utm_source"] == ["newsletter"]
    assert q["utm_medium"] == ["plugin"] and q["utm_content"] == ["oauth"]



def test_existing_query_bytes_are_not_reencoded():
    # A server-supplied link may be signed; %20 must not become + and order must not change.
    url = attribution_url("https://app.honcho.dev/device?b=x%20y&a=1&sig=abc%2F", "device")
    assert url == "https://app.honcho.dev/device?b=x%20y&a=1&sig=abc%2F&utm_source=hermes&utm_medium=plugin&utm_content=device"

def test_fully_tagged_link_is_returned_unchanged():
    tagged = attribution_url(HONCHO_APP_URL, "walkthrough")
    assert attribution_url(tagged, "something-else") == tagged


@pytest.mark.parametrize("url", ["", "not a url", "app.honcho.dev", "mailto:hello@honcho.dev"])
def test_non_web_or_malformed_input_comes_back_unchanged(url):
    assert attribution_url(url, "x") == url


def test_config_panel_docs_link_matches_the_helper():
    # config_schema.py can't import the helper (Hermes loads it standalone), so it hardcodes the URL.
    from hermes_plugin_honcho.config_schema import CONFIG_SCHEMA

    assert CONFIG_SCHEMA.docs_url == attribution_url(HONCHO_DOCS_URL, "config-panel")


def test_config_schema_file_stays_import_free_of_the_package():
    source = Path(__file__).resolve().parent.parent.joinpath("config_schema.py").read_text()
    assert "from ." not in source and "import attribution" not in source


def test_no_untagged_app_links_left_in_user_facing_code():
    plugin_dir = Path(__file__).resolve().parent.parent
    allowed = {"attribution.py", "oauth_flow.py"}  # the constant itself; the OAuth dashboard origin
    app_link = re.compile(re.escape(HONCHO_APP_URL))
    offenders = [
        f"{path.name}:{n}"
        for path in sorted(plugin_dir.glob("*.py")) if path.name not in allowed
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if app_link.search(line)
    ]
    assert offenders == []
