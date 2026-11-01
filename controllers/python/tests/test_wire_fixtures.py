"""Cross-language wire fixtures: every ``<kind>.<lang>.bin`` present must decode
to exactly the values in ``values.json``; the Python files must be present and
current."""

import os

import pytest
from astrum_client.wire import decode_command, decode_telemetry
from astrum_client.wire_fixture import (
    KINDS,
    LANGS,
    default_fixtures_dir,
    encode_kind,
    expected_command,
    expected_telemetry,
    load_values,
)

FIXTURES = default_fixtures_dir()
VALUES = load_values(FIXTURES)


def _path(kind, lang):
    return os.path.join(FIXTURES, f"{kind}.{lang}.bin")


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("kind", KINDS)
def test_fixture_decodes_to_values(schema, kind, lang):
    path = _path(kind, lang)
    if not os.path.exists(path):
        if lang == "python":
            pytest.fail(f"{path} missing: run `python -m astrum_client.wire_fixture`")
        pytest.skip(f"{os.path.basename(path)} not generated in this checkout")
    with open(path, "rb") as f:
        payload = f.read()
    if kind == "telemetry":
        assert decode_telemetry(schema, payload) == expected_telemetry(VALUES[kind])
    else:
        assert decode_command(schema, payload) == expected_command(VALUES[kind])


@pytest.mark.parametrize("kind", KINDS)
def test_python_fixture_is_current(schema, kind):
    with open(_path(kind, "python"), "rb") as f:
        assert f.read() == encode_kind(schema, kind, VALUES)
