"""Cross-language wire fixtures: every ``<kind>.<lang>.bin`` present must decode
to exactly the values in ``values.json``; the Python files must be present and
current. Event fixtures compare in the shared JSON shape (``Event.as_dict``)."""

import os

import pytest
from astrum_client.wire import decode_command, decode_event, decode_telemetry, decode_tick_end
from astrum_client.wire_fixture import (
    COMMAND_KINDS,
    EVENT_KINDS,
    KINDS,
    LANGS,
    default_fixtures_dir,
    encode_kind,
    expected_command,
    expected_telemetry,
    expected_tick_end,
    load_values,
)

FIXTURES = default_fixtures_dir()
VALUES = load_values(FIXTURES)


def _path(kind, lang):
    return os.path.join(FIXTURES, f"{kind}.{lang}.bin")


def _read(kind, lang):
    path = _path(kind, lang)
    if not os.path.exists(path):
        if lang == "python":
            pytest.fail(f"{path} missing: run `python -m astrum_client.wire_fixture`")
        pytest.skip(f"{os.path.basename(path)} not generated in this checkout")
    with open(path, "rb") as f:
        return f.read()


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("kind", KINDS)
def test_fixture_decodes_to_values(kind, lang):
    payload = _read(kind, lang)
    if kind == "telemetry":
        assert decode_telemetry(payload) == expected_telemetry(VALUES[kind])
    elif kind == "tick_end":
        assert decode_tick_end(payload) == expected_tick_end(VALUES[kind])
    else:
        assert kind in COMMAND_KINDS
        assert decode_command(payload) == expected_command(VALUES[kind])


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("kind", EVENT_KINDS)
def test_event_fixture_decodes_to_values(kind, lang):
    assert decode_event(_read(kind, lang)).as_dict() == VALUES[kind]


@pytest.mark.parametrize("kind", KINDS + EVENT_KINDS)
def test_python_fixture_is_current(kind):
    with open(_path(kind, "python"), "rb") as f:
        assert f.read() == encode_kind(kind, VALUES)
