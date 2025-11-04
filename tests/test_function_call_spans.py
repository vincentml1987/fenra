import sys
import types

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")
    requests_stub.get = lambda *a, **k: None
    requests_stub.post = lambda *a, **k: None
    class _StubResponse:  # pragma: no cover - minimal stub for typing
        pass

    requests_stub.Response = _StubResponse
    sys.modules["requests"] = requests_stub

if "fenra_ui" not in sys.modules:
    fenra_ui_stub = types.ModuleType("fenra_ui")

    class _StubFenraUI:  # pragma: no cover - minimal stub for import
        def __init__(self, *args, **kwargs):  # noqa: D401
            pass

    fenra_ui_stub.FenraUI = _StubFenraUI
    sys.modules["fenra_ui"] = fenra_ui_stub

import conductor


def test_extract_allows_internal_whitespace():
    text = '~rename_agent("Agent Prime", reason="Needs rename")~'
    assert conductor._extract_pwsh_commands(text) == [
        'rename_agent("Agent Prime", reason="Needs rename")'
    ]


def test_extract_rejects_whitespace_touching_markers():
    text = '~ rename_agent("Agent")~ and ~rename_agent("Agent") ~'
    assert conductor._extract_pwsh_commands(text) == []


def test_extract_rejects_newline_directly_after_open():
    text = '~\nrename_agent("Agent")~'
    assert conductor._extract_pwsh_commands(text) == []


def test_extract_rejects_newline_directly_before_close():
    text = '~rename_agent("Agent")\n~'
    assert conductor._extract_pwsh_commands(text) == []


def test_extract_allows_newlines_inside_span():
    text = '~rename_agent("Agent" + "\nPrime")~'
    assert conductor._extract_pwsh_commands(text) == [
        'rename_agent("Agent" + "\nPrime")'
    ]
