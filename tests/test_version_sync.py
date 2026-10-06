"""Guard: every version location listed in CLAUDE.md must carry the same version."""
import importlib.util
import os

_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "check_version_sync.py")


def _load():
    spec = importlib.util.spec_from_file_location("check_version_sync", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_version_locations_in_sync(capsys):
    mod = _load()
    rc = mod.main()
    out = capsys.readouterr().out
    assert rc == 0, f"version locations out of sync:\n{out}"


def test_version_sync_detects_mismatch(monkeypatch, capsys):
    """The checker itself must be able to fail."""
    mod = _load()
    real = mod.read_versions
    monkeypatch.setattr(
        mod, "read_versions",
        lambda: [(lbl, ("9.9.9.9" if i == 0 else v), n) for i, (lbl, v, n) in enumerate(real())],
    )
    assert mod.main() == 1
    assert "VERSION MISMATCH" in capsys.readouterr().out
