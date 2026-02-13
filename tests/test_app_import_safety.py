from pathlib import Path


def test_app_does_not_purge_utils_modules_during_startup():
    """Guard against reintroducing runtime sys.modules purges that caused Streamlit import KeyError."""
    app_text = Path('simulation_app/app.py').read_text(encoding='utf-8')

    assert 'del sys.modules[_k]' not in app_text
    assert '_stale_keys = [k for k in sys.modules' not in app_text
