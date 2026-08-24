import importlib


def test_app_modules_import_cleanly():
    import config
    import main

    assert config is not None
    assert main is not None
