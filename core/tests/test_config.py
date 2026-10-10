import stat

from earmark.config import Env, SettingsStore, UserSettings


def test_home_is_expanded_in_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EARMARK_DATA_DIR", "~/earmark-data")

    assert Env().data_dir == tmp_path / "earmark-data"


def test_token_is_created_once_private_and_reused(tmp_path):
    env = Env(data_dir=tmp_path / "data")

    token = env.resolve_api_token()

    assert len(token) >= 32
    assert Env(data_dir=tmp_path / "data").resolve_api_token() == token
    assert stat.S_IMODE(env.token_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(env.data_dir.stat().st_mode) == 0o700
    assert Env(data_dir=tmp_path / "data", api_token="fixed").resolve_api_token() == "fixed"


def test_settings_are_persisted_and_partial_updates_keep_the_rest(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    assert store.load() == UserSettings()

    store.update({"notes_folder": "Study", "notes_grouping": "session"})
    store.update({"llm_provider": "anthropic"})

    loaded = store.load()
    assert (loaded.notes_folder, loaded.notes_grouping, loaded.llm_provider) == (
        "Study",
        "session",
        "anthropic",
    )
    assert loaded.reminders_list == "Earmark"
