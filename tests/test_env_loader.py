import os

from src.env_loader import load_env_file


def test_load_env_file_sets_values_without_overriding_existing_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "\n".join(
            [
                "# Local settings",
                "ORDER_API_BASE_URL=http://localhost:5001",
                "LOG_LEVEL=DEBUG",
                "export OPENROUTER_BASE_URL='https://openrouter.ai/api/v1'",
                'OPENROUTER_API_KEY="test-key"',
            ]
        )
    )
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.delenv("ORDER_API_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    load_env_file(str(env_file))

    assert os.environ["ORDER_API_BASE_URL"] == "http://localhost:5001"
    assert os.environ["LOG_LEVEL"] == "INFO"
    assert os.environ["OPENROUTER_BASE_URL"] == "https://openrouter.ai/api/v1"
    assert os.environ["OPENROUTER_API_KEY"] == "test-key"


def test_load_env_file_can_override_existing_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env.local"
    env_file.write_text("LOG_LEVEL=DEBUG # local debug logging")
    monkeypatch.setenv("LOG_LEVEL", "INFO")

    load_env_file(str(env_file), override=True)

    assert os.environ["LOG_LEVEL"] == "DEBUG"
