from __future__ import annotations

import pytest
from pytest import MonkeyPatch

from promptlab.config import Settings, _parse_optional_bool


def test_parse_optional_bool() -> None:
    assert _parse_optional_bool(None) is None
    assert _parse_optional_bool("") is None
    assert _parse_optional_bool("true") is True
    assert _parse_optional_bool("FALSE") is False
    with pytest.raises(ValueError):
        _parse_optional_bool("maybe")


def test_qwen_think_defaults_false_and_mistral_omits(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODEL_A_THINK", raising=False)
    monkeypatch.delenv("MODEL_B_THINK", raising=False)
    settings = Settings.from_env()
    assert settings.models["mistral"].think is None
    assert settings.models["qwen"].think is False


def test_think_env_can_enable_qwen(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_B_THINK", "true")
    assert Settings.from_env().models["qwen"].think is True
