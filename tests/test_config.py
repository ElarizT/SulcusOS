from __future__ import annotations

from pathlib import Path

import pytest

from agentos.config import ConfigError, SulcusConfig, discover_config, load_config, resolve_config


VALID = """[sulcus]
execution_mode = "parallel"
require_tool_approval = true

[limits]
max_tool_calls_per_loop = 10
max_tool_calls_per_round = 4
tool_timeout_ms = 5000

[llm]
provider = "deterministic"
model = "local-default"
"""


def write_config(directory: Path, content: str = VALID) -> Path:
    path = directory / "sulcus.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_no_file_preserves_existing_defaults(tmp_path: Path) -> None:
    assert discover_config(tmp_path) is None
    assert load_config(cwd=tmp_path) == SulcusConfig()


def test_valid_file_is_discovered_and_typed(tmp_path: Path) -> None:
    path = write_config(tmp_path)
    config = load_config(cwd=tmp_path)
    assert config.source_path == path.resolve()
    assert config.runtime.execution_mode == "parallel"
    assert config.runtime.require_tool_approval is True
    assert config.limits.max_tool_calls_per_round == 4
    assert config.llm.model == "local-default"


def test_precedence_is_explicit_then_environment_then_file(tmp_path: Path) -> None:
    loaded = load_config(write_config(tmp_path))
    resolved = resolve_config(
        loaded,
        environ={"SULCUS_EXECUTION_MODE": "sequential", "SULCUS_LLM_MODEL": "env-model"},
        explicit={"execution_mode": "parallel", "model": "explicit-model"},
    )
    assert resolved.runtime.execution_mode == "parallel"
    assert resolved.llm.model == "explicit-model"
    assert resolved.limits.max_tool_calls_per_loop == 10


def test_existing_agentos_llm_environment_aliases_are_preserved() -> None:
    resolved = resolve_config(environ={"AGENTOS_LLM_PROVIDER": "openrouter", "AGENTOS_LLM_MODEL": "existing-model"})
    assert resolved.llm.provider == "openrouter"
    assert resolved.llm.model == "existing-model"


@pytest.mark.parametrize(
    ("content", "field"),
    [
        ("[sulcus]\nexecution_mode = 'fast'\n", "sulcus.execution_mode"),
        ("[sulcus]\nrequire_tool_approval = 'yes'\n", "sulcus.require_tool_approval"),
        ("[limits]\ntool_timeout_ms = -1\n", "limits.tool_timeout_ms"),
        ("[llm]\nprovider = 'mystery'\n", "llm.provider"),
        ("[unknown]\nvalue = 1\n", "unknown"),
        ("[limits]\nextra = 1\n", "limits.extra"),
    ],
)
def test_invalid_file_reports_path_and_field(tmp_path: Path, content: str, field: str) -> None:
    path = write_config(tmp_path, content)
    with pytest.raises(ConfigError) as caught:
        load_config(path)
    assert str(path.resolve()) in str(caught.value)
    assert field in str(caught.value)


def test_environment_validation_is_actionable() -> None:
    with pytest.raises(ConfigError, match="SULCUS_REQUIRE_TOOL_APPROVAL"):
        resolve_config(environ={"SULCUS_REQUIRE_TOOL_APPROVAL": "sometimes"})


def test_sanitized_output_has_no_environment_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTOS_LLM_API_KEY", "SECRET_VALUE")
    shown = resolve_config(environ=dict(__import__("os").environ)).sanitized()
    assert "SECRET_VALUE" not in repr(shown)
    assert "api_key" not in repr(shown).casefold()

