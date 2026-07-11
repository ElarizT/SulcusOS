"""Typed project configuration for Sulcus' existing Python runtime features."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
    import tomli as tomllib  # type: ignore[no-redef]


CONFIG_FILENAME = "sulcus.toml"
EXECUTION_MODES = frozenset(("sequential", "parallel"))
SUPPORTED_PROVIDERS = frozenset(("deterministic", "openai", "openrouter", "groq"))
_TOP_LEVEL_KEYS = frozenset(("sulcus", "limits", "llm"))
_SECTION_KEYS = {
    "sulcus": frozenset(("execution_mode", "require_tool_approval")),
    "limits": frozenset(("max_tool_calls_per_loop", "max_tool_calls_per_round", "tool_timeout_ms")),
    "llm": frozenset(("provider", "model")),
}


class ConfigError(ValueError):
    """Actionable configuration error safe to show in a CLI."""

    def __init__(self, path: Path | None, field: str, message: str) -> None:
        location = str(path) if path is not None else "configuration"
        super().__init__(f"{location}: invalid field '{field}': {message}")
        self.path = path
        self.field = field


@dataclass(frozen=True)
class RuntimeSettings:
    execution_mode: str = "sequential"
    require_tool_approval: bool = False


@dataclass(frozen=True)
class ResourceLimitSettings:
    max_tool_calls_per_loop: int | None = None
    max_tool_calls_per_round: int | None = None
    tool_timeout_ms: int | None = None


@dataclass(frozen=True)
class LLMSettings:
    provider: str = "deterministic"
    model: str = "default"


@dataclass(frozen=True)
class SulcusConfig:
    runtime: RuntimeSettings = RuntimeSettings()
    limits: ResourceLimitSettings = ResourceLimitSettings()
    llm: LLMSettings = LLMSettings()
    source_path: Path | None = None

    def sanitized(self) -> dict[str, object]:
        """Return the effective non-secret values suitable for display."""
        return {
            "sulcus": {
                "execution_mode": self.runtime.execution_mode,
                "require_tool_approval": self.runtime.require_tool_approval,
            },
            "limits": {
                "max_tool_calls_per_loop": self.limits.max_tool_calls_per_loop,
                "max_tool_calls_per_round": self.limits.max_tool_calls_per_round,
                "tool_timeout_ms": self.limits.tool_timeout_ms,
            },
            "llm": {"provider": self.llm.provider, "model": self.llm.model},
        }


def discover_config(cwd: str | Path | None = None) -> Path | None:
    """Return ``sulcus.toml`` in the selected/current directory, if present."""
    candidate = Path(cwd or Path.cwd()).resolve() / CONFIG_FILENAME
    return candidate if candidate.is_file() else None


def load_config(path: str | Path | None = None, *, cwd: str | Path | None = None) -> SulcusConfig:
    """Load and validate a TOML file, or return unchanged runtime defaults."""
    selected = Path(path).resolve() if path is not None else discover_config(cwd)
    if selected is None:
        return SulcusConfig()
    try:
        with selected.open("rb") as stream:
            raw = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(selected, "toml", str(exc)) from None
    if not isinstance(raw, dict):
        raise ConfigError(selected, "toml", "root must be a table")
    _reject_unknown(raw, selected)
    sulcus = _section(raw, "sulcus", selected)
    limits = _section(raw, "limits", selected)
    llm = _section(raw, "llm", selected)
    runtime = RuntimeSettings(
        execution_mode=_execution_mode(sulcus.get("execution_mode", "sequential"), selected, "sulcus.execution_mode"),
        require_tool_approval=_boolean(sulcus.get("require_tool_approval", False), selected, "sulcus.require_tool_approval"),
    )
    resource_limits = ResourceLimitSettings(**{
        name: _limit(limits.get(name), selected, f"limits.{name}")
        for name in _SECTION_KEYS["limits"]
    })
    provider = _provider(llm.get("provider", "deterministic"), selected, "llm.provider")
    model = _nonempty_string(llm.get("model", "default"), selected, "llm.model")
    return SulcusConfig(runtime, resource_limits, LLMSettings(provider, model), selected)


def resolve_config(
    config: SulcusConfig | None = None,
    *,
    explicit: Mapping[str, object | None] | None = None,
    environ: Mapping[str, str] | None = None,
) -> SulcusConfig:
    """Apply environment then explicit values over file/default configuration."""
    base = config or SulcusConfig()
    env = os.environ if environ is None else environ
    runtime = base.runtime
    limits = base.limits
    llm = base.llm

    mode = env.get("SULCUS_EXECUTION_MODE")
    approval = env.get("SULCUS_REQUIRE_TOOL_APPROVAL")
    if mode is not None:
        runtime = replace(runtime, execution_mode=_execution_mode(mode, None, "SULCUS_EXECUTION_MODE"))
    if approval is not None:
        runtime = replace(runtime, require_tool_approval=_environment_boolean(approval, "SULCUS_REQUIRE_TOOL_APPROVAL"))
    for field, variable in (
        ("max_tool_calls_per_loop", "SULCUS_MAX_TOOL_CALLS_PER_LOOP"),
        ("max_tool_calls_per_round", "SULCUS_MAX_TOOL_CALLS_PER_ROUND"),
        ("tool_timeout_ms", "SULCUS_TOOL_TIMEOUT_MS"),
    ):
        if variable in env:
            limits = replace(limits, **{field: _environment_limit(env[variable], variable)})
    provider = env.get("SULCUS_LLM_PROVIDER", env.get("AGENTOS_LLM_PROVIDER"))
    model = env.get("SULCUS_LLM_MODEL", env.get("AGENTOS_LLM_MODEL"))
    if provider is not None:
        llm = replace(llm, provider=_provider(provider, None, "SULCUS_LLM_PROVIDER"))
    if model is not None:
        llm = replace(llm, model=_nonempty_string(model, None, "SULCUS_LLM_MODEL"))

    values = {key: value for key, value in (explicit or {}).items() if value is not None}
    allowed = {"execution_mode", "require_tool_approval", "max_tool_calls_per_loop", "max_tool_calls_per_round", "tool_timeout_ms", "provider", "model"}
    unknown = set(values) - allowed
    if unknown:
        raise ConfigError(None, sorted(unknown)[0], "unknown explicit setting")
    if "execution_mode" in values:
        runtime = replace(runtime, execution_mode=_execution_mode(values["execution_mode"], None, "execution_mode"))
    if "require_tool_approval" in values:
        runtime = replace(runtime, require_tool_approval=_boolean(values["require_tool_approval"], None, "require_tool_approval"))
    for field in ("max_tool_calls_per_loop", "max_tool_calls_per_round", "tool_timeout_ms"):
        if field in values:
            limits = replace(limits, **{field: _limit(values[field], None, field)})
    if "provider" in values:
        llm = replace(llm, provider=_provider(values["provider"], None, "provider"))
    if "model" in values:
        llm = replace(llm, model=_nonempty_string(values["model"], None, "model"))
    return SulcusConfig(runtime, limits, llm, base.source_path)


def _reject_unknown(raw: Mapping[str, Any], path: Path) -> None:
    for key in raw:
        if key not in _TOP_LEVEL_KEYS:
            raise ConfigError(path, str(key), "unknown top-level section")
    for section in _TOP_LEVEL_KEYS:
        value = raw.get(section, {})
        if not isinstance(value, Mapping):
            raise ConfigError(path, section, "must be a TOML table")
        for key in value:
            if key not in _SECTION_KEYS[section]:
                raise ConfigError(path, f"{section}.{key}", "unknown key")


def _section(raw: Mapping[str, Any], name: str, path: Path) -> Mapping[str, Any]:
    value = raw.get(name, {})
    assert isinstance(value, Mapping)
    return value


def _execution_mode(value: object, path: Path | None, field: str) -> str:
    if not isinstance(value, str) or value not in EXECUTION_MODES:
        raise ConfigError(path, field, "must be 'sequential' or 'parallel'")
    return value


def _boolean(value: object, path: Path | None, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(path, field, "must be true or false")
    return value


def _environment_boolean(value: str, field: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ConfigError(None, field, "must be true/false, 1/0, yes/no, or on/off")


def _limit(value: object, path: Path | None, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(path, field, "must be a non-negative integer")
    return value


def _environment_limit(value: str, field: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise ConfigError(None, field, "must be a non-negative integer") from None
    result = _limit(parsed, None, field)
    assert result is not None
    return result


def _provider(value: object, path: Path | None, field: str) -> str:
    provider = _nonempty_string(value, path, field).casefold()
    if provider not in SUPPORTED_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
        raise ConfigError(path, field, f"unsupported provider; choose one of: {supported}")
    return provider


def _nonempty_string(value: object, path: Path | None, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(path, field, "must be a non-empty string")
    return value.strip()


__all__ = [
    "ConfigError", "LLMSettings", "ResourceLimitSettings", "RuntimeSettings",
    "SulcusConfig", "discover_config", "load_config", "resolve_config",
]
