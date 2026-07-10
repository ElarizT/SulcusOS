"""Compatibility boundary for the optional Rust ``agent_os_core`` extension."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType


class NativeCoreUnavailableError(RuntimeError):
    """Raised only when a caller invokes a native-only Sulcus OS capability."""


class NativeCoreImportError(RuntimeError):
    """The installed native extension failed to load and needs repair."""


try:
    import agent_os_core as _native_core
except ModuleNotFoundError as exc:
    # Do not hide ImportError/ModuleNotFoundError raised *inside* a present
    # extension: only the extension module itself is optional.
    if exc.name != "agent_os_core":
        raise
    _native_core: ModuleType | None = None
except Exception as exc:
    raise NativeCoreImportError(
        "Sulcus OS native core was found but could not be imported. "
        "Check its binary/dependency installation and rerun `maturin develop`."
    ) from exc


NATIVE_CORE_AVAILABLE = _native_core is not None


@dataclass(frozen=True)
class RuntimeCapabilities:
    """Small dependency-free report of runtime capabilities available now."""

    native_core_available: bool
    python_runtime_available: bool = True
    wasm_available: bool | None = None


def native_core_available() -> bool:
    """Return whether the Rust extension was successfully imported."""
    return NATIVE_CORE_AVAILABLE


def get_runtime_capabilities() -> RuntimeCapabilities:
    """Return the currently available Python and native runtime capabilities."""
    return RuntimeCapabilities(
        native_core_available=NATIVE_CORE_AVAILABLE,
        wasm_available=NATIVE_CORE_AVAILABLE,
    )


def require_native_core(feature: str | None = None) -> ModuleType:
    """Return the native module or raise an actionable capability error."""
    if _native_core is not None:
        return _native_core
    feature_text = f"The {feature} requires `agent_os_core`." if feature else "This capability requires `agent_os_core`."
    raise NativeCoreUnavailableError(
        "Sulcus OS native core is not installed.\n"
        f"{feature_text}\n\n"
        "From the project root, run:\n"
        "    maturin develop\n\n"
        "Python-only LLM, tool-runtime, and agent-tool-loop components remain available."
    )
