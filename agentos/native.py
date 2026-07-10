"""Stable public native-capability inspection API without raw extension access."""

from kernel.native_core import (
    NativeCoreImportError,
    NativeCoreUnavailableError,
    RuntimeCapabilities,
    get_runtime_capabilities,
    native_core_available,
    require_native_core,
)

__all__ = [
    "NativeCoreImportError",
    "NativeCoreUnavailableError",
    "RuntimeCapabilities",
    "get_runtime_capabilities",
    "native_core_available",
    "require_native_core",
]
