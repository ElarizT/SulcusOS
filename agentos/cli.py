"""Small Python-only diagnostic command for installed Sulcus OS packages."""

from __future__ import annotations

from agentos._version import __version__
from agentos.native import native_core_available


def runtime_check_main() -> int:
    """Print install/runtime capability status without requiring Rust bindings."""
    native_available = native_core_available()
    print(f"Sulcus OS {__version__}")
    print("Python runtime: available")
    print(f"Native core: {'available' if native_available else 'unavailable'}")
    if not native_available:
        print("Install native core for dashboard features with: maturin develop")
    return 0


if __name__ == "__main__":
    raise SystemExit(runtime_check_main())
