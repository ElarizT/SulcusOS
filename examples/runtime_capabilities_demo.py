"""Print Sulcus OS capability availability without requiring native bindings."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agentos.native import get_runtime_capabilities


def main() -> int:
    capabilities = get_runtime_capabilities()
    print("Sulcus OS Runtime Capabilities")
    print("Python runtime: available")
    print(f"Native core: {'available' if capabilities.native_core_available else 'unavailable'}")
    print(
        "Full dashboard runtime: "
        f"{'available' if capabilities.native_core_available else 'unavailable'}"
    )
    print("LLM/tool runtime: available")
    if not capabilities.native_core_available:
        print("Install native core with: maturin develop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
