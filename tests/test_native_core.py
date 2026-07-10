from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest

import kernel.native_core as native_core


def load_without_native(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setitem(sys.modules, "agent_os_core", None)
    return importlib.reload(native_core)


def restore_native() -> None:
    importlib.reload(native_core)


def test_compatibility_boundary_imports_without_native(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        module = load_without_native(monkeypatch)
        assert module.NATIVE_CORE_AVAILABLE is False
        assert module.native_core_available() is False
    finally:
        monkeypatch.undo()
        restore_native()


def test_missing_native_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        module = load_without_native(monkeypatch)
        with pytest.raises(module.NativeCoreUnavailableError) as error:
            module.require_native_core("full dashboard runtime")
        text = str(error.value)
        assert "full dashboard runtime" in text
        assert "maturin develop" in text
        assert "Python-only" in text
    finally:
        monkeypatch.undo()
        restore_native()


def test_capability_report_reflects_missing_native(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        module = load_without_native(monkeypatch)
        capabilities = module.get_runtime_capabilities()
        assert capabilities.python_runtime_available is True
        assert capabilities.native_core_available is False
        assert capabilities.wasm_available is False
    finally:
        monkeypatch.undo()
        restore_native()


def test_capability_report_uses_available_native_module(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = ModuleType("agent_os_core")
    fake.AgentMessage = object
    monkeypatch.setitem(sys.modules, "agent_os_core", fake)
    try:
        module = importlib.reload(native_core)
        assert module.native_core_available() is True
        assert module.require_native_core("test") is fake
        assert module.get_runtime_capabilities().wasm_available is True
    finally:
        monkeypatch.undo()
        restore_native()


def test_main_import_is_safe_without_native(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        load_without_native(monkeypatch)
        sys.modules.pop("main", None)
        main = importlib.import_module("main")
        assert callable(main.main)
        assert callable(main.format_external_agent_run)
    finally:
        monkeypatch.undo()
        restore_native()
