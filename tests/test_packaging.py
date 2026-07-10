from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

import agentos
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_project_metadata_declares_python_only_setuptools_build() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'build-backend = "setuptools.build_meta"' in pyproject
    assert 'name = "sulcus-os"' in pyproject
    assert 'version = {attr = "agentos._version.__version__"}' in pyproject
    assert 'requires-python = ">=3.10"' in pyproject
    build_system = pyproject.split("[build-system]", 1)[1].split("[project]", 1)[0]
    assert '"setuptools>=77"' in build_system
    assert '"maturin>=1.5"' not in build_system


def test_package_metadata_matches_public_version_when_installed() -> None:
    try:
        distribution_version = importlib.metadata.version("sulcus-os")
        metadata = importlib.metadata.metadata("sulcus-os")
        distribution_name = metadata["Name"]
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("distribution metadata is available after editable or wheel installation")
    assert distribution_name == "sulcus-os"
    assert distribution_version == agentos.__version__
    assert metadata["License-Expression"] == "LicenseRef-Unlicensed"


def test_manifest_and_package_discovery_exclude_development_junk() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "recursive-include agentos *.py" in manifest
    assert "recursive-include kernel *.py" in manifest
    assert "global-exclude *.log" in manifest
    assert "prune tests" in manifest
    assert "prune examples" in manifest


def test_core_public_imports_remain_python_only() -> None:
    from agentos.llm import DeterministicLLMProvider
    from agentos.native import native_core_available
    from agentos.runtime import AgentToolLoop
    from agentos.tools import ToolRegistry, ToolRuntime

    assert all((DeterministicLLMProvider, AgentToolLoop, ToolRegistry, ToolRuntime))
    assert isinstance(native_core_available(), bool)
    assert sys.version_info >= (3, 10)
