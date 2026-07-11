from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from agentos import __version__
from agentos.cli import main, runtime_check_main


def test_version_uses_single_version_source(capsys) -> None:
    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    assert capsys.readouterr().out.strip() == __version__


def test_check_is_successful_without_native_core(capsys) -> None:
    assert main(["check"]) == 0
    output = capsys.readouterr().out
    assert f"Sulcus OS {__version__}" in output
    assert "Python runtime: available" in output
    assert "Native core:" in output
    assert "Dashboard:" in output
    assert "Python LLM/tool runtime: available" in output


def test_compatibility_check_entry_point(capsys) -> None:
    assert runtime_check_main() == 0
    assert "Sulcus OS" in capsys.readouterr().out


def test_demo_options_map_to_existing_callable(capsys) -> None:
    assert main(["demo", "research-team", "--parallel", "--tight-limits"]) == 0
    output = capsys.readouterr().out
    assert "Supervised Research Team" in output
    assert "resource_denials=1" in output
    assert "DENIED" in output


def test_demo_approval_maps_to_simulated_publish(capsys) -> None:
    assert main(["demo", "research-team", "--approve-publish"]) == 0
    assert "APPROVED" in capsys.readouterr().out


def test_conflicting_options_are_clear_usage_errors_without_traceback() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "agentos.cli", "demo", "research-team", "--parallel", "--sequential"],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 2
    assert "not allowed with argument" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_unknown_demo_is_usage_error_without_traceback() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "agentos.cli", "demo", "unknown"],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 2
    assert "invalid choice" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_config_commands_without_file(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["config", "path"]) == 0
    assert "No sulcus.toml" in capsys.readouterr().out
    assert main(["config", "check"]) == 0
    assert "defaults" in capsys.readouterr().out
    assert main(["config", "show"]) == 0
    assert '"execution_mode": "sequential"' in capsys.readouterr().out


def test_config_check_returns_one_for_invalid_file(monkeypatch, tmp_path: Path, capsys) -> None:
    path = tmp_path / "sulcus.toml"
    path.write_text("[limits]\ntool_timeout_ms = -2\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert main(["config", "check"]) == 1
    output = capsys.readouterr().out
    assert str(path) in output
    assert "limits.tool_timeout_ms" in output
    assert "Traceback" not in output


def test_demo_explicit_mode_overrides_environment_and_file(monkeypatch, tmp_path: Path, capsys) -> None:
    (tmp_path / "sulcus.toml").write_text("[sulcus]\nexecution_mode = 'parallel'\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SULCUS_EXECUTION_MODE", "parallel")
    assert main(["demo", "research-team", "--sequential"]) == 0
    assert "FINAL REPORT" in capsys.readouterr().out
