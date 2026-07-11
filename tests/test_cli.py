from __future__ import annotations

import subprocess
import sys

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
