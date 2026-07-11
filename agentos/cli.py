"""Dependency-free command line interface for installed Sulcus packages."""

from __future__ import annotations

from agentos._version import __version__
import argparse
from importlib.util import find_spec
from typing import Sequence

from agentos.native import native_core_available


def _module_available(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def runtime_check_main() -> int:
    """Print capability status without requiring Rust bindings."""
    native_available = native_core_available()
    print(f"Sulcus OS {__version__}")
    print("Python runtime: available")
    print(f"Native core: {'available' if native_available else 'unavailable'}")
    dashboard_available = native_available and _module_available("textual")
    print(f"Dashboard: {'available' if dashboard_available else 'unavailable'}")
    print("Python LLM/tool runtime: available")
    if not native_available:
        print("Native features are optional. For local native development, run: maturin develop")
    if not _module_available("textual"):
        print("For dashboard Python dependencies, install: pip install 'sulcus-os[dashboard]'")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sulcus",
        description="Sulcus diagnostics and offline agent-runtime demos.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Report installed runtime capabilities.")

    demo = commands.add_parser("demo", help="Run a bundled demonstration.")
    demos = demo.add_subparsers(dest="demo_name", required=True)
    research = demos.add_parser(
        "research-team",
        help="Run the offline Supervised Research Team flagship demo.",
    )
    research.add_argument("--topic", help="Research topic (uses the demo default when omitted).")
    mode = research.add_mutually_exclusive_group()
    mode.add_argument("--parallel", action="store_true", help="Execute safe research tools in parallel.")
    mode.add_argument("--sequential", action="store_true", help="Execute research tools sequentially (default).")
    publication = research.add_mutually_exclusive_group()
    publication.add_argument("--approve-publish", action="store_true", help="Approve simulated publication.")
    publication.add_argument("--deny-publish", action="store_true", help="Deny simulated publication (default).")
    research.add_argument("--tight-limits", action="store_true", help="Demonstrate one safe resource denial.")
    research.add_argument("--show-timeline", action="store_true", help="Print the safe runtime timeline.")
    return parser


def _research_demo_args(args: argparse.Namespace) -> list[str]:
    forwarded: list[str] = []
    if args.topic is not None:
        forwarded.extend(("--topic", args.topic))
    if args.parallel:
        forwarded.extend(("--execution-mode", "parallel"))
    elif args.sequential:
        forwarded.extend(("--execution-mode", "sequential"))
    if args.approve_publish:
        forwarded.append("--approve-publish")
    elif args.deny_publish:
        forwarded.append("--deny-publish")
    if args.tight_limits:
        forwarded.append("--tight-limits")
    if args.show_timeline:
        forwarded.append("--show-timeline")
    return forwarded


def main(argv: Sequence[str] | None = None) -> int:
    """Run the installed ``sulcus`` command."""
    args = _build_parser().parse_args(argv)
    if args.command == "check":
        return runtime_check_main()
    if args.command == "demo" and args.demo_name == "research-team":
        try:
            from examples.supervised_research_team.demo import main as demo_main
        except ImportError:
            print("error: the research-team demo is missing; reinstall the Sulcus package")
            return 1
        try:
            return demo_main(_research_demo_args(args))
        except (RuntimeError, ValueError) as exc:
            print(f"error: research-team demo failed: {exc}")
            return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
