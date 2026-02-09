"""
Conventional Commit lint tool.

This tool validates commit messages against the Conventional Commits format.
It is intended to be used from pre-commit's commit-msg hook.

Usage:
    # pre-commit hook mode
    python -m devops_scripts.commitlint.conventional_commit_lint hook .git/COMMIT_EDITMSG

    # ad-hoc check mode
    python -m devops_scripts.commitlint.conventional_commit_lint check "feat(api): add search endpoint"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ALLOWED_TYPES = (
    "feat",
    "fix",
    "docs",
    "style",
    "refactor",
    "perf",
    "test",
    "build",
    "ci",
    "chore",
    "revert",
)

SKIP_ENV_VAR = "SKIP_CONVENTIONAL_COMMIT_CHECK"

CONVENTIONAL_HEADER_RE = re.compile(
    r"^(?P<type>feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)"
    r"(?:\((?P<scope>[a-zA-Z0-9._/\-]+)\))?"
    r"(?P<breaking>!)?: "
    r"(?P<subject>\S.*)$"
)

SPECIAL_ALLOWED_RE = (
    re.compile(r"^Merge (branch|remote-tracking branch|pull request) .+$"),
    re.compile(r'^Revert ".+"$'),
    re.compile(r"^(fixup|squash)! .+$"),
)


def _read_commit_message(msg_file: Path) -> str:
    content = msg_file.read_text(encoding="utf-8", errors="replace")
    lines = []
    for line in content.splitlines():
        # Git comments are ignored in commit message validation
        if line.startswith("#"):
            continue
        lines.append(line.rstrip())
    # Trim outer empty lines
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _is_valid_conventional_header(header: str) -> bool:
    if CONVENTIONAL_HEADER_RE.match(header):
        return True
    for pattern in SPECIAL_ALLOWED_RE:
        if pattern.match(header):
            return True
    return False


def _print_error(header: str) -> None:
    allowed = ", ".join(ALLOWED_TYPES)
    print("Conventional Commit lint failed.", file=sys.stderr)
    print(f"Header was: {header!r}", file=sys.stderr)
    print(
        "Expected format: <type>(<scope>)?: <description> or <type>!: <description>",
        file=sys.stderr,
    )
    print(f"Allowed types: {allowed}", file=sys.stderr)
    print("Examples:", file=sys.stderr)
    print("  feat: add memory retrieval endpoint", file=sys.stderr)
    print("  fix(api): handle empty group_id", file=sys.stderr)
    print("  refactor(core)!: simplify DI lifecycle", file=sys.stderr)


def lint_message(message: str) -> int:
    if os.getenv(SKIP_ENV_VAR) == "1":
        return 0

    first_line = message.splitlines()[0].strip() if message.strip() else ""
    if not first_line:
        _print_error(first_line)
        return 1

    if not _is_valid_conventional_header(first_line):
        _print_error(first_line)
        return 1

    return 0


def cmd_hook(files: list[str]) -> int:
    if os.getenv(SKIP_ENV_VAR) == "1":
        return 0

    if not files:
        print("No commit message file provided to hook.", file=sys.stderr)
        return 1

    msg_file = Path(files[0])
    if not msg_file.exists():
        print(f"Commit message file not found: {msg_file}", file=sys.stderr)
        return 1

    message = _read_commit_message(msg_file)
    return lint_message(message)


def cmd_check(message: str) -> int:
    return lint_message(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Conventional Commit lint tool")
    subparsers = parser.add_subparsers(dest="command")

    hook_parser = subparsers.add_parser(
        "hook", help="Validate commit message file (pre-commit hook mode)"
    )
    hook_parser.add_argument(
        "files", nargs="*", help="Commit message file path from pre-commit"
    )

    check_parser = subparsers.add_parser("check", help="Validate a raw commit message")
    check_parser.add_argument("message", help="Commit message to validate")

    args = parser.parse_args()

    if args.command == "hook":
        raise SystemExit(cmd_hook(args.files))

    if args.command == "check":
        raise SystemExit(cmd_check(args.message))

    parser.print_help()
    raise SystemExit(1)


if __name__ == "__main__":
    main()
