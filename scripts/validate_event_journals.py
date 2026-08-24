#!/usr/bin/env python3
"""Validate append-only benchmark journals without rewriting evidence."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from local_inference_bench.journal_integrity import (
    HistoricalLegacyConfig,
    JournalIssue,
    validate_append_only_record_prefix,
    validate_repository_journals,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _historical_legacy_config(value: str) -> HistoricalLegacyConfig:
    candidate_id, separator, config_sha256 = value.rpartition(":")
    if not separator or not candidate_id:
        raise argparse.ArgumentTypeError("expected CANDIDATE_ID:CONFIG_SHA256")
    if len(config_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in config_sha256
    ):
        raise argparse.ArgumentTypeError("config SHA-256 must be 64 lowercase hex characters")
    return candidate_id, config_sha256


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sustained-journal",
        type=Path,
        default=Path("results/sustained-events.jsonl"),
    )
    parser.add_argument(
        "--quality-journal",
        type=Path,
        default=Path("results/quality-events.jsonl"),
    )
    parser.add_argument(
        "--bounded-journal",
        type=Path,
        default=Path("results/bounded-events.jsonl"),
    )
    parser.add_argument(
        "--legacy-journal",
        type=Path,
        default=Path("results/events.jsonl"),
    )
    parser.add_argument(
        "--sustained-registry",
        type=Path,
        default=Path("registries/sustained_candidates.json"),
    )
    parser.add_argument(
        "--candidate-registry",
        type=Path,
        default=Path("registries/candidates.json"),
    )
    parser.add_argument(
        "--allow-historical-legacy-config",
        action="append",
        default=[],
        type=_historical_legacy_config,
        metavar="CANDIDATE_ID:CONFIG_SHA256",
        help="admit one exact pre-index smoke config using the hash printed by a strict run",
    )
    return parser


def _tracked_head_prefix_issues(paths: list[Path]) -> list[JournalIssue]:
    issues: list[JournalIssue] = []
    for path in paths:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            continue
        try:
            tracked_at_head = subprocess.run(
                [
                    "git",
                    "-C",
                    str(PROJECT_ROOT),
                    "ls-tree",
                    "--name-only",
                    "-z",
                    "HEAD",
                    "--",
                    relative,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except OSError as error:
            return [
                JournalIssue(
                    PROJECT_ROOT,
                    None,
                    "git_head_unavailable",
                    f"cannot execute Git for append-only validation: {error}",
                )
            ]
        if tracked_at_head.returncode != 0:
            issues.append(
                JournalIssue(
                    resolved,
                    None,
                    "git_head_unavailable",
                    f"cannot enumerate HEAD journal path {relative!r}",
                )
            )
            continue
        if not tracked_at_head.stdout:
            continue
        try:
            head = subprocess.run(
                ["git", "-C", str(PROJECT_ROOT), "show", f"HEAD:{relative}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except OSError as error:
            return [
                JournalIssue(
                    PROJECT_ROOT,
                    None,
                    "git_head_unavailable",
                    f"cannot execute Git for append-only validation: {error}",
                )
            ]
        if head.returncode != 0:
            issues.append(
                JournalIssue(
                    resolved,
                    None,
                    "git_head_unavailable",
                    f"cannot read HEAD journal blob for {relative!r}",
                )
            )
            continue
        issues.extend(validate_append_only_record_prefix(resolved, head.stdout))
    return issues


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    issues = validate_repository_journals(
        sustained_journal=args.sustained_journal,
        quality_journal=args.quality_journal,
        bounded_journal=args.bounded_journal,
        sustained_registry=args.sustained_registry,
        legacy_journal=args.legacy_journal,
        candidate_registry=args.candidate_registry,
        historical_legacy_configs=args.allow_historical_legacy_config,
    )
    issues.extend(
        _tracked_head_prefix_issues(
            [
                args.sustained_journal,
                args.quality_journal,
                args.bounded_journal,
                args.legacy_journal,
            ]
        )
    )
    issues.sort(
        key=lambda issue: (
            str(issue.path),
            issue.line_number if issue.line_number is not None else -1,
            issue.code,
            issue.message,
        )
    )
    if issues:
        for issue in issues:
            print(issue.format())
        print(f"FAIL: {len(issues)} journal integrity issue(s)")
        return 1
    print("PASS: journal lifecycles, corrections, replacements, and registry identities resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
