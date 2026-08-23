import argparse
from pathlib import Path

from .generate_fallback_inputs import write_manifest
from .launch_candidate import run_candidate
from .project_paths import INPUT_MANIFEST_PATH, PROJECT_ROOT
from .render_report import render_report
from .run_sustained import run_sustained_candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("make-inputs")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--candidate", required=True)
    sustained_parser = subparsers.add_parser("sustained")
    sustained_parser.add_argument("--candidate", required=True)
    sustained_parser.add_argument("--workload", required=True, type=Path)
    sustained_parser.add_argument(
        "--phase",
        required=True,
        choices=("screen", "sustained", "quality", "compatibility"),
    )
    sustained_parser.add_argument("--target-wall-seconds", required=True, type=float)
    sustained_parser.add_argument("--trials", type=int, default=1)
    sustained_parser.add_argument("--config-index", type=int, action="append")
    subparsers.add_parser("report")
    args = parser.parse_args()
    if args.command == "make-inputs":
        write_manifest(PROJECT_ROOT / "data" / "inputs" / "generated", INPUT_MANIFEST_PATH)
    elif args.command == "run":
        run_candidate(args.candidate)
    elif args.command == "sustained":
        run_sustained_candidate(
            args.candidate,
            args.workload,
            phase=args.phase,
            target_wall_seconds=args.target_wall_seconds,
            trial_count=args.trials,
            config_indices=(
                tuple(args.config_index) if args.config_index is not None else None
            ),
        )
    else:
        render_report()


if __name__ == "__main__":
    main()
