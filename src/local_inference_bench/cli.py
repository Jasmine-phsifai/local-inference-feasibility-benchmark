import argparse

from .generate_fallback_inputs import write_manifest
from .launch_candidate import run_candidate
from .project_paths import INPUT_MANIFEST_PATH, PROJECT_ROOT
from .render_report import render_report


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("make-inputs")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--candidate", required=True)
    subparsers.add_parser("report")
    args = parser.parse_args()
    if args.command == "make-inputs":
        write_manifest(PROJECT_ROOT / "data" / "inputs" / "generated", INPUT_MANIFEST_PATH)
    elif args.command == "run":
        run_candidate(args.candidate)
    else:
        render_report()


if __name__ == "__main__":
    main()
