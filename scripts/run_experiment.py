from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ltr_system.experiment import ExperimentConfig, ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run learning-to-rank experiments.")
    parser.add_argument("--train", type=str, help="Path to train.txt")
    parser.add_argument("--valid", type=str, help="Path to vali.txt")
    parser.add_argument("--test", type=str, help="Path to test.txt")
    parser.add_argument(
        "--fold-dir",
        action="append",
        default=[],
        help="Fold directory containing train.txt/vali.txt/test.txt (repeatable)",
    )
    parser.add_argument("--num-features", type=int, default=136)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def _prepare_output_dir(output_dir: str, run_name: str | None) -> Path:
    base = Path(output_dir)
    name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_metadata(args: argparse.Namespace, run_dir: Path, mode: str) -> None:
    metadata = {
        "mode": mode,
        "num_features": args.num_features,
        "k": args.k,
        "seed": args.seed,
        "train": args.train,
        "valid": args.valid,
        "test": args.test,
        "fold_dirs": args.fold_dir,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    config = ExperimentConfig(
        num_features=args.num_features,
        k=args.k,
        random_state=args.seed,
    )
    runner = ExperimentRunner(config)
    run_dir = _prepare_output_dir(args.output_dir, args.run_name)

    if args.fold_dir:
        all_folds, summary = runner.run_folds(args.fold_dir)
        all_folds.to_csv(run_dir / "per_fold_results.csv", index=False)
        summary.to_csv(run_dir / "summary_results.csv", index=False)
        _write_metadata(args, run_dir, mode="folds")
        print("\\nPer-fold results:")
        print(all_folds.to_string(index=False))
        print("\\nAverage across folds:")
        print(summary.to_string(index=False))
        print(f"\\nSaved files to: {run_dir}")
        return

    if args.train and args.valid and args.test:
        result = runner.run_single_split(args.train, args.valid, args.test)
        result.to_csv(run_dir / "single_split_results.csv", index=False)
        _write_metadata(args, run_dir, mode="single_split")
        print(result.to_string(index=False))
        print(f"\\nSaved files to: {run_dir}")
        return

    raise SystemExit(
        "Provide either --fold-dir (one or more) OR all of --train --valid --test."
    )


if __name__ == "__main__":
    main()
