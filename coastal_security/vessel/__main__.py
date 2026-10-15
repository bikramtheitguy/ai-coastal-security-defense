"""Run with: python -m coastal_security.vessel --output artifacts/vessel-demo."""

import argparse
import json
from pathlib import Path

from coastal_security.vessel.model import TrainingConfig
from coastal_security.vessel.run import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the synthetic vessel-speed benchmark")
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory; never overwritten"
    )
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-fpr",
        type=float,
        default=0.05,
        help="Validation false-positive-rate budget (default: 0.05)",
    )
    args = parser.parse_args()
    try:
        report = run_experiment(
            args.output,
            args.rows,
            TrainingConfig(seed=args.seed, max_false_positive_rate=args.max_fpr),
        )
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Benchmark error: {exc}\n")
    print("Synthetic benchmark only; no operational performance claim.")
    print(json.dumps(report["test"]["selected_threshold"], indent=2, allow_nan=False))
    print(f"Experiment bundle: {args.output.resolve()}")


if __name__ == "__main__":
    main()
