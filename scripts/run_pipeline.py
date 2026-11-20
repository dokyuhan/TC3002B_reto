"""Single entrypoint to run the full offline preprocessing, training, and evaluation pipeline."""

from __future__ import annotations

import argparse

from scripts.build_datasets import main as build_datasets_main
from scripts.evaluate import main as evaluate_main
from scripts.generate_plots import main as generate_plots_main
from scripts.train import main as train_main


def parse_args() -> argparse.Namespace:
    """Parse optional stage skip flags."""
    parser = argparse.ArgumentParser(
        description="Run dataset preprocessing, training, evaluation, and plot generation.",
    )
    parser.add_argument(
        "--skip-datasets",
        action="store_true",
        help="Skip raw dataset parsing when data/processed/ caches already exist.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip regenerating PNG evaluation figures.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute all offline pipeline stages sequentially."""
    args = parse_args()

    if not args.skip_datasets:
        print("=== Stage 1/4: Building dataset manifests and .npz AST caches ===")
        build_datasets_main()

    print("=== Stage 2/4: Training calibrated classifier and TF IDF baseline ===")
    train_main()

    print("=== Stage 3/4: Evaluating baselines, boilerplate shift, and IR Plag ===")
    evaluate_main()

    if not args.skip_plots:
        print("=== Stage 4/4: Generating evaluation figures ===")
        generate_plots_main()

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
