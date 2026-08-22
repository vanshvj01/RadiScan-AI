"""Pre-flight data check.

Run this before every training run. It costs three seconds and it is the
difference between a model and 223,000 black rectangles.
"""

from __future__ import annotations

import argparse
import sys

from radiscan.cli._common import add_common_args, setup_logging
from radiscan.config import Config
from radiscan.constants import get_labels
from radiscan.data.datasets import DatasetVerificationError, verify_dataset


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description="Verify dataset integrity"))
    parser.add_argument("--sample", type=int, default=400)
    parser.add_argument("--min-hit-rate", type=float, default=0.99)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    cfg = Config.from_yaml(args.config)
    labels = get_labels(cfg.data.dataset)
    failures = 0

    for split in ("train_csv", "val_csv", "test_csv"):
        csv_path = getattr(cfg.data, split)
        if not csv_path:
            continue
        print(f"\n=== {split}: {csv_path} ===")
        try:
            report = verify_dataset(
                csv_path,
                cfg.data.data_dir,
                labels,
                sample=args.sample,
                min_hit_rate=args.min_hit_rate,
            )
        except DatasetVerificationError as exc:
            failures += 1
            print(f"FAIL\n{exc}")
            continue
        except FileNotFoundError as exc:
            failures += 1
            print(f"FAIL: {exc}")
            continue

        print(f"  rows            : {report['rows']:,}")
        print(f"  path column     : {report['path_column']}")
        print(
            f"  image hit rate  : {report['hit_rate']:.1%} ({report['found']}/{report['sampled']})"
        )
        rates = report.get("positive_rates", {})
        if rates:
            print("  positive rates  :")
            for label, rate in sorted(rates.items(), key=lambda kv: -kv[1]):
                bar = "#" * int(rate * 50)
                flag = "  <- very rare" if rate < 0.02 else ""
                print(f"      {label:<28} {rate:6.2%} {bar}{flag}")
        print("  OK")

    if failures:
        print(f"\n{failures} split(s) failed verification. Do not start training.", file=sys.stderr)
        return 1
    print("\nAll splits verified. Safe to train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
