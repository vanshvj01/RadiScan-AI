"""Single-image prediction from the command line.

python -m radiscan.cli.predict --image chest.jpg --heatmaps out/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radiscan.cli._common import add_common_args, setup_logging
from radiscan.config import Config
from radiscan.inference.predictor import RadiScanPredictor


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description="Predict on one radiograph"))
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--heatmaps", default=None, help="directory for Grad-CAM overlays")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json", dest="as_json", action="store_true")
    parser.add_argument("--mc-samples", type=int, default=20)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    cfg = Config.from_yaml(args.config)
    ckpt = args.checkpoint or str(Path(cfg.train.checkpoint_dir) / "best_model.pth")

    predictor = RadiScanPredictor(checkpoint=ckpt, device=args.device, mc_samples=args.mc_samples)
    result = predictor.predict(
        args.image,
        explain=args.heatmaps is not None,
        heatmap_dir=args.heatmaps,
        top_k_heatmaps=args.top_k,
        scan_id=Path(args.image).stem,
    )

    if args.as_json:
        payload = {k: v for k, v in result.items() if k != "report"}
        payload["report"] = result["report"].to_dict()
        print(json.dumps(payload, indent=2))
    else:
        print(result["report"].to_text())
        if result["heatmaps"]:
            print("\nHEATMAPS")
            print("-" * 62)
            for label, path in result["heatmaps"].items():
                print(f"  {label:<28} {path}")
        print(f"\n  Inference: {result['inference_ms']:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
