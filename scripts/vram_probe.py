"""Find the largest batch size that actually fits on this GPU.

    python scripts/vram_probe.py --config configs/chexpert_6gb.yaml

Runs a real forward *and backward* pass through the real model at the config's
image size, doubling the batch until it runs out of memory, and reports the
largest that fit plus the peak VRAM each one used.

Why forward-and-backward, not just forward: training memory is dominated by the
stored activations needed for the backward pass, so a forward-only test can
report comfortable headroom for a batch size that then OOMs on the first
optimiser step. And why the real model rather than a stock torchvision one: the
multi-task heads, the fusion layer and AMP all change the picture.

The output includes a batch_size/accumulate_steps pair that keeps the effective
batch at 32, which is what the LR schedule and ASL parameters are tuned for.

Windows caveat that this script exists to catch
-----------------------------------------------
On Windows, the WDDM driver enables CUDA system-memory fallback by default. A
batch that exceeds VRAM does NOT raise OutOfMemoryError -- it silently spills
into system RAM across PCIe and keeps running, perhaps ten times slower. So
"it did not crash" is not evidence that a batch size fits. This probe compares
peak allocation against physical VRAM and treats anything at or above it as a
spill, because a training run that quietly halves its own throughput is worse
than one that fails loudly.
"""

from __future__ import annotations

import argparse
import sys

import torch

from radiscan.config import Config
from radiscan.constants import get_labels
from radiscan.models.network import build_model
from radiscan.utils.device import describe_device

TARGET_EFFECTIVE_BATCH = 32


def try_batch(model, batch: int, image_size: int, device, amp: bool) -> tuple[bool, float]:
    """Return (fitted, peak_gb). Frees everything before returning either way."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        images = torch.randn(batch, 3, image_size, image_size, device=device)
        targets = torch.randint(0, 2, (batch, len(model.labels)), device=device).float()

        with torch.autocast(device_type="cuda", enabled=amp):
            out = model(images)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                out["disease_logits"], targets
            )
        loss.backward()  # activations peak here, not in the forward
        model.zero_grad(set_to_none=True)

        peak = torch.cuda.max_memory_allocated() / 1e9
        del images, targets, out, loss
        return True, peak
    except torch.cuda.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        return False, float("nan")
    finally:
        torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Probe usable batch size on this GPU")
    p.add_argument("--config", default="configs/chexpert_6gb.yaml")
    p.add_argument("--image-size", type=int, default=None, help="override the config")
    p.add_argument("--max-batch", type=int, default=64)
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        print("No CUDA device visible -- nothing to probe.")
        print("Training will run on CPU, where batch size is bounded by RAM and patience.")
        return 0

    cfg = Config.from_yaml(args.config)
    image_size = args.image_size or cfg.data.image_size
    device = torch.device("cuda")
    total = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"Device      : {describe_device(device)}")
    print(f"Total VRAM  : {total:.2f} GB")
    print(f"Backbone    : {cfg.model.backbone}")
    print(f"Image size  : {image_size}px")
    print(f"AMP         : {cfg.train.amp}")
    print(f"\n{'batch':>7}  {'result':<9}  peak VRAM")
    print("-" * 34)

    model = build_model(cfg, get_labels(cfg.data.dataset)).to(device).train()

    # A real run also holds the EMA shadow copy, AdamW moment buffers and
    # whatever the desktop compositor is using, so require the probe's peak to
    # sit at or below this fraction of physical VRAM.
    headroom = 0.80

    best, best_peak = 0, 0.0
    batch = 1
    while batch <= args.max_batch:
        fitted, peak = try_batch(model, batch, image_size, device, cfg.train.amp)
        if not fitted:
            print(f"{batch:>7}  {'OOM':<9}  -")
            break
        if peak >= total:
            # Exceeded physical VRAM without raising: the driver spilled to host
            # RAM. Usable in the sense that it runs, useless in the sense that
            # it runs at PCIe speed.
            print(f"{batch:>7}  {'SPILLED':<9}  {peak:.2f} GB  <- exceeds {total:.2f} GB VRAM")
            break
        marker = "" if peak <= total * headroom else "  <- too tight"
        print(f"{batch:>7}  {'fits':<9}  {peak:.2f} GB{marker}")
        if peak <= total * headroom:
            best, best_peak = batch, peak
        batch *= 2

    print("-" * 34)
    if best == 0:
        print("\nNothing fits with headroom, even at batch 1.")
        print("Lower data.image_size to 256 and retry.")
        return 1

    safe = best
    accum = max(round(TARGET_EFFECTIVE_BATCH / safe), 1)

    print(
        f"\nLargest with headroom : {safe}  (peak {best_peak:.2f} GB of {total:.2f} GB, "
        f"{best_peak / total:.0%})"
    )
    print("\nPut this in your config:\n")
    print("  train:")
    print(f"    batch_size: {safe}")
    print(f"    accumulate_steps: {accum}      # {safe} x {accum} = effective batch {safe * accum}")
    print("\nOr from the command line:")
    print(f"  --set train.batch_size={safe} train.accumulate_steps={accum}")
    print(
        "\nClose your browser before a long run. Windows lets it hold several hundred MB\n"
        "of VRAM, which is the difference between fitting and an OOM in epoch 12."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
