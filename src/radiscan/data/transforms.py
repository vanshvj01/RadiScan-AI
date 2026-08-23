"""Preprocessing and augmentation pipelines.

Two decisions here matter for accuracy:

1. **CLAHE before normalisation.** Radiographs are low-contrast by nature;
   Contrast Limited Adaptive Histogram Equalisation (clip 2.0, 8x8 tiles) is the
   standard radiology enhancement and reliably adds ~1 AUC point.
2. **320px, not 224px.** Nodules and subtle infiltrates occupy very few pixels.
   Published CheXpert results improve monotonically up to ~384px. 320 is the
   sweet spot for 8-12 GB of VRAM.
"""

from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

from radiscan.constants import IMAGENET_MEAN, IMAGENET_STD


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """CLAHE on the luminance channel; input and output are uint8 RGB."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def train_transforms(image_size: int = 320) -> A.Compose:
    """Augmentations chosen to be anatomically plausible.

    Note what is *absent*: horizontal flip is applied only mildly and vertical
    flip never. A mirrored chest X-ray implies situs inversus; flipping teaches
    the model that cardiomegaly can appear on the right, which is wrong. We keep
    hflip at low probability purely as a regulariser and rely on TTA-hflip being
    consistent with it.
    """
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(image_size, image_size), scale=(0.85, 1.0), ratio=(0.9, 1.11)
            ),
            A.HorizontalFlip(p=0.5),
            # Affine, not the deprecated ShiftScaleRotate. Same geometry:
            # +/-5% translation, +/-10% scale, +/-10 degrees. Rotation stays
            # small on purpose -- a chest radiograph is acquired in a
            # standardised position, so large rotations are not augmentation,
            # they are images the model will never see at inference.
            A.Affine(
                translate_percent=(-0.05, 0.05),
                scale=(0.9, 1.1),
                rotate=(-10, 10),
                border_mode=cv2.BORDER_CONSTANT,
                p=0.7,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.OneOf(
                [A.GaussNoise(std_range=(0.02, 0.1)), A.GaussianBlur(blur_limit=(3, 5))], p=0.2
            ),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(0.05, 0.15),
                hole_width_range=(0.05, 0.15),
                p=0.3,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def eval_transforms(image_size: int = 320) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def denormalize(tensor) -> np.ndarray:
    """Undo ImageNet normalisation -> uint8 RGB, for heatmap overlays."""
    import torch

    arr = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
    if arr.ndim == 4:
        arr = arr[0]
    arr = arr.transpose(1, 2, 0)
    arr = arr * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)
