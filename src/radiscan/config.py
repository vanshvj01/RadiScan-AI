"""Typed configuration. Single source of truth for every tunable."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DataConfig:
    dataset: str = "chexpert"
    data_dir: str = "data/raw/chexpert"
    train_csv: str = ""
    val_csv: str = ""
    test_csv: str = ""
    image_size: int = 320
    uncertain_policy: str = "lsr"  # zeros | ones | ignore | lsr (label smoothing)
    lsr_low: float = 0.05
    lsr_high: float = 0.85
    max_train_images: int | None = None
    max_val_images: int | None = None
    num_workers: int = 4
    balanced_sampling: bool = True
    frontal_only: bool = True
    # Crop away photo backgrounds and empty borders before the model sees the
    # image (radiscan.data.enhance). Recorded in the checkpoint, so inference
    # applies exactly what training did.
    auto_crop: bool = True


@dataclass
class ModelConfig:
    backbone: str = (
        "densenet121"  # any timm name; densenet121 | convnext_tiny | tf_efficientnetv2_s
    )
    pretrained: bool = True
    use_transformer: bool = False
    vit_name: str = "vit_small_patch16_224"
    dropout: float = 0.2
    fracture_head: bool = True
    confidence_head: bool = True


@dataclass
class LossConfig:
    kind: str = "asl"  # asl | focal | bce
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 0.0
    asl_clip: float = 0.05
    focal_gamma: float = 2.0
    pos_weight: bool = True
    disease_weight: float = 1.0
    fracture_weight: float = 0.3


@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 32
    lr: float = 1e-4
    head_lr_mult: float = 10.0
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    scheduler: str = "cosine"  # cosine | plateau | none
    amp: bool = True
    ema_decay: float = 0.999
    grad_clip: float = 5.0
    accumulate_steps: int = 1
    early_stopping_patience: int = 6
    freeze_backbone_epochs: int = 0
    mixup_alpha: float = 0.2
    seed: int = 42
    checkpoint_dir: str = "models/checkpoints"
    log_dir: str = "logs"
    monitor: str = "val_mean_auc"  # maximised


@dataclass
class InferenceConfig:
    mc_dropout_samples: int = 20
    tta_hflip: bool = True
    temperature: float | None = None  # filled by calibration
    thresholds: dict[str, float] = field(default_factory=dict)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # -- (de)serialisation ---------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        return cls(
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            loss=LossConfig(**raw.get("loss", {})),
            train=TrainConfig(**raw.get("train", {})),
            inference=InferenceConfig(**raw.get("inference", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    def override(self, **kwargs: Any) -> Config:
        """Apply flat CLI overrides like ``epochs=5`` or ``data.batch_size=8``."""
        for key, value in kwargs.items():
            if value is None:
                continue
            section, _, attr = key.partition(".")
            if attr:
                setattr(getattr(self, section), attr, value)
                continue
            for sec in (self.data, self.model, self.loss, self.train, self.inference):
                if hasattr(sec, key):
                    setattr(sec, key, value)
                    break
            else:
                raise AttributeError(f"Unknown config key: {key}")
        return self
