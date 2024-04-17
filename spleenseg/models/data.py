#!/usr/bin/env python
from monai.losses.dice import DiceLoss
from monai.metrics.meandice import DiceMetric
from monai.networks.layers.factories import Norm
from monai.networks.nets.unet import UNet
import torch

from pathlib import Path
from argparse import Namespace
from dataclasses import dataclass, field

from typing import Any, Optional, Callable
from monai.data.dataset import CacheDataset
from monai.utils.misc import set_determinism
from torch.utils.data import DataLoader


@dataclass
class LoaderCache:
    loader: DataLoader
    cache: CacheDataset


@dataclass
class TrainingParams:
    max_epochs: int = 600
    val_interval = 2
    best_metric: float = -1.0
    best_metric_epoch = -1
    modelPth: Path = Path("")
    modelONNX: Path = Path("")
    determinismSeed: int = 0

    def __init__(self, options: Namespace):
        self.options = options
        if options is not None:
            self.max_epochs = self.options.maxEpochs
            self.modelPth = Path(options.outputdir) / "model.pth"
            self.modelONNX = Path(options.outputdir) / "model.onnx"
            self.determinismSeed = self.options.determinismSeed
            set_determinism(self.determinismSeed)


@dataclass
class TrainingLog:
    loss_per_epoch: list[float] = field(default_factory=list)
    metric_per_epoch: list[float] = field(default_factory=list)


@dataclass
class ModelParams:
    device: torch.device = torch.device("cpu")
    model: UNet = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    )
    fn_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = DiceLoss(
        to_onehot_y=True, softmax=True
    )
    optimizer: torch.optim.Adam | None = None
    dice_metric: DiceMetric = DiceMetric(include_background=False, reduction="mean")

    def __init__(self, options: Namespace):
        self.options = options
        if options is not None:
            self.device = torch.device(self.options.device)
            self.model = self.model.to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), 1e-4)