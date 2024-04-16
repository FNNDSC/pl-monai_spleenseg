#!/usr/bin/env python

from collections.abc import Iterable
from pathlib import Path
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
from dataclasses import dataclass, field

import os, sys
from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandAffined,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.transforms import LoadImage
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import glob
import pudb
from typing import Any, Optional, Callable
import numpy as np


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


class NeuralNet:
    def __init__(self, options: Namespace):
        self.network: ModelParams = ModelParams(options)
        self.training: TrainingParams = TrainingParams(options)

        self.input: torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]
        self.output: torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]
        self.target: torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]

        self.trainingLog: TrainingLog = TrainingLog()

        self.f_outputPost: Compose | None = None
        self.f_labelPost: Compose | None = None

        self.trainingSpace: LoaderCache
        self.validationSpace: LoaderCache
        self.novelSpace: LoaderCache

    def tensor_assign(
        self,
        to: str,
        T: torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor],
    ):
        if T != None:
            match to.lower():
                case "input":
                    self.input = T
                case "output":
                    self.output = T
                case "target":
                    self.target = T
                case _:
                    self.input = T
        return T

    def feedForward(
        self, input: torch.Tensor = Optional[None]
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]:
        """
        Simply run the self.input and generate an output
        """
        if input:
            self.input = input
        self.output = self.network.model(self.input)
        return self.output

    def evalAndCorrect(
        self, input: torch.Tensor, target: torch.Tensor = Optional[None]
    ) -> float:
        self.network.optimizer.zero_grad()
        if target:
            self.target = target
        f_loss: torch.Tensor = self.network.fn_loss(
            self.feedForward(input), self.target
        )
        f_loss.backward()
        self.network.optimizer.step()
        return f_loss.item()

    def train_overSampleSpace_retLoss(self, trainingSpace: LoaderCache) -> float:
        sample: int = 0
        sample_loss: float = 0.0
        total_loss: float = 0.0
        for trainingInstance in trainingSpace.loader:
            sample += 1
            self.input, self.target = (
                trainingInstance["image"].to(self.network.device),
                trainingInstance["label"].to(self.network.device),
            )
            sample_loss = self.evalAndCorrect(self.input, self.target)
            total_loss += sample_loss
            print(
                f"{sample}/{int(trainingSpace.cache) // trainingSpace.loader.batch_size}, "
                f"sample loss: {sample_loss:.4f}"
            )
        total_loss /= sample
        return total_loss

    def train(
        self,
        trainingSpace: LoaderCache | None = None,
        validationSpace: LoaderCache | None = None,
    ):
        epoch: int = 0
        epoch_loss: float = 0.0
        if trainingSpace:
            self.trainingSpace = trainingSpace
        if validationSpace:
            self.validationSpace = validationSpace
        self.network.model.train()
        for epoch in range(self.training.max_epochs):
            print("-" * 10)
            print(f"epoch {epoch:03 + 1} / {self.training.max_epochs}")
            self.network.model.train()
            epoch_loss = self.train_overSampleSpace_retLoss(self.trainingSpace)
            print(f"epoch {epoch:03 + 1}, average loss: {epoch_loss:.4f}")
            self.trainingLog.loss_per_epoch.append(epoch_loss)

            self.slidingWindowInference_do(self.validationSpace)
            print(f"current epoch: {epoch + 1}, current mean dice")

    def inference_metricsProcess(self):
        metric: float = self.network.dice_metric.aggregate().item()
        self.trainingLog.metric_per_epoch.append(metric)
        self.network.dice_metric.reset()
        if metric > self.training.best_metric:
            self.training.best_metric = metric
            self.training.best_metric_epoch = epoch + 1
            torch.save(self.network.model.state_dict(), str(self.training.modelPth))
            print("saved new best metric model")

    def slidingWindowInference_do(
        self, inferCache: LoaderCache, truthCache: LoaderCache | None = None
    ):
        self.network.model.eval()
        with torch.no_grad():
            for sample in inferCache.loader:
                input: torch.Tensor = sample["image"].to(self.network.device)
                roi_size: tuple[int, int, int] = (160, 160, 160)
                sw_batch_size: int = 4
                outputRaw: torch.Tensor = sliding_window_inference(
                    input, roi_size, sw_batch_size, self.network.model
                )
                outputPostProc = [
                    self.f_outputPost(i) for i in decollate_batch(outputRaw)
                ]
                if truthCache:
                    labelTruth: torch.Tensor = sample["label"].to(self.network.device)
                    labelPostProc = [
                        self.f_labelPost(i) for i in decollate_batch(labelTruth)
                    ]
                    self.network.dice_metric(
                        y_pred=training.val_outputs, y=training.val_labels
                    )
            if truthCache:
                self.inference_metricsProcess()
