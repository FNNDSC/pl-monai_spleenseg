#!/usr/bin/env python

from collections.abc import Iterable
from pathlib import Path
from argparse import Namespace
from dataclasses import dataclass, field

import os, sys
from monai.transforms.compose import Compose
from monai.utils import first, set_determinism

# from monai.transforms import (
#     AsDiscrete,
#     AsDiscreted,
#     EnsureChannelFirstd,
#     Compose,
#     CropForegroundd,
#     LoadImaged,
#     Orientationd,
#     RandCropByPosNegLabeld,
#     RandAffined,
#     SaveImaged,
#     ScaleIntensityRanged,
#     Spacingd,
#     Invertd,
# )
from monai.handlers.utils import from_engine
from monai.networks.nets.unet import UNet

# from monai.transforms import LoadImage
from monai.networks.layers.factories import Norm
from monai.metrics.meandice import DiceMetric
from monai.losses.dice import DiceLoss
from monai.inferers.utils import sliding_window_inference
from monai.data.dataset import CacheDataset
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.data.utils import decollate_batch
from monai.config.deviceconfig import print_config
from monai.apps.utils import download_and_extract
import torch

# import matplotlib.pyplot as plt
# import tempfile
# import shutil
# import glob
# import pudb
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
            set_determinism(seed=self.determinismSeed)


@dataclass
class TrainingLog:
    loss_per_epoch: list[float] = field(default_factory=list)
    metric_per_epoch: list[float] = field(default_factory=list)


@dataclass
class ModelParams:
    optimizer: torch.optim.Adam
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
    dice_metric: DiceMetric = DiceMetric(include_background=False, reduction="mean")

    def __init__(self, options: Namespace):
        self.options = options
        if options is not None:
            self.device = torch.device(self.options.device)
            torch.manual_seed(42)
            self.model = self.model.to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), 1e-4)


def tensor_desc(
    T: torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor], **kwargs
) -> torch.Tensor:
    strAs: str = "meanstd"
    v1: float = 0.0
    v2: float = 0.0
    tensor: torch.Tensor = torch.Tensor([v1, v2])
    for k, v in kwargs.items():
        if k.lower() == "desc":
            strAs = v
    T = torch.as_tensor(T)
    match strAs:
        case "meanstd":
            tensor = torch.Tensor([T.mean().item(), T.std().item()])
        case "l1l2":
            tensor = torch.Tensor([T.abs().sum().item(), T.pow(2).sum().sqrt().item()])
        case "minmax":
            tensor = torch.Tensor([T.min().item(), T.max().item()])
        case "simplified":
            tensor = T.mean(dim=(1, 2), keepdim=True)
    return tensor


class NeuralNet:
    def __init__(self, options: Namespace):
        self.network: ModelParams = ModelParams(options)
        self.training: TrainingParams = TrainingParams(options)

        self.input: torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]
        self.output: torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]
        self.target: torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]

        self.trainingLog: TrainingLog = TrainingLog()

        self.f_outputPost: Compose
        self.f_labelPost: Compose

        self.trainingSpace: LoaderCache
        self.validationSpace: LoaderCache
        self.novelSpace: LoaderCache

    def loaderCache_create(
        self, fileList: list[dict[str, str]], transforms: Compose, batch_size: int = 2
    ) -> LoaderCache:
        """
        ## Define CacheDataset and DataLoader for training and validation

        Here we use CacheDataset to accelerate training and validation process,
        it's 10x faster than the regular Dataset.

        To achieve best performance, set `cache_rate=1.0` to cache all the data,
        if memory is not enough, set lower value.

        Users can also set `cache_num` instead of `cache_rate`, will use the
        minimum value of the 2 settings.

        Set `num_workers` to enable multi-threads during caching.

        NB: Parameterize all params!!
        """
        ds: CacheDataset = CacheDataset(
            data=fileList, transform=transforms, cache_rate=1.0, num_workers=4
        )

        # use batch_size=2 to load images and use RandCropByPosNegLabeld
        # to generate 2 x 4 images for network training
        loader: DataLoader
        if batch_size == 2:
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
        else:
            loader = DataLoader(ds, batch_size=batch_size, num_workers=4)

        loaderCache: LoaderCache = LoaderCache(cache=ds, loader=loader)
        return loaderCache

    def tensor_assign(
        self,
        to: str,
        T: torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor],
    ):
        if T is not None:
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
        self,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]:
        """
        Simply run the self.input and generate/return/store an output
        """
        # print(tensor_desc(self.input))
        self.output = self.network.model(self.input)
        # print(tensor_desc(self.output))
        return self.output

    def evalAndCorrect(self) -> float:
        self.network.optimizer.zero_grad()

        self.feedForward()
        f_loss: torch.Tensor = self.network.fn_loss(
            torch.as_tensor(self.output),
            torch.as_tensor(self.target),
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
            sample_loss = self.evalAndCorrect()
            total_loss += sample_loss
            if (
                trainingSpace.cache is not None
                and trainingSpace.loader.batch_size is not None
            ):
                print(
                    f"{sample:02}/{len(trainingSpace.cache) // trainingSpace.loader.batch_size}, "
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
        for epoch in range(self.training.max_epochs):
            print("-" * 10)
            print(f"epoch {epoch+1:03} / {self.training.max_epochs}")
            self.network.model.train()
            epoch_loss = self.train_overSampleSpace_retLoss(self.trainingSpace)
            print(f"epoch {epoch+1:03}, average loss: {epoch_loss:.4f}")
            self.trainingLog.loss_per_epoch.append(epoch_loss)
            if (epoch + 1) % self.training.val_interval == 0:
                self.slidingWindowInference_do(self.validationSpace, epoch)

    def inference_metricsProcess(self, epoch: int) -> float:
        metric: float = self.network.dice_metric.aggregate().item()
        self.trainingLog.metric_per_epoch.append(metric)
        self.network.dice_metric.reset()
        if metric > self.training.best_metric:
            self.training.best_metric = metric
            self.training.best_metric_epoch = epoch + 1
            torch.save(self.network.model.state_dict(), str(self.training.modelPth))
            print("saved new best metric model")
        print(
            f"current epoch: {epoch + 1}, current mean dice: {metric:.4f}"
            f"\nbest mean dice: {self.training.best_metric:.4f}"
            f"at epoch: {self.training.best_metric_epoch}"
        )
        return metric

    def slidingWindowInference_do(
        self, inferCache: LoaderCache, epoch: int | None = None
    ) -> float:
        metric: float = 0.0
        self.network.model.eval()
        with torch.no_grad():
            for sample in inferCache.loader:
                input: torch.Tensor = sample["image"].to(self.network.device)
                roi_size: tuple[int, int, int] = (160, 160, 160)
                sw_batch_size: int = 4
                outputRaw: torch.Tensor = torch.as_tensor(
                    sliding_window_inference(
                        input, roi_size, sw_batch_size, self.network.model
                    )
                )
                outputPostProc = [
                    self.f_outputPost(i)
                    for i in decollate_batch(outputRaw)  # type: ignore[arg-type]
                ]
                if epoch is not None:
                    labelTruth: torch.Tensor = sample["label"].to(self.network.device)
                    labelPostProc = [
                        self.f_labelPost(i)
                        for i in decollate_batch(labelTruth)  # type: ignore[arg-type]
                    ]
                    self.network.dice_metric(
                        y_pred=outputPostProc,  # type: ignore
                        y=labelPostProc,  # type: ignore
                    )
            if epoch is not None:
                metric = self.inference_metricsProcess(epoch)
        return metric
