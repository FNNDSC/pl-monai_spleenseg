#!/usr/bin/env python

from collections.abc import Iterable
from pathlib import Path
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
from dataclasses import dataclass
from dataclasses import dataclass

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


from chris_plugin import chris_plugin, PathMapper

__version__ = "1.0.0"

DISPLAY_TITLE = r"""

███████╗██████╗ ██╗     ███████╗███████╗███╗   ██╗███████╗███████╗ ██████╗
██╔════╝██╔══██╗██║     ██╔════╝██╔════╝████╗  ██║██╔════╝██╔════╝██╔════╝
███████╗██████╔╝██║     █████╗  █████╗  ██╔██╗ ██║███████╗█████╗  ██║  ███╗
╚════██║██╔═══╝ ██║     ██╔══╝  ██╔══╝  ██║╚██╗██║╚════██║██╔══╝  ██║   ██║
███████║██║     ███████╗███████╗███████╗██║ ╚████║███████║███████╗╚██████╔╝
╚══════╝╚═╝     ╚══════╝╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝ ╚═════╝
"""


parser = ArgumentParser(
    description="""
A ChRIS DS plugin based on Project MONAI 3D Spleen Segmentation.
This plugin implements both training and inference, with some
refactoring and pervasive type hinting.
    """,
    formatter_class=ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--mode",
    type=str,
    default="training",
    help="mode of behaviour: training or inference",
)
parser.add_argument(
    "--trainImageDir",
    type=str,
    default="imagesTr",
    help="name of directory containing training images",
)
parser.add_argument(
    "--trainLabelsDir",
    type=str,
    default="labelsTr",
    help="name of directory containing training labels",
)
parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="GPU/CPU device to use",
)
parser.add_argument(
    "--determinismSeed",
    type=int,
    default=42,
    help="the determinism seed for training/evaluation"
)
parser.add_argument(
    "--maxEpochs",
    type=int,
    default=600,
    help="max number of epochs to consider",
)
parser.add_argument(
    "--validateSize",
    type=int,
    default=9,
    help="size of the validation set in the input raw/label space",
)
parser.add_argument(
    "--pattern", type=str, default="**/[!._]*nii.gz", help="filter glob for input files"
)
parser.add_argument(
    "-V", "--version", action="version", version=f"%(prog)s {__version__}"
)


@dataclass
class LoaderCache:
    loader: DataLoader
    cache: CacheDataset

@dataclass
class TrainingParams:
    max_epochs: int = 600
    val_interval = 2
    epoch_loss_values: list[float] = []
    metric_values: list[float] = []
    best_metric: float = -1.0
    best_metric_epoch = -1
    modelPth: Path = Path("")
    modelONNX: Path = Path("")
    determinismSeed:int = 0

    def __init__(self, options: Namespace):
        self.options = options
        if options is not None:
            self.max_epochs = self.options.maxEpochs
            self.modelPth = options.outputdir / "model.pth"
            self.modelONNX = options.outputdir / "model.onnx"
            self.determinismSeed = self.options.determinismSeed
            set_determinism(self.determinismSeed)

@dataclass
class TrainingLog:
    loss_per_epoch: list[float] = []
    metric_per_epoch: list[float] = []

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
    fn_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer: torch.optim.Adam = None
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

        self.input: torch.Tensor|tuple[torch.Tensor, ...]|dict[Any, torch.Tensor]
        self.output: torch.Tensor|tuple[torch.Tensor, ...]|dict[Any, torch.Tensor]
        self.target: torch.Tensor|tuple[torch.Tensor, ...]|dict[Any, torch.Tensor]

        self.trainingLog: TrainingLog = TrainingLog()

        self.f_outputPost: Compose = None
        self.f_labelPost: Compose = None

        self.trainingSpace: LoaderCache
        self.validationSpace: LoaderCache
        self.novelSpace: LoaderCache


    def tensor_assign(self, to:str, T: torch.Tensor|tuple[torch.Tensor, ...]|dict[Any, torch.Tensor]):
        if T != None:
            match(to.lower()):
                case 'input':
                    self.input = T
                case 'output':
                    self.output = T
                case 'target':
                    self.target = T
                case _:
                    self.input = T
        return T

    def feedForward(self, input: torch.Tensor = Optional[None]) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[Any,torch.Tensor]:
        """
        Simply run the self.input and generate an output
        """
        if input:
            self.input = input
        self.output = self.network.model(self.input)
        return self.output

    def evalAndCorrect(self, input: torch.Tensor, target: torch.Tensor = Optional[None]) -> float:
        self.network.optimizer.zero_grad()
        if target:
            self.target = target
        f_loss: torch.Tensor = self.network.fn_loss(
                                    self.feedForward(input), self.target)
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
                trainingInstance["label"].to(self.network.device)
            )
            sample_loss = self.evalAndCorrect(self.input, self.target)
            total_loss += sample_loss
            print(
                f"{sample}/{int(trainingSpace.cache) // trainingSpace.loader.batch_size}, "
                f"sample loss: {sample_loss:.4f}"
            )
        total_loss /= sample
        return total_loss


    def train(self, trainingSpace: LoaderCache|None = None, validationSpace: LoaderCache|None = None):
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
            print(
                f"current epoch: {epoch + 1}, current mean dice"
            )

    def inference_metricsProcess(self):
        metric: float = self.network.dice_metric.aggregate().item()
        self.trainingLog.metric_per_epoch.append(metric)
        self.network.dice_metric.reset()
        if metric > self.training.best_metric:
            self.training.best_metric = metric
            self.training.best_metric_epoch = epoch + 1
            torch.save(
                self.network.model.state_dict(),
                str(self.training.modelPth)
            )
            print("saved new best metric model")

    def slidingWindowInference_do(self, inferCache: LoaderCache, truthCache: LoaderCache|None = None):
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
                        y_pred = training.val_outputs, y=training.val_labels
                    )
            if truthCache:
                self.inference_metricsProcess()


@dataclass
class trainingMetrics:
    max_epochs: int = 600
    val_interval = 2
    epoch_loss_values: list[float] = []
    metric_values: list[float] = []
    best_metric: float = -1.0
    best_metric_epoch = -1
    modelPth: Path = Path("")
    modelONNX: Path = Path("")
    train_outputs: torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor] = (
        torch.Tensor([])
    )
    train_labels: list = []
    val_outputs: torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor] = (
        torch.Tensor([])
    )
    val_labels: list = []

    def __init__(self, options: Namespace):
        self.options = options
        if options is not None:
            self.max_epochs = self.options.maxEpochs
            self.modelPth = options.outputdir / "model.pth"
            self.modelONNX = options.outputdir / "model.onnx"

@dataclass
class inferenceMetrics:
    image_outputs: torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor] = (
        torch.Tensor([])
    )
    train_labels: list = []
    label_outputs: torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor] = (
        torch.Tensor([])
    )
    val_labels: list = []




def trainingData_prep(options: Namespace, inputDir: Path) -> list[dict[str, str]]:
    """
    Generates a list of dictionary entries, each of which is simply the name
    of an image file and its corresponding label file.
    """
    trainRaw: list[Path] = []
    trainLbl: list[Path] = []
    for group in [options.trainImageDir, options.trainLabelsDir]:
        for path in inputDir.rglob(group):
            if group == path.name and path.name == options.trainImageDir:
                trainRaw.extend(path.glob(options.pattern))
            elif group == path.name and path.name == options.trainLabelsDir:
                trainLbl.extend(path.glob(options.pattern))
    trainRaw.sort()
    trainLbl.sort()
    return [
        {"image": str(image_name), "label": str(label_name)}
        for image_name, label_name in zip(trainRaw, trainLbl)
    ]


def inputFiles_splitInto_train_validate(
    options: Namespace, inputDir: Path
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """
    Returns a list of image+label filenames to use for training and
    a list to use for validation.
    """
    trainingSpace: list[dict[str, str]] = trainingData_prep(options, inputDir)
    trainingSet: list[dict[str, str]] = trainingSpace[: -options.validateSize]
    validateSet: list[dict[str, str]] = trainingSpace[-options.validateSize :]
    return trainingSet, validateSet



"""
Setup transforms for training and validation

Here we use several transforms to augment the dataset:
*. `LoadImaged` loads the spleen CT images and labels from NIfTI format files.
*. `EnsureChannelFirstd` ensures the original data to construct "channel first" shape.
*. `Orientationd` unifies the data orientation based on the affine matrix.
*. `Spacingd` adjusts the spacing by `pixdim=(1.5, 1.5, 2.)` based on the affine matrix.
*. `ScaleIntensityRanged` extracts intensity range [-57, 164] and scales to [0, 1].
*. `CropForegroundd` removes all zero borders to focus on the valid body area of the images and labels.
*. `RandCropByPosNegLabeld` randomly crop patch samples from big image based on pos / neg ratio.
The image centers of negative samples must be in valid body area.
*. `RandAffined` efficiently performs `rotate`, `scale`, `shear`, `translate`, etc. together based on PyTorch affine transform.
"""

def f_LoadImaged() -> Callable[[dict[str, Any]], dict[str, Any]]:
    return LoadImaged(keys=["image", "label"])


def f_EnsureChannelFirstd() -> Callable[[dict[str, Any]], dict[str, Any]]:
    return EnsureChannelFirstd(keys=["image", "label"])


def f_ScaleIntensityRanged() -> Callable[[dict[str, Any]], dict[str, Any]]:
    return ScaleIntensityRanged(
        keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True
    )


def f_CropForegroundd() -> Callable[[dict[str, Any]], dict[str, Any]]:
    return CropForegroundd(keys=["image", "label"], source_key="image")


def f_Orientationd() -> Callable[[dict[str, Any]], dict[str, Any]]:
    return Orientationd(keys=["image", "label"], axcodes="RAS"))


def f_Spaceingd() -> Callable[[dict[str, Any]], dict[str, Any]]:
    return Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        )


def f_RandCropByPosNegLabeld() -> Callable[[dict[str, Any]], dict[str, Any]]:
    return RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            )


def f_RandAffined() -> Callable[[dict[str, Any]], dict[str, Any]]:
    return RandAffined(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=1.0,
                spatial_size=(96, 96, 96),
                rotate_range=(0, 0, np.pi / 15),
                scale_range=(0.1, 0.1, 0.1),
            )


def f_Invertd(transform: Compose) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return Invertd(
            keys="pred",
            transform=transform,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
            device="cpu",
        )

def f_predAsDiscreted() -> Callable[[dict[str, Any]], dict[str, Any]]:
    return AsDiscreted(keys="pred", argmax=True, to_onehot=2)

def f_labelAsDiscreted() -> Callable[[dict[str, Any]], dict[str, Any]]:
    return AsDiscreted(keys="label", to_onehot=2)

def f_AsDiscreteArgMax() -> Callable[[dict[str, Any]], dict[str, Any]]:
    return AsDiscrete(argmax=True, to_onehot=2)

def f_AsDiscrete() -> Callable[[dict[str, Any]], dict[str, Any]]:
    return AsDiscrete(to_onehot=2)

def transforms_build(f_transform: list[Callable]) -> Compose:
    transforms: Compose = Compose(f_transform)
    return transforms

# def transforms_setup(
#     randCropByPosNegLabeld: bool = False, randAffined: bool = False
# ) -> Compose:
#     """
#     Setup transforms for training and validation
#
#     Here we use several transforms to augment the dataset:
#     *. `LoadImaged` loads the spleen CT images and labels from NIfTI format files.
#     *. `EnsureChannelFirstd` ensures the original data to construct "channel first" shape.
#     *. `Orientationd` unifies the data orientation based on the affine matrix.
#     *. `Spacingd` adjusts the spacing by `pixdim=(1.5, 1.5, 2.)` based on the affine matrix.
#     *. `ScaleIntensityRanged` extracts intensity range [-57, 164] and scales to [0, 1].
#     *. `CropForegroundd` removes all zero borders to focus on the valid body area of the images and labels.
#     *. `RandCropByPosNegLabeld` randomly crop patch samples from big image based on pos / neg ratio.
#     The image centers of negative samples must be in valid body area.
#     *. `RandAffined` efficiently performs `rotate`, `scale`, `shear`, `translate`, etc. together based on PyTorch affine transform.
#     """
#     composeList: list = []
#     composeList.append(LoadImaged(keys=["image", "label"]))
#     composeList.append(EnsureChannelFirstd(keys=["image", "label"]))
#     composeList.append(
#         ScaleIntensityRanged(
#             keys=["image"],
#             a_min=-57,
#             a_max=164,
#             b_min=0.0,
#             b_max=1.0,
#             clip=True,
#         )
#     )
#     composeList.append(CropForegroundd(keys=["image", "label"], source_key="image"))
#     composeList.append(Orientationd(keys=["image", "label"], axcodes="RAS"))
#     composeList.append(
#         Spacingd(
#             keys=["image", "label"],
#             pixdim=(1.5, 1.5, 2.0),
#             mode=("bilinear", "nearest"),
#         )
#     )
#     if randCropByPosNegLabeld:
#         composeList.append(
#             RandCropByPosNegLabeld(
#                 keys=["image", "label"],
#                 label_key="label",
#                 spatial_size=(96, 96, 96),
#                 pos=1,
#                 neg=1,
#                 num_samples=4,
#                 image_key="image",
#                 image_threshold=0,
#             )
#         )
#     if randAffined:
#         composeList.append(
#             RandAffined(
#                 keys=["image", "label"],
#                 mode=("bilinear", "nearest"),
#                 prob=1.0,
#                 spatial_size=(96, 96, 96),
#                 rotate_range=(0, 0, np.pi / 15),
#                 scale_range=(0.1, 0.1, 0.1),
#             )
#         )
#     transforms: Compose = Compose(composeList)
#     return transforms
#

# def f_Invertd(transform: Compose) -> Callable[[dict[str, Any]], dict[str, Any]]:
#     return Invertd(
#             keys="pred",
#             transform=transform,
#             orig_keys="image",
#             meta_keys="pred_meta_dict",
#             orig_meta_keys="image_meta_dict",
#             meta_key_postfix="meta_dict",
#             nearest_interp=False,
#             to_tensor=True,
#             device="cpu",
#         )
#
# def f_predAsDiscreted() -> Callable[[dict[str, Any]], dict[str, Any]]:
#     return AsDiscreted(keys="pred", argmax=True, to_onehot=2)
#
# def f_labelAsDiscreted() -> Callable[[dict[str, Any]], dict[str, Any]]:
#     return AsDiscreted(keys="label", to_onehot=2)
#
# def transforms_post(transform: Compose, additionalTransforms: list) -> Compose:
#     composeList: list = []
#     composeList.append(
#         Invertd(
#             keys="pred",
#             transform=transform,
#             orig_keys="image",
#             meta_keys="pred_meta_dict",
#             orig_meta_keys="image_meta_dict",
#             meta_key_postfix="meta_dict",
#             nearest_interp=False,
#             to_tensor=True,
#             device="cpu",
#         )
#     )
#     composeList.append(
#         AsDiscreted(keys="pred", argmax=True, to_onehot=2),
#     )
#     if len(additionalTransforms):
#         composeList.extend(additionalTransforms)
#     transforms: Compose = Compose(composeList)
#     return transforms


def transforms_check(
    outputdir: Path, files: list[dict[str, str]], transforms: Compose
) -> bool:
    check_ds: Dataset = Dataset(data=files, transform=transforms)
    check_loader: DataLoader = DataLoader(check_ds, batch_size=1)
    check_data: Any | None = first(check_loader)
    if not check_data:
        return False
    image, label = (check_data["image"][0][0], check_data["label"][0][0])
    print(f"image shape: {image.shape}, label shape: {label.shape}")
    plot_imageAndLabel(image, label, outputdir / "exemplar_image_label.jpg")
    return True


def plot_imageAndLabel(
    image: torch.Tensor, label: torch.Tensor, savefile: Path
) -> None:
    plt.figure("check", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("image")
    plt.imshow(image[:, :, 80], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("label")
    plt.imshow(label[:, :, 80])
    plt.savefig(str(savefile))


def plot_trainingMetrics(training: trainingMetrics, savefile: Path) -> None:
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(training.epoch_loss_values))]
    y = training.epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [training.val_interval * (i + 1) for i in range(len(training.metric_values))]
    y = training.metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.savefig(str(savefile))


def loaderCache_create(
    fileList: list[dict[str, str]], transforms: Compose, batch_size: int = 2
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
    loader: DataLoader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=4
    )
    loaderCache: LoaderCache = LoaderCache(cache=ds, loader=loader)
    return loaderCache


def training_do(
    training: trainingMetrics,
    network: ModelParams,
    trainMeta: LoaderCache,
    valMeta: LoaderCache,
    outputDir: Path,
) -> trainingMetrics:
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    for epoch in range(training.max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{training.max_epochs}")
        network.model.train()
        epoch_loss = 0.0
        step = 0
        for batch_data in trainMeta.loader:
            step += 1
            inputs, training.train_labels = (
                batch_data["image"].to(network.device),
                batch_data["label"].to(network.device),
            )
            network.optimizer.zero_grad()
            training.train_outputs = network.model(inputs)
            loss = network.loss_function(training.train_outputs, training.train_labels)
            loss.backward()
            network.optimizer.step()
            epoch_loss += loss.item()
            print(
                f"{step}/{int(len(trainMeta.cache)) // int(trainMeta.loader.batch_size)}, "
                f"train_loss: {loss.item():.4f}"
            )
        epoch_loss /= step
        training.epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % training.val_interval == 0:
            network.model.eval()
            with torch.no_grad():
                for val_data in valMeta.loader:
                    val_inputs, training.val_labels = (
                        val_data["image"].to(network.device),
                        val_data["label"].to(network.device),
                    )
                    roi_size = (160, 160, 160)
                    sw_batch_size = 4
                    training.val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, network.model
                    )
                    training.val_outputs = [
                        post_pred(i) for i in decollate_batch(training.val_outputs)
                    ]
                    training.val_labels = [
                        post_label(i) for i in decollate_batch(training.val_labels)
                    ]
                    # compute metric for current iteration
                    network.dice_metric(
                        y_pred=training.val_outputs, y=training.val_labels
                    )

                # aggregate the final mean dice result
                metric = network.dice_metric.aggregate().item()
                # reset the status for next validation round
                network.dice_metric.reset()

                training.metric_values.append(metric)
                if metric > training.best_metric:
                    training.best_metric = metric
                    training.best_metric_epoch = epoch + 1
                    torch.save(
                        network.model.state_dict(),
                        str(training.modelPth),
                    )
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {training.best_metric:.4f} "
                    f"at epoch: {training.best_metric_epoch}"
                )
    print(
        f"train completed, best_metric: {training.best_metric:.4f} "
        f"at epoch: {training.best_metric_epoch}"
    )
    return training


def IO_plotsDo(
    input: dict[str, torch.Tensor], output: torch.Tensor, title: str
) -> None:
    plt.figure("check", (18, 6))
    plt.subplot(1, 3, 1)
    plt.title(f"image {title}")
    plt.imshow(input["image"][0, 0, :, :, 80], cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title(f"label {title}")
    plt.imshow(input["label"][0, 0, :, :, 80])
    plt.subplot(1, 3, 3)
    plt.title(f"output {title}")
    plt.imshow(torch.argmax(output, dim=1).detach().cpu()[0, :, :, 80])
    plt.show()


def inferenceOverDataSpace_do(
    network: ModelData,
    data_loader: DataLoader,
    data_transforms: Compose = None,
    doPlots: bool = False,
    earlyExit: int = 0,
) -> tuple[torch.Tensor, float]:
    network.model.load_state_dict(torch.load(str(metrics.modelPth)))
    network.model.eval()
    output: torch.Tensor = torch.Tensor(())
    input: torch.Tensor
    postData: list[dict[str, Any]] = []
    diceMetric: DiceMetric = DiceMetric(include_background=True, reduction="mean")
    metric: float = 0
    with torch.no_grad():
        for i, inputdata in enumerate(data_loader):
            roi_size = (160, 160, 160)
            sw_batch_size = 4
            input = inputdata["image"].to(network.device)
            output = sliding_window_inference(
                input,
                roi_size,
                sw_batch_size,
                network.model,
            )
            if doPlots:
                IO_plotsDo(inputdata, output, f"{i}")
            if earlyExit and earlyExit == i:
                break
            if data_transforms:
                postData = [data_transforms(j) for j in decollate_batch(inputdata)]
                postOutputs, postLabels = from_engine(["pred", "label"])(postData)
                diceMetric(y_pred=postOutputs, y=postLabels)
        if data_transforms:
            metric = diceMetric.aggregate().item()
            diceMetric.reset()

    return output, metric


def inference_validateCheck(network: ModelData, validationLoader: DataLoader) -> None:
    doPlots: bool = True
    exitOnLoop2: int = 2
    transformNone = None
    inferenceOverDataSpace_do(
        network, validationLoader, transformNone, doPlots, exitOnLoop2
    )


def inference_diceMetricGet(
    network: ModelData, validationLoader: DataLoader, transforms: Compose
) -> float:
    plotFalse: bool = False
    output: torch.Tensor
    metric: float
    output, metric = inferenceOverDataSpace_do(
        network, validationLoader, transforms, plotFalse
    )
    return metric


@chris_plugin(
    parser=parser,
    title="Spleen 3D image segmentation (MONAI)",
    category="",  # ref. https://chrisstore.co/plugins
    min_memory_limit="16Gi",  # supported units: Mi, Gi
    min_cpu_limit="1000m",  # millicores, e.g. "1000m" = 1 CPU core
    min_gpu_limit=0,  # set min_gpu_limit=1 to enable GPU
)
def main(options: Namespace, inputdir: Path, outputdir: Path):
    """
    *ChRIS* plugins usually have two positional arguments: an **input directory** containing
    input files and an **output directory** where to write output files. Command-line arguments
    are passed to this main method implicitly when ``main()`` is called below without parameters.

    :param options: non-positional arguments parsed by the parser given to @chris_plugin
    :param inputdir: directory containing (read-only) input files
    :param outputdir: directory where to write output files
    """

    # pudb.set_trace()


    # network: ModelData = ModelData(options)
    # trainingResults: trainingMetrics = trainingMetrics(options)

    print(DISPLAY_TITLE)
    print_config()
    neuralNet: NeuralNet = NeuralNet(options)

    trainingDataSet, validationSet = inputFiles_splitInto_train_validate(
        options, inputdir
    )

    # set_determinism(seed=0)
    f_TrainingTransforms: list[Callable[dict[str, Any]], dict[str, Any]] = [
             f_LoadImaged,
             f_EnsureChannelFirstd,
             f_ScaleIntensityRanged,
             f_CropForegroundd,
             f_Orientationd,
             f_Spaceingd
    ]
    trainingDataTransforms: Compose = transforms_build(
        f_TrainingTransforms + [f_RandCropByPosNegLabeld]
    )
    validationTransforms: Compose = transforms_build(f_TrainingTransforms)

    transformsOK: bool = transforms_check(
        outputdir, validationSet, validationTransforms
    )
    if not transformsOK:
        sys.exit(1)

    neuralNet.trainingSpace = loaderCache_create(
        trainingDataSet, trainingDataTransforms
    )
    neuralNet.validationSpace = loaderCache_create(
        validationSet, validationTransforms
    )
    neuralNet.train()

    # trainingCache: LoaderCache
    # trainingLoader: DataLoader
    #
    # trainingCache, trainingLoader = loaderCache_create(
    #     trainingDataSet, trainingDataTransforms
    # )
    #
    # validationCache: LoaderCache
    # validationLoader: DataLoader
    # validationCache, validationLoader = loaderCache_create(
    #     validationSet, validationTransforms
    # )
    #
    trainingResults = training_do(
        trainingResults, network, trainingCache, validationCache, outputdir
    )
    plot_trainingMetrics(trainingResults, outputdir / "trainingResults.jpg")

    inference_validateCheck(network, validationLoader)

    postTraining_transforms: Compose = transforms_build(
        [
            f_Invertd(validationTransforms),
            f_predAsDiscreted,
            f_labelAsDiscreted
        ]
    )


if __name__ == "__main__":
    main()
