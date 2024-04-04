#!/usr/bin/env python

from collections.abc import Iterable
from pathlib import Path
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
from dataclasses import dataclass
from dataclasses import dataclass

import os
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
from typing import Any, Optional
import numpy as np


from chris_plugin import chris_plugin, PathMapper

__version__ = "1.0.0"

DISPLAY_TITLE = r"""

███████╗██████╗ ██╗     ███████╗███████╗███╗   ██╗███████╗███████╗ ██████╗      ████████╗██████╗  █████╗ ██╗███╗   ██╗
██╔════╝██╔══██╗██║     ██╔════╝██╔════╝████╗  ██║██╔════╝██╔════╝██╔════╝      ╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║
███████╗██████╔╝██║     █████╗  █████╗  ██╔██╗ ██║███████╗█████╗  ██║  ███╗        ██║   ██████╔╝███████║██║██╔██╗ ██║
╚════██║██╔═══╝ ██║     ██╔══╝  ██╔══╝  ██║╚██╗██║╚════██║██╔══╝  ██║   ██║        ██║   ██╔══██╗██╔══██║██║██║╚██╗██║
███████║██║     ███████╗███████╗███████╗██║ ╚████║███████║███████╗╚██████╔╝███████╗██║   ██║  ██║██║  ██║██║██║ ╚████║
╚══════╝╚═╝     ╚══════╝╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝ ╚═════╝ ╚══════╝╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝
"""


parser = ArgumentParser(
    description="""
A ChRIS DS plugin based on Project MONAI 3D Spleen Segmentation.
This plugin performs the training component, with some gentle
refactoring and pervasive type hinting.
    """,
    formatter_class=ArgumentDefaultsHelpFormatter,
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
class Model:
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
    loss_function: DiceLoss = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer: torch.optim.Adam = None
    dice_metric: DiceMetric = DiceMetric(include_background=False, reduction="mean")

    def __init__(self, options: Namespace):
        self.options = options
        if options is not None:
            self.device = torch.device(self.options.device)
            self.model = self.model.to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), 1e-4)


@dataclass
class LoaderCache:
    loader: DataLoader
    cache: CacheDataset


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


def transforms_setup(
    randCropByPosNegLabeld: bool = False, randAffined: bool = False
) -> Compose:
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
    composeList: list = []
    composeList.append(LoadImaged(keys=["image", "label"]))
    composeList.append(EnsureChannelFirstd(keys=["image", "label"]))
    composeList.append(
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        )
    )
    composeList.append(CropForegroundd(keys=["image", "label"], source_key="image"))
    composeList.append(Orientationd(keys=["image", "label"], axcodes="RAS"))
    composeList.append(
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        )
    )
    if randCropByPosNegLabeld:
        composeList.append(
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            )
        )
    if randAffined:
        composeList.append(
            RandAffined(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=1.0,
                spatial_size=(96, 96, 96),
                rotate_range=(0, 0, np.pi / 15),
                scale_range=(0.1, 0.1, 0.1),
            )
        )
    transforms: Compose = Compose(composeList)
    return transforms


def validationTransforms_setup() -> Compose:
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
        ]
    )
    return val_transforms


def validationTransforms_check(
    validationFiles: list[dict[str, str]], validationTransforms: Compose
) -> tuple[torch.Tensor, torch.Tensor]:
    check_ds: Dataset = Dataset(data=validationFiles, transform=validationTransforms)
    check_loader: DataLoader = DataLoader(check_ds, batch_size=1)
    check_data: Any | None = first(check_loader)
    image, label = (check_data["image"][0][0], check_data["label"][0][0])
    return image, label


def plot_inputAndLabel(
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


def loaders_create(
    trainingList: list[dict[str, str]],
    trainingTransforms: Compose,
    validationList: list[dict[str, str]],
    validationTransforms,
) -> tuple[LoaderCache, LoaderCache]:
    """
    ## Define CacheDataset and DataLoader for training and validation

    Here we use CacheDataset to accelerate training and validation process, it's 10x faster than the regular Dataset.
    To achieve best performance, set `cache_rate=1.0` to cache all the data, if memory is not enough, set lower value.
    Users can also set `cache_num` instead of `cache_rate`, will use the minimum value of the 2 settings.
    And set `num_workers` to enable multi-threads during caching.
    If want to to try the regular Dataset, just change to use the commented code below.
    """
    train_ds: CacheDataset = CacheDataset(
        data=trainingList, transform=trainingTransforms, cache_rate=1.0, num_workers=4
    )
    # train_ds = Dataset(data=train_files, transform=train_transforms)

    # use batch_size=2 to load images and use RandCropByPosNegLabeld
    # to generate 2 x 4 images for network training
    train_loader: DataLoader = DataLoader(
        train_ds, batch_size=2, shuffle=True, num_workers=4
    )

    val_ds: CacheDataset = CacheDataset(
        data=validationList,
        transform=validationTransforms,
        cache_rate=1.0,
        num_workers=4,
    )
    # val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader: DataLoader = DataLoader(val_ds, batch_size=1, num_workers=4)
    trainMeta: LoaderCache = LoaderCache(cache=train_ds, loader=train_loader)
    valMeta: LoaderCache = LoaderCache(cache=val_ds, loader=val_loader)
    return trainMeta, valMeta


def training_do(
    network: Model, trainMeta: LoaderCache, valMeta: LoaderCache, outputDir: Path
):
    max_epochs = 600
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        network.model.train()
        epoch_loss = 0.0
        step = 0
        for batch_data in trainMeta.loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(network.device),
                batch_data["label"].to(network.device),
            )
            network.optimizer.zero_grad()
            outputs = network.model(inputs)
            loss = network.loss_function(outputs, labels)
            loss.backward()
            network.optimizer.step()
            epoch_loss += loss.item()
            print(
                f"{step}/{int(len(trainMeta.cache)) // int(trainMeta.loader.batch_size)}, "
                f"train_loss: {loss.item():.4f}"
            )
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            network.model.eval()
            with torch.no_grad():
                for val_data in valMeta.loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(network.device),
                        val_data["label"].to(network.device),
                    )
                    roi_size = (160, 160, 160)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, network.model
                    )
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    network.dice_metric(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice result
                metric = network.dice_metric.aggregate().item()
                # reset the status for next validation round
                network.dice_metric.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(
                        network.model.state_dict(),
                        str(outputDir / "best_metric_model.pth"),
                    )
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )
    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}"
    )


@chris_plugin(
    parser=parser,
    title="Spleen 3D image segmentation training (MONAI)",
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
    network: Model = Model(options)
    trainingMeta: LoaderCache
    validationMeta: LoaderCache

    print(DISPLAY_TITLE)
    print_config()
    trainingSet, validationSet = inputFiles_splitInto_train_validate(options, inputdir)

    set_determinism(seed=0)
    doRandCropbyPosNegLabel: bool = True
    trainingTransforms: Compose = transforms_setup(doRandCropbyPosNegLabel)
    validationTransforms: Compose = transforms_setup()

    image, label = validationTransforms_check(validationSet, validationTransforms)
    print(f"image shape: {image.shape}, label shape: {label.shape}")

    plot_inputAndLabel(image, label, outputdir / "exemplar_image_label.jpg")

    trainingMeta, validationMeta = loaders_create(
        trainingSet, trainingTransforms, validationSet, validationTransforms
    )

    training_do(network, trainingMeta, validationMeta, outputdir)

if __name__ == "__main__":
    main()
