#!/usr/bin/env python

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

import torch
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.utils.misc import first
from monai.transforms.post.array import AsDiscrete
from monai.transforms.post.dictionary import AsDiscreted, Invertd
from monai.transforms.utility.dictionary import EnsureChannelFirstd
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged, SaveImaged
from monai.transforms.croppad.dictionary import CropForegroundd, RandCropByPosNegLabeld
from monai.transforms.spatial.dictionary import Orientationd, RandAffined, Spacingd
from monai.transforms.intensity.dictionary import ScaleIntensityRanged
from monai.transforms.io.array import LoadImage
from monai.config.type_definitions import PathLike
from typing import Any, Callable, Hashable, Mapping, Sequence
from monai.data.meta_tensor import MetaTensor
import numpy as np
from numpy import ndarray
from pathlib import Path
from spleenseg.plotting import plotting


def f_LoadImaged(
    keys: list[str] = ["image", "label"],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return LoadImaged(keys=keys)


def f_LoadImage() -> Callable[
    [PathLike | Sequence[PathLike]],
    torch.Tensor
    | Any
    | MetaTensor
    | tuple[
        torch.Tensor | Any | MetaTensor,
        dict[Any, Any]
        | Any
        | ndarray[Any, Any]
        | tuple[Any, ...]
        | list[Any]
        | bool
        | str
        | float
        | int
        | None,
    ],
]:
    return LoadImage()


def f_SaveImaged(outputDir: Path) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return SaveImaged(
        keys="pred",
        meta_keys="pred_meta_dict",
        output_dir=str(outputDir),
        output_postfix="seg",
        resample=False,
    )


def f_EnsureChannelFirstd(
    keys: list[str] = ["image", "label"],
) -> Callable[[Mapping[Hashable, torch.Tensor]], Mapping[Hashable, torch.Tensor]]:
    return EnsureChannelFirstd(keys=keys)


def f_ScaleIntensityRanged() -> (
    Callable[[dict[Hashable, NdarrayOrTensor]], dict[Hashable, NdarrayOrTensor]]
):
    return ScaleIntensityRanged(
        keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True
    )


def f_CropForegroundd(
    keys: list[str] = ["image", "label"],
) -> Callable[[Mapping[Hashable, torch.Tensor]], Mapping[Hashable, torch.Tensor]]:
    return CropForegroundd(keys=keys, source_key="image", allow_smaller=True)


def f_Orientationd(
    keys: list[str] = ["image", "label"],
) -> Callable[[Mapping[Hashable, torch.Tensor]], Mapping[Hashable, torch.Tensor]]:
    return Orientationd(keys=keys, axcodes="RAS")


def f_Spaceingd(
    keys: list[str] = ["image", "label"],
    mode: tuple[str, ...] = ("bilinear", "nearest"),
) -> Callable[[Mapping[Hashable, torch.Tensor]], Mapping[Hashable, torch.Tensor]]:
    return Spacingd(
        keys=keys,
        pixdim=(1.5, 1.5, 2.0),
        mode=mode,
    )


def f_RandCropByPosNegLabeld() -> (
    Callable[[Mapping[Hashable, torch.Tensor]], list[dict[Hashable, torch.Tensor]]]
):
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


def f_RandAffined() -> (
    Callable[[Mapping[Hashable, torch.Tensor]], dict[Hashable, NdarrayOrTensor]]
):
    return RandAffined(
        keys=["image", "label"],
        mode=("bilinear", "nearest"),
        prob=1.0,
        spatial_size=(96, 96, 96),
        rotate_range=(0, 0, np.pi / 15),
        scale_range=(0.1, 0.1, 0.1),
    )


def f_Invertd(
    transform: Compose,
) -> Callable[[Mapping[Hashable, Any]], dict[Hashable, Any]]:
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


def f_predAsDiscreted() -> (
    Callable[[dict[Hashable, NdarrayOrTensor]], dict[Hashable, NdarrayOrTensor]]
):
    return AsDiscreted(keys="pred", argmax=True, to_onehot=2)


def f_labelAsDiscreted() -> (
    Callable[[dict[Hashable, NdarrayOrTensor]], dict[Hashable, NdarrayOrTensor]]
):
    return AsDiscreted(keys="label", to_onehot=2)


def f_AsDiscreteArgMax() -> Callable[[NdarrayOrTensor], NdarrayOrTensor]:
    return AsDiscrete(argmax=True, to_onehott=2)


def f_AsDiscrete() -> Callable[[NdarrayOrTensor], NdarrayOrTensor]:
    return AsDiscrete(to_onehot=2)


def transforms_build(f_transform: list[Callable]) -> Compose:
    transforms: Compose = Compose(f_transform)
    return transforms


def trainingAndValidation_transformsSetup() -> tuple[Compose, Compose]:
    baseTrainingTransforms: list = [
        f_LoadImaged(),
        f_EnsureChannelFirstd(),
        f_ScaleIntensityRanged(),
        f_CropForegroundd(),
        f_Orientationd(),
        f_Spaceingd(),
    ]

    trainingTransforms: Compose = transforms_build(
        baseTrainingTransforms + [f_RandCropByPosNegLabeld()]
    )

    validationTransforms: Compose = transforms_build(baseTrainingTransforms)

    return trainingTransforms, validationTransforms


def validation_transformsOnOriginal() -> Compose:
    transforms: list = [
        f_LoadImaged(),
        f_EnsureChannelFirstd(),
        f_Orientationd(["image"]),
        f_Spaceingd(["image"], tuple(["bilinear"])),
        f_ScaleIntensityRanged(),
        f_CropForegroundd(["image"]),
    ]
    return transforms_build(transforms)


def inferenceUse_transforms() -> Compose:
    transforms: list = [
        f_LoadImaged(["image"]),
        f_EnsureChannelFirstd(["image"]),
        f_Orientationd(["image"]),
        f_Spaceingd(["image"], tuple(["bilinear"])),
        f_ScaleIntensityRanged(),
        f_CropForegroundd(["image"]),
    ]
    return transforms_build(transforms)


def transforms_check(
    outputdir: Path, files: list[dict[str, str]], transforms: Compose
) -> bool:
    check_ds: Dataset = Dataset(data=files, transform=transforms)
    check_loader: DataLoader = DataLoader(check_ds, batch_size=1)
    check_data: Any | None = first(check_loader)
    if not check_data:
        return False
    image, label = (check_data["image"][0][0], check_data["label"][0][0])
    print("")
    print("Checking transforms... :")
    print(f"sample 'image' transform shape: {image.shape}")
    print(f"sample 'label' transform shape: {label.shape}")
    plotting.plot_imageAndLabel(image, label, outputdir / "exemplar_image_label.jpg")
    return True
