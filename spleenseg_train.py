#!/usr/bin/env python

from pathlib import Path
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter

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

from chris_plugin import chris_plugin, PathMapper

__version__ = "1.0.0"

DISPLAY_TITLE = r"""
       _                                    _             _                                   _             _
      | |                                  (_)           | |                                 | |           (_)
 _ __ | |______ _ __ ___   ___  _ __   __ _ _   ___ _ __ | | ___  ___ _ __  ___  ___  __ _   | |_ _ __ __ _ _ _ __
| '_ \| |______| '_ ` _ \ / _ \| '_ \ / _` | | / __| '_ \| |/ _ \/ _ \ '_ \/ __|/ _ \/ _` |  | __| '__/ _` | | '_ \
| |_) | |      | | | | | | (_) | | | | (_| | | \__ \ |_) | |  __/  __/ | | \__ \  __/ (_| |  | |_| | | (_| | | | | |
| .__/|_|      |_| |_| |_|\___/|_| |_|\__,_|_| |___/ .__/|_|\___|\___|_| |_|___/\___|\__, |   \__|_|  \__,_|_|_| |_|
| |                                        ______  | |                                __/ |_____
|_|                                       |______| |_|                               |___/______|
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


def trainingData_prep(options: Namespace, inputDir: Path) -> list[dict[str, str]]:
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
    trainingSpace: list[dict[str, str]] = trainingData_prep(options, inputDir)
    trainingSet: list[dict[str, str]] = trainingSpace[: -options.validateSize]
    validateSet: list[dict[str, str]] = trainingSpace[-options.validateSize :]
    return trainingSet, validateSet


# The main function of this *ChRIS* plugin is denoted by this ``@chris_plugin`` "decorator."
# Some metadata about the plugin is specified here. There is more metadata specified in setup.py.
#
# documentation: https://fnndsc.github.io/chris_plugin/chris_plugin.html#chris_plugin
@chris_plugin(
    parser=parser,
    title="Spleen 3D image segmentation training (MONAI)",
    category="",  # ref. https://chrisstore.co/plugins
    min_memory_limit="100Mi",  # supported units: Mi, Gi
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
    pudb.set_trace()
    print(DISPLAY_TITLE)
    print_config()
    trainingSet, validateSet = inputFiles_splitInto_train_validate(options, inputdir)

    # Typically it's easier to think of programs as operating on individual files
    # rather than directories. The helper functions provided by a ``PathMapper``
    # object make it easy to discover input files and write to output files inside
    # the given paths.
    #
    # Refer to the documentation for more options, examples, and advanced uses e.g.
    # adding a progress bar and parallelism.
    mapper = PathMapper.file_mapper(
        inputdir, outputdir, glob=options.pattern, suffix=".count.txt"
    )
    for input_file, output_file in mapper:
        # The code block below is a small and easy example of how to use a ``PathMapper``.
        # It is recommended that you put your functionality in a helper function, so that
        # it is more legible and can be unit tested.
        break


if __name__ == "__main__":
    main()
