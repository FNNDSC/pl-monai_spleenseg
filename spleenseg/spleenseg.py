#!/usr/bin/env python

from collections.abc import Iterable
from pathlib import Path
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
from dataclasses import dataclass

from monai.config.deviceconfig import print_config

import os, sys
import pudb
from typing import Any, Optional, Callable

from spleenseg.core import neuralnet
from spleenseg.transforms import transforms

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
    help="the determinism seed for training/evaluation",
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


def trainingData_prep(options: Namespace) -> list[dict[str, str]]:
    """
    Generates a list of dictionary entries, each of which is simply the name
    of an image file and its corresponding label file.
    """
    trainRaw: list[Path] = []
    trainLbl: list[Path] = []
    for group in [options.trainImageDir, options.trainLabelsDir]:
        for path in Path(options.inputdir).rglob(group):
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
    options: Namespace,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """
    Returns a list of image+label filenames to use for training and
    a list to use for validation.
    """
    trainingSpace: list[dict[str, str]] = trainingData_prep(options)
    trainingSet: list[dict[str, str]] = trainingSpace[: -options.validateSize]
    validateSet: list[dict[str, str]] = trainingSpace[-options.validateSize :]
    return trainingSet, validateSet


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

    pudb.set_trace()

    print_config()
    neuralNet: neuralnet.NeuralNet = neuralnet.NeuralNet(options)

    trainingDataSet, validationDataSet = inputFiles_splitInto_train_validate(options)

    trainingTransforms, validationTransforms = (
        transforms.trainingAndValidation_transformsSetup()
    )
    if not transforms.transforms_check(
        outputdir, validationDataSet, validationTransforms
    ):
        sys.exit(1)

    neuralNet.trainingSpace = neuralNet.loaderCache_create(
        trainingDataSet, trainingTransforms
    )
    neuralNet.validationSpace = neuralNet.loaderCache_create(
        validationDataSet, validationTransforms
    )

    neuralNet.train()
    #
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
    # trainingResults = training_do(
    #     trainingResults, network, trainingCache, validationCache, outputdir
    # )
    # plot_trainingMetrics(trainingResults, outputdir / "trainingResults.jpg")
    #
    # inference_validateCheck(network, validationLoader)
    #
    # postTraining_transforms: Compose = transforms_build(
    #     [f_Invertd(validationTransforms), f_predAsDiscreted, f_labelAsDiscreted]
    # )


if __name__ == "__main__":
    main()
