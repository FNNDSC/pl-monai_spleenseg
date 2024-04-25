#!/usr/bin/env python

from pathlib import Path
from monai.data.meta_tensor import MetaTensor
import torch
import matplotlib.pyplot as plt
import pudb
from spleenseg.models import data


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
    plt.clf()


def plot_trainingMetrics(
    log: data.TrainingLog, training: data.TrainingParams, savefile: Path
) -> None:
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(log.loss_per_epoch))]
    y = log.loss_per_epoch
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Validation Mean Dice")
    x = [training.val_interval * (i + 1) for i in range(len(log.metric_per_epoch))]
    y = log.metric_per_epoch
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.savefig(str(savefile))
    plt.clf()


def plot_bestModelOnValidate(
    input: dict[str, torch.Tensor | MetaTensor | int],
    output: torch.Tensor,
    title: str,
    savefile: Path,
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
    plt.savefig(str(savefile))
    plt.clf()


def plot_infer(
    input: torch.Tensor, result: torch.Tensor, title: str, savefile: Path
) -> None:
    plt.figure("Infer", (18, 6))
    plt.subplot(1, 2, 1)
    plt.title(f"input: {title}")
    plt.imshow(input[:, :, 20], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title(f"result: {title}")
    plt.imshow(result.detach().cpu()[1, :, :, 20])
    plt.savefig(str(savefile))
    plt.clf()
