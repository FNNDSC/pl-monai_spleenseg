#!/usr/bin/env python

from argparse import Namespace
from collections.abc import Callable
from pathlib import Path

from monai.transforms.compose import Compose

from monai.inferers.utils import sliding_window_inference
from monai.data.dataset import CacheDataset
from monai.data.dataloader import DataLoader

from monai.data.utils import decollate_batch
from monai.data.meta_tensor import MetaTensor
from monai.handlers.utils import from_engine

import torch

from typing import Any, Sequence
import numpy as np
import nibabel as nib
from nibabel.nifti1 import Nifti1Image

from spleenseg.transforms import transforms
from spleenseg.models import data
from spleenseg.plotting import plotting
import pudb
import shutil


def dictFirstKeyValue_getFromList(lst: list[dict[str, Any]], i: int = 0) -> Any:
    val: Any = None
    if len(lst):
        if lst[i]:
            key: str = list(lst[i].keys())[0]
            val = lst[i][key]
    return val


def tensor_desc(
    T: torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor], **kwargs
) -> torch.Tensor:
    """
    A simple method to "characterize" or describe a (possibly high dimensional) tensor
    as some 2D structure.
    """
    strAs: str = "meanstd"
    v1: float = 0.0
    v2: float = 0.0
    tensor: torch.Tensor = torch.Tensor([v1, v2])
    for k, v in kwargs.items():
        if k.lower() == "desc":
            strAs = v
    Tt: torch.Tensor = torch.as_tensor(T)
    match strAs:
        case "meanstd":
            tensor = torch.Tensor([Tt.mean().item(), Tt.std().item()])
        case "l1l2":
            tensor = torch.Tensor(
                [Tt.abs().sum().item(), Tt.pow(2).sum().sqrt().item()]
            )
        case "minmax":
            tensor = torch.Tensor([Tt.min().item(), Tt.max().item()])
        case "simplified":
            tensor = Tt.mean(dim=(1, 2), keepdim=True)
    return tensor


class NeuralNet:
    def __init__(self, options: Namespace):
        self.network: data.ModelParams = data.ModelParams(options)
        self.trainingParams: data.TrainingParams = data.TrainingParams(options)

        self.input: torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]
        self.output: torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]
        self.target: torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]

        self.trainingLog: data.TrainingLog = data.TrainingLog()
        self.trainingEpoch: int = 0
        self.whileTrainingNIfTIsaved: bool = False
        self.whileTrainingValidationNIfTIsaved: bool = False
        self.postTrainingBestModelNIfTIsaved: bool = False
        self.postTrainingSpacingsNIfTIsaved: bool = False
        self.novelInferenceNIfTIsaved: bool = False

        self.f_outputPost: Compose
        self.f_labelPost: Compose

        self.trainingFileSet: list[dict[str, str]]
        self.validationFileSet: list[dict[str, str]]
        self.testingFileSet: list[dict[str, str]]

        self.trainingSpace: data.LoaderCache
        self.validationSpace: data.LoaderCache
        self.testingSpace: data.LoaderCache

    def loaderCache_create(
        self,
        fileList: list[dict[str, str]],
        transforms: Compose,
        batch_size: int = 2,
        title: str = "",
        saveExemplar: Path = Path(""),
    ) -> data.LoaderCache:
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
        NIfTIfile: str = dictFirstKeyValue_getFromList(fileList)
        if saveExemplar.parts:
            print(f"         set exemplar:   save:  {saveExemplar}")
            shutil.copy(str(NIfTIfile), str(saveExemplar))
        if NIfTIfile is not None:
            nifti: nib.nifti1 | nib.nifti2 = nib.load(NIfTIfile)
            if len(title):
                print(f"{title}", end="")
            print(f"shape: '{NIfTIfile}': {nifti.shape}")
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

        loaderCache: data.LoaderCache = data.LoaderCache(cache=ds, loader=loader)
        return loaderCache

    def trainingTransformsAndSpace_setup(self) -> bool:
        setupOK: bool = True
        trainingTransforms: Compose
        validationTransforms: Compose
        trainingTransforms, validationTransforms = (
            transforms.trainingAndValidation_transformsSetup()
        )
        if not transforms.transforms_check(
            Path(self.network.options.outputdir),
            self.validationFileSet,
            validationTransforms,
        ):
            return False

        self.trainingSpace = self.loaderCache_create(
            self.trainingFileSet,
            trainingTransforms,
            2,
            "training set exemplar:   ",
            self.trainingParams.preTrainingIO / "input.nii.gz",
        )
        self.validationSpace = self.loaderCache_create(
            self.validationFileSet,
            validationTransforms,
            1,
            "validation set exemplar: ",
            self.trainingParams.preTrainingIO / "validation.nii.gz",
        )
        return setupOK

    def testingTransformsAndSpace_setup(self) -> bool:
        testingTransforms: Compose
        testingTransforms = transforms.inferenceUse_transforms()
        self.testingSpace = self.loaderCache_create(
            self.testingFileSet, testingTransforms, 1
        )
        return True

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

    def metaTensor_toNIfTI(self, metaTensor: MetaTensor, savefile: Path):
        singleVolume: np.ndarray
        if metaTensor.dim() == 5:
            singleVolume = metaTensor[0, 0].cpu().numpy()
        if metaTensor.dim() == 3:
            singleVolume = metaTensor.cpu().numpy()
        affine: np.ndarray = np.eye(4)
        niftiVolume: Nifti1Image = Nifti1Image(singleVolume, affine)
        nib.save(niftiVolume, savefile)

    def sample_showInfo(
        self,
        tensor: list[MetaTensor | torch.Tensor],
        niftiTelemetry: data.NIfTItelemetry,
        saveVolumes: bool = True,
    ):
        # pudb.set_trace()
        for T, txt, savefile in zip(
            tensor, niftiTelemetry.info, niftiTelemetry.savePath
        ):
            print(f"{txt} shape: {T.shape}")
            if self.trainingParams.options.logTransformVols and saveVolumes:
                print(f"{txt} save:  {savefile}")
                self.metaTensor_toNIfTI(T, savefile)

    def sample_showSummary(
        self, sample: int, sample_loss: float, trainingSpace: data.LoaderCache
    ):
        if (
            trainingSpace.cache is not None
            and trainingSpace.loader.batch_size is not None
        ):
            print(
                f"    training run {sample:02}/"
                f"{len(trainingSpace.cache) // trainingSpace.loader.batch_size}, "
                f"sample loss: {sample_loss:.4f}"
            )

    def train_overSampleSpace_retLoss(self, trainingSpace: data.LoaderCache) -> float:
        sample: int = 0
        sample_loss: float = 0.0
        total_loss: float = 0.0
        niftiTelemetry: data.NIfTItelemetry = data.NIfTItelemetry()
        for trainingInstance in trainingSpace.loader:
            sample += 1
            self.input, self.target = (
                trainingInstance["image"].to(self.network.device),
                trainingInstance["label"].to(self.network.device),
            )
            if sample == 1:
                niftiTelemetry.info = ["in training image", "in training label"]
                niftiTelemetry.savePath = [
                    self.trainingParams.whileTrainingIO / "input.nii.gz",
                    self.trainingParams.whileTrainingIO / "label.nii.gz",
                ]
                self.sample_showInfo(
                    [self.input, self.target],
                    niftiTelemetry,
                    not self.whileTrainingNIfTIsaved,
                )
                self.whileTrainingNIfTIsaved = True
            sample_loss = self.evalAndCorrect()
            total_loss += sample_loss
            self.sample_showSummary(sample, sample_loss, trainingSpace)
        total_loss /= sample
        return total_loss

    def train_evaluateCurrentEpochWithTelemetry(self):
        print("evaluating current model at this epoch")
        niftiTelemetry: data.NIfTItelemetry = data.NIfTItelemetry()
        niftiTelemetry.info = [
            "in training validation inference input",
            "in training validation inference output",
        ]
        niftiTelemetry.savePath = [
            self.trainingParams.whileTrainingValidation / "input.nii.gz",
            self.trainingParams.whileTrainingValidation / "output.nii.gz",
        ]
        self.slidingWindowInference_do(
            self.validationSpace, self.validate, niftiTelemetry
        )

    def train(self, useModelFile: Path | None = None):
        self.f_outputPost = transforms.transforms_build(
            [transforms.f_AsDiscreteArgMax()]
        )
        # niftiTelemetry: data.NIfTItelemetry = data.NIfTItelemetry()
        self.f_labelPost = transforms.transforms_build([transforms.f_AsDiscrete()])
        self.trainingEpoch = 0
        epoch_loss: float = 0.0
        if useModelFile is not None:
            self.network.model.load_state_dict(torch.load(str(useModelFile)))
        for self.trainingEpoch in range(self.trainingParams.max_epochs):
            print("-" * 10)
            print(
                f"epoch {self.trainingEpoch+1:03} / {self.trainingParams.max_epochs:03}"
            )
            self.network.model.train()
            epoch_loss = self.train_overSampleSpace_retLoss(self.trainingSpace)
            print(f"epoch {self.trainingEpoch+1:03}, average loss: {epoch_loss:.4f}")
            self.trainingLog.loss_per_epoch.append(epoch_loss)
            if (self.trainingEpoch + 1) % self.trainingParams.val_interval == 0:
                self.train_evaluateCurrentEpochWithTelemetry()
                # print("evaluating current model")
                # niftiTelemetry.info = [
                #     "in training validation inference input",
                #     "in training validation inference output",
                # ]
                # niftiTelemetry.savePath = [
                #     self.trainingParams.whileTrainingValidation / "input.nii.gz",
                #     self.trainingParams.whileTrainingValidation / "output.nii.gz",
                # ]
                # self.slidingWindowInference_do(
                #     self.validationSpace, self.validate, niftiTelemetry
                # )
        print("-" * 10)
        print(
            "Training complete: "
            f"best metric: {self.trainingLog.best_metric:.4f} "
            f"at epoch: {self.trainingLog.best_metric_epoch}"
        )

    def inference_metricsProcess(self) -> float:
        metric: float = self.network.dice_metric.aggregate().item()  # type: ignore
        self.trainingLog.metric_per_epoch.append(metric)
        self.network.dice_metric.reset()
        if metric > self.trainingLog.best_metric:
            self.trainingLog.best_metric = metric
            self.trainingLog.best_metric_epoch = self.trainingEpoch + 1
            torch.save(
                self.network.model.state_dict(), str(self.trainingParams.modelPth)
            )
            print("  (saved new best metric model)")
        print(
            f"    current mean dice: {metric:.4f}"
            f"\n       best mean dice: {self.trainingLog.best_metric:.4f} "
            f"\n           best epoch: {self.trainingLog.best_metric_epoch:03} "
        )
        return metric

    def diceMetric_do(
        self, outputPostProc: list[MetaTensor], truth: torch.Tensor
    ) -> torch.Tensor:
        labelPostProc = [
            self.f_labelPost(i)
            for i in decollate_batch(truth)  # type: ignore[arg-type]
        ]
        Tdm: torch.Tensor = self.network.dice_metric(
            y_pred=outputPostProc,  # type: ignore
            y=labelPostProc,  # type: ignore
        )
        return Tdm

    def validate(
        self,
        sample: dict[str, MetaTensor | torch.Tensor | int],
        space: data.LoaderCache,
        result: torch.Tensor,
        telemetry: data.NIfTItelemetry | None = None,
    ) -> float:
        """
        Callback method called in the inference stage.


        Given a 'sample' from a LoaderCache iteration, a data space containing
        the sample, the sample 'index', and the result, perform some validation.
        """
        metric: float = -1.0
        outputPostProc: list[MetaTensor] = [
            self.f_outputPost(i)
            for i in decollate_batch(result)  # type: ignore[arg-type]
        ]
        if isinstance(sample["label"], torch.Tensor):
            Tdm: torch.Tensor = self.diceMetric_do(
                outputPostProc, sample["label"].to(self.network.device)
            )
            if space.loader.batch_size:
                print(
                    f"  validation run {sample['index']:02}/"
                    f"{len(space.cache) // space.loader.batch_size:02}, "
                    f"dice metric: {Tdm}"
                )
                if sample["index"] == len(space.cache) // space.loader.batch_size:
                    if telemetry is not None:
                        if isinstance(
                            sample["input"], (MetaTensor | torch.Tensor)
                        ) and isinstance(sample["output"], (MetaTensor, torch.Tensor)):
                            self.sample_showInfo(
                                [sample["input"], sample["output"]],
                                telemetry,
                                not self.whileTrainingValidationNIfTIsaved,
                            )
                            self.whileTrainingValidationNIfTIsaved = True
                    metric = self.inference_metricsProcess()
        return metric

    def slidingWindowInference_do(
        self,
        inferSpace: data.LoaderCache,
        f_callback: (
            Callable[
                [
                    dict[str, MetaTensor | torch.Tensor | int],
                    data.LoaderCache,
                    torch.Tensor,
                    data.NIfTItelemetry | None,
                ],
                float,
            ]
            | None
        ) = None,
        telemetry: data.NIfTItelemetry | None = None,
    ) -> float:
        metric: float = 0.0
        self.network.model.eval()
        index: int = 0
        with torch.no_grad():
            for sample in inferSpace.loader:
                index += 1
                input: torch.Tensor = sample["image"].to(self.network.device)
                roi_size: tuple[int, int, int] = (160, 160, 160)
                sw_batch_size: int = 4
                outputRaw: torch.Tensor = torch.as_tensor(
                    sliding_window_inference(
                        input, roi_size, sw_batch_size, self.network.model
                    )
                )
                sample["index"] = index
                sample["input"] = input.cpu()
                sample["output"] = outputRaw.cpu()
                if f_callback is not None:
                    metric = f_callback(sample, inferSpace, outputRaw, telemetry)
        return metric

    def plot_bestModel(
        self,
        sample: dict[str, MetaTensor | torch.Tensor | int],
        space: data.LoaderCache,
        result: torch.Tensor,
        telemetry: data.NIfTItelemetry | None = None,
    ) -> float:
        index: int = int(sample["index"])
        print(f"Plotting output of best model applied to validation sample {index}")
        if telemetry is not None:
            input = sample["input"]
            output = sample["output"]
            if isinstance(input, (MetaTensor, torch.Tensor)) and isinstance(
                output, (MetaTensor, torch.Tensor)
            ):
                self.sample_showInfo(
                    [input, output],
                    telemetry,
                    not self.postTrainingBestModelNIfTIsaved,
                )
            self.postTrainingBestModelNIfTIsaved = True

        plotting.plot_bestModelOnValidate(
            sample,
            result,
            str(index),
            self.trainingParams.postTrainingValidation / f"bestModel-val-{index}.png",
        )
        return 0.0

    def bestModel_runOverValidationSpace(self):
        self.network.model.load_state_dict(
            torch.load(str(self.trainingParams.modelPth))
        )
        niftiTelemetry: data.NIfTItelemetry = data.NIfTItelemetry()
        niftiTelemetry.info = [
            "post train best model image",
            "post train best model output",
        ]
        niftiTelemetry.savePath = [
            self.trainingParams.postTrainingValidation / "input.nii.gz",
            self.trainingParams.postTrainingValidation / "output.nii.gz",
        ]
        self.slidingWindowInference_do(
            self.validationSpace, self.plot_bestModel, niftiTelemetry
        )

    def diceMetric_onValidationSpacing(
        self,
        sample: dict[str, MetaTensor | torch.Tensor | int],
        space: data.LoaderCache,
        result: torch.Tensor,
        telemetry: data.NIfTItelemetry | None = None,
    ) -> float:
        metric: float = -1.0
        sample["pred"] = result
        lsample = [
            self.f_outputPost(i)
            for i in decollate_batch(sample)  # type: ignore[arg-type]
        ]
        predictions: torch.Tensor
        labels: torch.Tensor
        predictions, labels = from_engine(["pred", "label"])(lsample)
        Dm: torch.Tensor = self.network.dice_metric(
            y_pred=predictions,  # type: ignore
            y=labels,  # type: ignore
        )
        print(
            f"validation spacing {sample['index']}, best prediction dice metric: {Dm}"
        )
        if space.loader.batch_size:
            if sample["index"] == len(space.cache) // space.loader.batch_size:
                metric = self.network.dice_metric.aggregate().item()
                print(f"metric on original image spacing: {metric}")
                if telemetry is not None:
                    if isinstance(sample["image"], (MetaTensor | torch.Tensor)):
                        # pudb.set_trace()
                        self.sample_showInfo(
                            [sample["image"], sample["pred"]],
                            telemetry,
                            not self.postTrainingSpacingsNIfTIsaved,
                        )
                        self.postTrainingSpacingsNIfTIsaved = True
        return metric

    def bestModel_evaluateImageSpacings(self):
        self.network.model.load_state_dict(
            torch.load(str(self.trainingParams.modelPth))
        )
        validationOnOrigTransforms: Compose = (
            transforms.validation_transformsOnOriginal()
        )
        self.validationSpace = self.loaderCache_create(
            self.validationFileSet, validationOnOrigTransforms, 1
        )
        self.f_outputPost = transforms.transforms_build(
            [
                transforms.f_Invertd(validationOnOrigTransforms),
                transforms.f_predAsDiscreted(),
                transforms.f_labelAsDiscreted(),
            ]
        )
        niftiTelemetry: data.NIfTItelemetry = data.NIfTItelemetry()
        niftiTelemetry.info = ["image spacings input", "image spacings output"]
        niftiTelemetry.savePath = [
            self.trainingParams.postTrainingImageSpacings / "input.nii.gz",
            self.trainingParams.postTrainingImageSpacings / "output.nii.gz",
        ]
        self.slidingWindowInference_do(
            self.validationSpace, self.diceMetric_onValidationSpacing, niftiTelemetry
        )

    def inference_post(
        self,
        sample: dict[str, MetaTensor | torch.Tensor | int],
        space: data.LoaderCache,
        result: torch.Tensor,
        telemetry: data.NIfTItelemetry | None = None,
    ) -> float:
        index: int = int(sample["index"])
        print(f"[{index}]--------> novel inference")
        sample["pred"] = result
        sample = [self.f_outputPost(i) for i in decollate_batch(sample)]
        prediction = from_engine(["pred"])(sample)
        fi = transforms.f_LoadImage()
        input = fi(prediction[0].meta["filename_or_obj"])
        Ti: torch.Tensor = torch.as_tensor(input)
        Pi: torch.Tensor = prediction[0]
        plotting.plot_infer(
            Ti,
            Pi,
            f"{index}",
            Path(
                Path(self.network.options.outputdir)
                / "inference"
                / f"infer-{index}.png"
            ),
        )
        if telemetry is not None:
            self.sample_showInfo(
                [Ti, result], telemetry, not self.novelInferenceNIfTIsaved
            )
            self.novelInferenceNIfTIsaved = True

        return 0.0

    def infer_usingModel(self, model: Path):
        # Check if model exists...
        self.network.model.load_state_dict(torch.load(str(model)))
        niftiTelemetry: data.NIfTItelemetry = data.NIfTItelemetry()
        testingTransforms: Compose
        testingTransforms = transforms.inferenceUse_transforms()
        self.testingSpace = self.loaderCache_create(
            self.testingFileSet,
            testingTransforms,
            1,
            " testing set exmeplar:  ",
            self.trainingParams.novelInference / "orig.nii.gz",
        )
        self.f_outputPost = transforms.transforms_build(
            [
                transforms.f_Invertd(testingTransforms),
                transforms.f_predAsDiscreted(),
                transforms.f_SaveImaged(
                    Path(self.network.options.outputdir) / "inference"
                ),
            ]
        )
        niftiTelemetry.info = ["novel inference input ", "novel inference output"]
        niftiTelemetry.savePath = [
            self.trainingParams.novelInference / "input.nii.gz",
            self.trainingParams.novelInference / "output.nii.gz",
        ]
        self.slidingWindowInference_do(
            self.testingSpace, self.inference_post, niftiTelemetry
        )
