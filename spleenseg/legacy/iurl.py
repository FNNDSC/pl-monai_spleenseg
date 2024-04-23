import os
import sys
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import numpy as np

import pudb

from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
from typing import Any
from requests import post

parser = ArgumentParser(
    description="""
    A testing script for reading a NIfTI file and
    calling an inference endpoint.
    """,
    formatter_class=ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--niiFile", type=str, default="", help="NIfTI file to read")

options = parser.parse_args()


def reshape(image: Nifti1Image, desired_shape: tuple[int, int, int]) -> Nifti1Image:
    data: np.ndarray = image.get_fdata()
    reshaped_data = data[: desired_shape[0], desired_shape[1], desired_shape[2]]
    reshaped_img: Nifti1Image = Nifti1Image(reshaped_data, image.affine, image.header)
    return reshaped_img


def serialize(image) -> dict[str, Any]:
    npImage = np.array(image.get_fdata())
    npImage = np.expand_dims(npImage, axis=0)
    npImage = np.expand_dims(npImage, axis=0)
    payload: dict[str, Any] = {
        "inputs": [
            {
                "name": "x",
                "shape": list(npImage.shape),
                "datatype": "FP32",
                "data": npImage.tolist(),
            }
        ]
    }
    return payload


def response_get(payload: dict[str, Any], url: str):
    raw_response = post(url, json=payload)
    try:
        response = raw_response.json()
    except:
        print(
            f"Failed to deserialize service response.\n"
            f"Status code: {raw_response.status_code}\n"
            f"Response body: {raw_response.text}"
        )

    try:
        model_output = response["outputs"][0]["data"]
        # Convert the output data to a numpy array
        output_array = np.array(model_output)

        # Convert the output array to a NIfTI volume
        output_nifti = nib.Nifti1Image(output_array, affine=np.eye(4))

        return output_nifti
    except:
        print(
            f"Failed to extract model output from service response.\n"
            f"Service response: {response}"
        )


def main(*args) -> int:
    pudb.set_trace()
    options = parser.parse_args()
    image = nib.nifti1.load(options.niiFile)
    image = reshape(image, (226, 157, 113))
    url: str = (
        "https://v98-spleen.apps.rhods-internal.61tk.p1.openshiftapps.com/v2/models/v98/infer"
    )

    payload: dict[str, Any] = serialize(image)
    response_get(payload, url)


if __name__ == "__main__":
    sys.exit(main())
