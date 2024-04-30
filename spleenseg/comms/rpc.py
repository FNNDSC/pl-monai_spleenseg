from numpy import who
from pyfiglet import FigletBuilder
import requests
from pathlib import Path
from argparse import Namespace, ArgumentParser, RawTextHelpFormatter
import sys
import pudb


def file_toBytes(uploadFile: Path) -> bytes:
    fileBytes: bytes
    with open(str(uploadFile), "rb") as f:
        fileBytes = f.read()
    return fileBytes


def file_serializeToBytes(file: Path) -> dict[str, bytes]:
    return {"file": file_toBytes(file)}


class Rpc:
    def __init__(self, options: Namespace):
        self.options = options

    def do(self) -> None:
        match self.options.do:
            case "infer":
                self.onFile_infer(Path(self.options.NIfTIfile))
            case "sendmodel":
                self.modelSend(self.options.pthModel, self.options.modelID)

    def modelSend(self, modelFile: Path, modelID: str) -> requests.Response:
        url: str = self.options.pfms + "spleenseg/modelpth/"
        resp: requests.Response = self.localFiletoURL_post(
            modelFile, url, {"modelID": modelID}
        )
        # files = file_serializeToBytes(modelFile)
        # params = {"modelID": modelID}
        # resp: requests.Response = requests.post(
        #     url,
        #     params=params,
        #     files=files,
        # )
        return resp

    def localFiletoURL_post(
        self, uploadFile: Path = Path(""), url: str = "", params: dict = {}
    ) -> requests.Response:
        fileToSend: dict[str, bytes] = file_serializeToBytes(uploadFile)
        response: requests.Response
        if params:
            response = requests.post(url, files=fileToSend, params=params)
        else:
            response = requests.post(url, files=fileToSend)
        return response

    def onFile_infer(self, uploadFile: Path = Path("")):
        if not uploadFile.parts:
            uploadFile = self.options.NIfTIfile
        url: str = self.options.pfms + "spleenseg/NIfTIinference/"
        params = {"modelID": self.options.modelID}
        response: requests.Response = self.localFiletoURL_post(uploadFile, url, params)

        if response.status_code == 200:
            result = response.json()
            print(f"Input file name: {result['inputfilename']}")
            print(f"Shape: {result['shape']}")
            print(f"Data type: {result['dtype']}")
            print(f"Data: {result['data'][:10]}...")  # Print the first 10 values
        else:
            print(f"Error: {response.status_code} - {response.text}")


def parser_setup(str_desc: str = "") -> ArgumentParser:
    description: str = ""
    if len(str_desc):
        description = str_desc
    parser = ArgumentParser(
        description=description, formatter_class=RawTextHelpFormatter
    )

    parser.add_argument(
        "--NIfTIfile",
        type=str,
        default="",
        help="NIfTI file to upload for inference on remote server",
    )

    parser.add_argument(
        "--pthModel",
        type=str,
        default="",
        help="model file in pth format to upload for inference on remote server",
    )

    parser.add_argument(
        "--modelID",
        type=str,
        default="",
        help="model identifier to be associated with --pthModel",
    )

    parser.add_argument(
        "--url",
        type=str,
        default="",
        help="endpoint to access on the hosted model server",
    )

    parser.add_argument(
        "--pfms",
        type=str,
        default="",
        help="base URL for pfms",
    )

    parser.add_argument(
        "--do",
        type=str,
        default="",
        help="action to perform -- usually pushing a model file or running inference",
    )

    return parser


def parser_interpret(parser: ArgumentParser, *args) -> Namespace:
    """
    Interpret the list space of *args, or sys.argv[1:] if
    *args is empty
    """
    options: Namespace
    if len(args):
        options = parser.parse_args(*args[1:])
    else:
        options = parser.parse_args(sys.argv[1:])
    return options


def main(*args) -> None:
    options: Namespace = parser_interpret(parser_setup(), args)
    rpc = Rpc(options)
    rpc.do()


if __name__ == "__main__":
    pudb.set_trace()
    main(sys.argv)
