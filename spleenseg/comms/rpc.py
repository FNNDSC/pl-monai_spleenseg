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


class Rpc:
    def __init__(self, options: Namespace):
        self.options = options

    def file_toBytes(self, uploadFile: Path) -> bytes:
        with open(str(uploadFile), "rb") as f:
            filebytes: bytes = f.read()
        return filebytes

    def onFile_infer(self, uploadFile: Path = Path("")):
        if not uploadFile.parts:
            uploadFile = self.options.NIfTIfile
        fileToSend: dict[str, bytes] = {"file": file_toBytes(uploadFile)}

        response: requests.Response = requests.post(self.options.url, files=fileToSend)

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
        "--url",
        type=str,
        default="http://localhost:2024/api/v1/spleenseg/NIfTIinference/",
        help="Endpoint to access on the hosted model server",
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
    rpc.onFile_infer()


if __name__ == "__main__":
    pudb.set_trace()
    main(sys.argv)
