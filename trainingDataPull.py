import tempfile
import os
from pathlib import Path
from monai.apps.utils import download_and_extract

directory: str | None = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir: Path = (
    Path(str(tempfile.TemporaryDirectory())) if directory is None else Path(directory)
)
print(f"Saving training data to {root_dir}")


# ## Download dataset
#
# Downloads and extracts the dataset.
# The dataset comes from http://medicaldecathlon.com/.
resource: str = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
md5: str = "410d4a301da4e5b2f6f86ec3ddba524e"

compressed_file: Path = root_dir / "Task09_Spleen.tar"
data_dir: Path = root_dir / "Task09_Spleen"
if not data_dir.exists():
    download_and_extract(resource, str(compressed_file), str(root_dir), md5)
