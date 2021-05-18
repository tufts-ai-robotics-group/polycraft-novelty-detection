import importlib.resources
from pathlib import Path
import shutil
import urllib.request


# constants related to data labels and locations
DATA_LABELS = [
    "train",
    "val",
    "test",
]
DATA_URLS = {
    "train": "https://tufts.box.com/shared/static/dp619k317ibdyi1gjpn1bm8zkpfjwwwm.tar",
    "val": "https://tufts.box.com/shared/static/ib5d824annvax7blwj4pwlb4n1lknkz4.tar",
    "test": "https://tufts.box.com/shared/static/sji8936t31pzyzrjp51j0887flhxdeib.tar",
}

with importlib.resources.path("polycraft_nov_det", "base_data") as dataset_root:
    DATASET_ROOT = Path(dataset_root) / Path("MiniImageNet")
DATA_PATHS = {label: DATASET_ROOT / Path(label) for label in DATA_LABELS}
# constants related to shape of data
IMAGE_SHAPE = (3, 84, 84)
PATCH_SHAPE = (3, 32, 32)


def download_datasets():
    """Download Mini-ImageNet datasets if not downloaded
    """
    for label, data_path in DATA_PATHS.items():
        # make the directory if it doesn't exist
        data_path.mkdir(exist_ok=True)
        archive_path = data_path / Path(label + ".tar")
        # assume data is downloaded if contents exist and only download otherwise
        if len(list(data_path.iterdir())) == 0:
            # download, extract, and delete archive of the data
            urllib.request.urlretrieve(DATA_URLS[label], archive_path)
            shutil.unpack_archive(archive_path, DATASET_ROOT)
            archive_path.unlink()
