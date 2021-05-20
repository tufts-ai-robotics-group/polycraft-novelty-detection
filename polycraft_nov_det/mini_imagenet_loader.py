import importlib.resources
from pathlib import Path
import shutil
import urllib.request

from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder

from polycraft_nov_data.dataset_transforms import collate_patches, filter_dataset
from polycraft_nov_data.image_transforms import SamplePatch, ToPatches


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
RESIZE_SHAPE = (3, 64, 64)
PATCH_SHAPE = (3, 32, 32)


def download_datasets():
    """Download Mini-ImageNet datasets if not downloaded
    """
    DATASET_ROOT.mkdir(exist_ok=True)
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


def mini_imagenet_dataset(transform=None):
    download_datasets()
    return ImageFolder(DATASET_ROOT, transform=transform)


def mini_imagenet_dataloaders(batch_size=32, shuffle=True, all_patches=False):
    """torch DataLoaders for patched Mini-ImageNet datasets

    Args:
        batch_size (int, optional): batch_size for DataLoaders. Defaults to 32.
        shuffle (bool, optional): shuffle for DataLoaders. Defaults to True.
        all_patches (bool, optional): Whether to replace batches with all patches from an image.
                                      Defaults to False.

    Returns:
        (DataLoader, DataLoader, DataLoader): Mini-ImageNet train, validation, and test sets.
                                              Contains batches of (3, 32, 32) images,
                                              with values 0-1.
    """
    # if using patches, override batch dim to hold the set of patches
    if not all_patches:
        collate_fn = None
        transform = TrainPreprocess()
    else:
        batch_size = None
        collate_fn = collate_patches
        transform = TestPreprocess()
    # get the dataset
    dataset = mini_imagenet_dataset(transform)
    # split into datasets
    train_set = filter_dataset(dataset, include_classes=[dataset.class_to_idx["train"]])
    valid_set = filter_dataset(dataset, include_classes=[dataset.class_to_idx["val"]])
    test_set = filter_dataset(dataset, include_classes=[dataset.class_to_idx["test"]])
    # get DataLoaders for datasets
    num_workers = 4
    dataloader_kwargs = {
        "num_workers": num_workers,
        "prefetch_factor": batch_size//num_workers,
        "persistent_workers": True,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": collate_fn,
    }
    return (data.DataLoader(train_set, **dataloader_kwargs),
            data.DataLoader(valid_set, **dataloader_kwargs),
            data.DataLoader(test_set, **dataloader_kwargs))


class TrainPreprocess:
    def __init__(self, image_scale=1.0):
        """Image preprocessing for training

        Args:
            image_scale (float, optional): Scaling to apply to image. Defaults to 1.0.
        """
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(RESIZE_SHAPE[1:]),
            SamplePatch(PATCH_SHAPE),
        ])

    def __call__(self, tensor):
        return self.preprocess(tensor)


class TestPreprocess:
    def __init__(self, image_scale=1.0):
        """Image preprocessing for testing

        Args:
            image_scale (float, optional): Scaling to apply to image. Defaults to 1.0.
        """
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(RESIZE_SHAPE[1:]),
            ToPatches(PATCH_SHAPE),
        ])

    def __call__(self, tensor):
        return self.preprocess(tensor)
