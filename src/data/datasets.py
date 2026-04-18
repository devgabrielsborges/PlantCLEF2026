import itertools
import os

from kornia.contrib import compute_padding, extract_tensor_patches
from PIL import Image
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    def __init__(self, patches, transform=None):
        self.patches = patches.squeeze(0)
        self.transform = transform

    def __len__(self):
        return self.patches.size(0)

    def __getitem__(self, idx):
        patch = self.patches[idx]

        if self.transform:
            patch = self.transform(patch)
        return patch


class TestDataset(Dataset):
    def __init__(
        self, image_folder, patch_size=518, stride=259, transform=None, use_pad=False
    ):
        self.image_folder = image_folder
        self.image_paths = [
            os.path.join(image_folder, f) for f in os.listdir(image_folder)
        ]
        self.transform = transform
        self.use_pad = use_pad
        self.patch_size = patch_size
        self.stride = stride

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image).unsqueeze(0)

        h, w = image.shape[-2:]

        if self.use_pad:
            pad = compute_padding(
                original_size=(h, w), window_size=self.patch_size, stride=self.stride
            )
            patches = extract_tensor_patches(
                image, self.patch_size, self.stride, padding=pad
            )
        else:
            patches = extract_tensor_patches(image, self.patch_size, self.stride)

        return patches, image_path


class DomainDataset(Dataset):
    """
    General dataset for DANN that returns an image and its domain label.
    Supports labeled (source) and unlabeled (target) data.
    """

    def __init__(self, root_dir, df, domain_label, transform=None, is_source=True):
        self.root_dir = root_dir
        if self.root_dir is None:
            raise ValueError(
                "root_dir (data path) cannot be None. Ensure environment variables are loaded."
            )
        self.df = df
        self.domain_label = domain_label
        self.transform = transform
        self.is_source = is_source

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = (
            row["image_path"] if "image_path" in row else row["image_id"] + ".jpg"
        )
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.is_source:
            label = row["species_id"]
        else:
            # Target domain data is unlabeled for domain adaptation
            label = -1

        if self.transform:
            image = self.transform(image)

        return image, label, self.domain_label


def get_interleaved_loader(source_loader, target_loader):
    """
    Helper function to interleave source and target loaders.
    Cycles through the shorter loader to match the longer one.
    """
    for source_batch, target_batch in zip(
        source_loader, itertools.cycle(target_loader)
    ):
        yield source_batch, target_batch
