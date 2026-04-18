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
