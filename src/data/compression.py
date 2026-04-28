from __future__ import annotations

import io
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


def resize_lanczos(image: Image.Image, size: int = 512) -> Image.Image:
    return image.convert("RGB").resize((size, size), resample=Image.Resampling.LANCZOS)


def normalize_jpeg(image: Image.Image, quality: int = 95) -> Image.Image:
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be between 1 and 100.")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def build_tensor_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ]
    )


def build_normalize_transform() -> transforms.Normalize:
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def preprocess_pil_image(
    image: Image.Image,
    image_size: int = 512,
    jpeg_normalize_enabled: bool = True,
    jpeg_quality: int = 95,
) -> torch.Tensor:
    resized = resize_lanczos(image=image, size=image_size)
    if jpeg_normalize_enabled:
        resized = normalize_jpeg(resized, quality=jpeg_quality)
    return build_tensor_transform()(resized)


def load_and_preprocess_image(
    image_path: str | Path,
    image_size: int = 512,
    jpeg_normalize_enabled: bool = True,
    jpeg_quality: int = 95,
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    return preprocess_pil_image(
        image=image,
        image_size=image_size,
        jpeg_normalize_enabled=jpeg_normalize_enabled,
        jpeg_quality=jpeg_quality,
    )

