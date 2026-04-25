import numpy as np
import pytest
import torch
from PIL import Image

from src.data.compression import normalize_jpeg, preprocess_pil_image, resize_lanczos


def test_resize_lanczos_outputs_target_size():
    image = Image.fromarray(np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8))
    out = resize_lanczos(image, size=512)
    assert out.size == (512, 512)


def test_preprocess_returns_float_tensor_in_unit_range():
    image = Image.fromarray(np.random.randint(0, 255, (320, 240, 3), dtype=np.uint8))
    tensor = preprocess_pil_image(
        image=image,
        image_size=512,
        jpeg_normalize_enabled=True,
        jpeg_quality=90,
    )
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 512, 512)
    assert tensor.dtype == torch.float32
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_normalize_jpeg_rejects_invalid_quality():
    image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        normalize_jpeg(image, quality=0)

