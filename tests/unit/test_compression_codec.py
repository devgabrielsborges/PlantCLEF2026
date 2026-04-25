from pathlib import Path

import torch

from src.models.compression import CompressAICodec


class _DummyModel:
    def eval(self):
        return self

    def to(self, _device):
        return self

    def __call__(self, tensor):
        return {"x_hat": tensor * 0.5}


def test_encode_decode_returns_same_shape_when_enabled():
    codec = CompressAICodec(
        enabled=True,
        model_name="bmshj2018-factorized",
        quality=1,
        pretrained=False,
        device="cpu",
    )
    codec._model = _DummyModel()

    x = torch.rand(3, 32, 32)
    y = codec.encode_decode_tensor(x)
    assert y.shape == (1, 3, 32, 32)
    assert float(y.max()) <= 1.0


def test_encode_decode_passthrough_when_disabled():
    codec = CompressAICodec(enabled=False)
    x = torch.rand(3, 16, 16)
    y = codec.encode_decode_tensor(x)
    assert y.shape == (1, 3, 16, 16)
    assert torch.allclose(y, x.unsqueeze(0))


def test_encode_decode_with_cache_reuses_saved_tensor(tmp_path):
    codec = CompressAICodec(enabled=False)
    x = torch.rand(3, 12, 12)
    cache_dir = tmp_path / "cache"

    first = codec.encode_decode_with_cache("image.jpg", x, cache_dir)
    second = codec.encode_decode_with_cache("image.jpg", x * 0.0, cache_dir)

    assert torch.allclose(first, second)
    cached_files = list(Path(cache_dir).glob("*.pt"))
    assert len(cached_files) == 1
