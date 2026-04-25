from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import torch


def _resolve_builder(model_name: str):
    from compressai.zoo import (
        bmshj2018_factorized,
        bmshj2018_hyperprior,
        mbt2018,
        mbt2018_mean,
    )

    builders = {
        "bmshj2018-factorized": bmshj2018_factorized,
        "bmshj2018-hyperprior": bmshj2018_hyperprior,
        "mbt2018": mbt2018,
        "mbt2018-mean": mbt2018_mean,
    }
    if model_name not in builders:
        available = ", ".join(sorted(builders))
        raise ValueError(
            f"Unsupported COMPRESSAI_MODEL '{model_name}'. Available: {available}"
        )
    return builders[model_name]


@dataclass(slots=True)
class CompressAICodec:
    enabled: bool = True
    model_name: str = "bmshj2018-factorized"
    quality: int = 3
    pretrained: bool = True
    device: str = "cuda"
    _model: object | None = field(default=None, init=False, repr=False)

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self.enabled:
            return None
        builder = _resolve_builder(self.model_name)
        model = builder(quality=self.quality, pretrained=self.pretrained).eval()
        self._model = model.to(self.device)
        return self._model

    @torch.inference_mode()
    def encode_decode_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 4:
            raise ValueError("Input tensor must be CHW or BCHW.")

        if not self.enabled:
            return tensor

        model = self._load_model()
        output = model(tensor.to(self.device))
        x_hat = output["x_hat"].clamp(0, 1).detach().cpu()
        return x_hat

    def _cache_key(self, image_path: str | Path) -> str:
        payload = (
            f"{Path(image_path)}::{self.model_name}::{self.quality}::{self.pretrained}"
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    @torch.inference_mode()
    def encode_decode_with_cache(
        self,
        image_path: str | Path,
        input_tensor: torch.Tensor,
        cache_dir: str | Path,
    ) -> torch.Tensor:
        cache_path = Path(cache_dir) / f"{self._cache_key(image_path)}.pt"
        if cache_path.exists():
            return torch.load(cache_path, map_location="cpu")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        output = self.encode_decode_tensor(input_tensor)
        torch.save(output, cache_path)
        return output
