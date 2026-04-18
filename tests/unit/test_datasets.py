from unittest.mock import MagicMock

import pandas as pd
from torch.utils.data import DataLoader

from src.data.datasets import DomainDataset, get_interleaved_loader


def test_domain_dataset():
    """Verify DomainDataset returns correctly labeled images and domain labels."""
    df = pd.DataFrame({"image_path": ["img1.jpg"], "species_id": [42]})

    # Mock Image.open to avoid filesystem dependency
    from PIL import Image

    Image.open = MagicMock(return_value=Image.new("RGB", (224, 224)))

    # Source Dataset
    ds_s = DomainDataset(root_dir=".", df=df, domain_label=0, is_source=True)
    img, label, domain = ds_s[0]
    assert label == 42
    assert domain == 0

    # Target Dataset
    ds_t = DomainDataset(root_dir=".", df=df, domain_label=1, is_source=False)
    img, label, domain = ds_t[0]
    assert label == -1  # Target is unlabeled
    assert domain == 1


def test_interleaved_loader():
    """Verify get_interleaved_loader correctly cycles through the shorter loader."""
    # Dummy data
    ds_s = [("s1", 0, 0), ("s2", 1, 0), ("s3", 2, 0)]
    ds_t = [("t1", -1, 1)]

    # Use real DataLoader for testing
    loader_s = DataLoader(ds_s, batch_size=1)
    loader_t = DataLoader(ds_t, batch_size=1)

    interleaved = get_interleaved_loader(loader_s, loader_t)
    results = list(interleaved)

    # Total batches should match the longer loader (source: 3 batches)
    assert len(results) == 3

    # Check if target cycled: t1 should repeat 3 times
    for batch_s, batch_t in results:
        assert batch_t[0] == ("t1",)
