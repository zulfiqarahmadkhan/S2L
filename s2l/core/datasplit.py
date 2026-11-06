"""Training / test split of an image + mask folder, saved next to the trained model.

The split is written to a JSON file so the "Test model" step evaluates exactly
the images that were kept out of training.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from s2l.core.image_files import IMAGE_EXTENSIONS


def find_pairs(folder: str, mask_suffix: str = "_cp_masks") -> list[tuple[str, str]]:
    """(image, mask) path pairs: every image ``X.ext`` with a mask ``X<mask_suffix>.ext``."""
    folder = Path(folder)
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    by_stem = {p.stem: p for p in files}
    pairs = []
    for p in files:
        if mask_suffix in p.stem:
            continue
        mask = by_stem.get(p.stem + mask_suffix)
        if mask is not None:
            pairs.append((str(p), str(mask)))
    return pairs


def split_pairs(pairs: list, test_percent: float, seed: int = 0) -> tuple[list, list]:
    """Shuffle reproducibly and split into (train, test); at least one image stays in each when possible."""
    if test_percent <= 0 or len(pairs) < 2:
        return list(pairs), []
    order = np.random.default_rng(seed).permutation(len(pairs))
    n_test = int(round(len(pairs) * test_percent / 100))
    n_test = min(max(1, n_test), len(pairs) - 1)
    test = [pairs[i] for i in sorted(order[:n_test])]
    train = [pairs[i] for i in sorted(order[n_test:])]
    return train, test


def split_file_for(model_path: str) -> Path:
    """StarDist models are folders (file inside); Cellpose models are files (file next to it)."""
    p = Path(model_path)
    return p / "s2l_split.json" if p.is_dir() else p.with_name(p.name + "_s2l_split.json")


def save_split(model_path: str, engine: str, train: list, test: list, start_from: str,
               settings: dict, train_losses=None) -> Path:
    info = {
        "engine": engine,
        "model_path": str(model_path),
        "start_from": start_from,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_train": len(train),
        "n_test": len(test),
        "train": [{"image": i, "mask": m} for i, m in train],
        "test": [{"image": i, "mask": m} for i, m in test],
        "settings": settings,
        "train_losses": [float(x) for x in (train_losses if train_losses is not None else [])],
    }
    path = split_file_for(model_path)
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return path


def load_split(model_path: str) -> dict | None:
    path = split_file_for(model_path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
