from pathlib import Path
import pickle

import torch
from torch.utils.data import Dataset


def _load_embedding_object(path):
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        with open(path, "rb") as handle:
            return pickle.load(handle)


def _extract_feature_tensor(obj, feature_key):
    if isinstance(obj, dict):
        if feature_key in obj:
            value = obj[feature_key]
        else:
            raise KeyError(f"Feature key '{feature_key}' was not found. Available keys: {sorted(obj.keys())}")
    elif hasattr(obj, feature_key):
        value = getattr(obj, feature_key)
    else:
        raise KeyError(f"Feature key '{feature_key}' was not found in object of type {type(obj)}.")

    if not torch.is_tensor(value):
        value = torch.as_tensor(value)

    value = value.detach().cpu().float()
    if value.ndim == 3 and value.shape[0] == 1:
        value = value.squeeze(0)
    if value.ndim != 2:
        raise ValueError(
            f"Expected '{feature_key}' to be a 2D tensor shaped like [num_tokens, hidden_size], got {tuple(value.shape)}."
        )
    return value


class EmbeddingFeatureDataset(Dataset):
    """Simple file-backed dataset for precomputed multimodal features."""

    def __init__(self, root_dir, feature_key="h_V_last_layer", file_extensions=None):
        self.root_dir = Path(root_dir).expanduser() if root_dir is not None else None
        self.feature_key = feature_key
        self.file_extensions = tuple(file_extensions or (".pt", ".pth", ".bin", ".pkl"))
        self.files = []

        if self.root_dir is not None and self.root_dir.exists():
            for ext in self.file_extensions:
                self.files.extend(sorted(self.root_dir.rglob(f"*{ext}")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.load_tensor(self.files[index])

    def resolve_path(self, feature_file):
        feature_path = Path(feature_file).expanduser()
        if feature_path.is_absolute():
            return feature_path
        if self.root_dir is None:
            return feature_path
        return self.root_dir / feature_path

    def load_tensor(self, feature_file):
        feature_path = self.resolve_path(feature_file)
        obj = _load_embedding_object(feature_path)
        return _extract_feature_tensor(obj, self.feature_key)

    def infer_hidden_size(self):
        if not self.files:
            raise ValueError(
                "Cannot infer precomputed feature dimension because no embedding files were found. "
                "Set --precomputed_feature_dim explicitly or place files under --precomputed_feature_dir."
            )
        sample = self.load_tensor(self.files[0])
        return int(sample.shape[-1])
