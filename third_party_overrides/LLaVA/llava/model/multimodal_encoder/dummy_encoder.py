from types import SimpleNamespace

import torch
import torch.nn as nn

from .embedding_dataset import EmbeddingFeatureDataset


class PrecomputedEmbeddingProcessor:
    def __init__(self, root_dir=None, feature_key="h_V_last_layer", feature_dim=None, file_extensions=None):
        self.root_dir = root_dir
        self.feature_key = feature_key
        self.file_extensions = tuple(file_extensions or (".pt", ".pth", ".bin", ".pkl"))
        self.is_precomputed_embedding = True
        self.crop_size = {"height": 1, "width": 1}

        self.dataset = EmbeddingFeatureDataset(
            root_dir=root_dir,
            feature_key=feature_key,
            file_extensions=self.file_extensions,
        )
        self.hidden_size = int(feature_dim) if feature_dim is not None else self.dataset.infer_hidden_size()

    def load(self, feature_file):
        return self.dataset.load_tensor(feature_file)


class DummyVisionTower(nn.Module):
    """A minimal tower that forwards precomputed embeddings into the projector."""

    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        feature_dir = getattr(args, "precomputed_feature_dir", None)
        feature_key = getattr(args, "precomputed_feature_key", "h_V_last_layer")
        feature_dim = getattr(args, "precomputed_feature_dim", None)
        file_extensions = getattr(args, "precomputed_feature_extensions", None)

        if feature_dir is None and vision_tower not in {None, "dummy", "ligandmpnn", "precomputed"}:
            feature_dir = vision_tower

        self.vision_tower_name = vision_tower
        self.image_processor = PrecomputedEmbeddingProcessor(
            root_dir=feature_dir,
            feature_key=feature_key,
            feature_dim=feature_dim,
            file_extensions=file_extensions,
        )
        self.cfg_only = SimpleNamespace(hidden_size=self.image_processor.hidden_size)
        self.is_loaded = True
        self.register_buffer("_dummy", torch.zeros(1), persistent=False)

    def load_model(self, device_map=None):
        self.is_loaded = True

    @torch.no_grad()
    def forward(self, images):
        if isinstance(images, list):
            return [image.to(device=self.device, dtype=self.dtype) for image in images]

        if images.dim() == 2:
            images = images.unsqueeze(0)

        return images.to(device=self.device, dtype=self.dtype)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self._dummy.dtype

    @property
    def device(self):
        return self._dummy.device

    @property
    def config(self):
        return self.cfg_only

    @property
    def hidden_size(self):
        return self.image_processor.hidden_size

    @property
    def num_patches_per_side(self):
        return 1

    @property
    def num_patches(self):
        return 1
