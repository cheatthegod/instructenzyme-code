import os
from .dummy_encoder import DummyVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    precomputed_feature_dir = getattr(vision_tower_cfg, 'precomputed_feature_dir', None)
    use_precomputed_features = getattr(vision_tower_cfg, 'use_precomputed_mm_features', False)
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)

    if use_precomputed_features or vision_tower in {"dummy", "ligandmpnn", "precomputed"} or precomputed_feature_dir is not None:
        return DummyVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
