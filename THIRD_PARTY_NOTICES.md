# Third-Party Notices

This export contains custom code and file-level overrides intended to be applied on top of upstream open-source projects.

Upstream projects referenced by this code:

- `LLaVA`
  - Local path in the original workspace: `LLaVA/`
  - Upstream project: https://github.com/haotian-liu/LLaVA
  - See the upstream repository for license terms.

- `LigandMPNN`
  - Local path in the original workspace: `LigandMPNN/`
  - Upstream project: https://github.com/dauparas/LigandMPNN
  - See the upstream repository for license terms.

- `ProGen2`
  - Local architecture/config files are included under `third_party_overrides/LLaVA/llava/model/language_model/progen2_hf/`.
  - Pretrained weights are not redistributed in this export.
  - Model used in the reported runs: `hugohrban/progen2-base`
  - See the model card / repository for its applicable terms.

- `ProGen3`
  - Local architecture/config files are included under `third_party_overrides/LLaVA/llava/model/language_model/progen3/`.
  - Pretrained weights are not redistributed in this export.
  - See the upstream project / model distribution for its applicable terms.

This repository intentionally excludes:

- model weights
- checkpoints
- datasets
- generated structure embeddings
- training/evaluation artifacts
