# InstructEnzyme Code-Only Export

Chinese version: [README_zh.md](README_zh.md)

## What This Repository Is

This is a **code-only** export of the current `InstructEnzyme` work.

It packages the parts of the project that were actually needed to run the completed pipeline:

1. convert enzyme-substrate complex structures into `PDB`
2. extract `LigandMPNN` residue embeddings (`h_V_last_layer`)
3. build sequence/embedding manifests and optional `WebDataset` shards
4. train a **Stage-1 adapter-only** structure-to-language alignment model on top of `ProGen2-base`
5. vendor the required `LLaVA` language-model-side wrappers plus local `ProGen2/ProGen3` backbone code
6. evaluate teacher-forcing metrics (`loss`, `perplexity`, `recovery`, `top-5 recovery`)
7. run a 4-GPU batched free-running generation benchmark

This export intentionally **does not include**:

- model weights
- pretrained parameter checkpoints
- `LigandMPNN` parameters
- raw datasets
- generated `PDB` files
- precomputed embeddings
- training checkpoints
- evaluation outputs

## Current Status

The code in this export corresponds to a pipeline that was already executed end-to-end on the original workspace.

### Main completed run

- backbone: `ProGen2-base`
- training mode: `Stage-1 adapter-only`
- structure source: precomputed `LigandMPNN` residue embeddings
- GPUs: `4 x H100`

### Main training result

Best checkpoint during Stage-1 training:

- best periodic validation loss during training: `1.2134275436401367`
- best step: `900`

### Full validation metrics

Computed afterwards on the full validation set (`600` samples):

- `val_loss = 1.228981488920364`
- `val_ppl = 3.417746750030258`
- `val_recovery = 0.6206568231089237`
- `val_top5_recovery = 0.8653590750180661`

### Full held-out test metrics

Computed on the full held-out test set (`584` samples):

- `test_loss = 1.2520202748411564`
- `test_ppl = 3.497401537260594`
- `test_recovery = 0.6154701920678474`
- `test_top5_recovery = 0.8623696682464455`

### 4-GPU batched generation benchmark

Free-running greedy generation on the held-out test set (`584` samples), using `4 x H100` and `batch_size=8` per GPU:

- original greedy benchmark (`batch_size=8`):
  - `mean_sequence_recovery = 0.06586172052768441`
  - `global_residue_recovery = 0.06456971813419805`
  - `exact_match_rate = 0.0`
  - `stop_rate = 0.05650684931506849`
  - `mean_length_ratio = 1.0409335093377838`
- patched greedy benchmark with per-sample stopping (`batch_size=16`):
  - `mean_sequence_recovery = 0.06530344306544608`
  - `global_residue_recovery = 0.0636118732851085`
  - `exact_match_rate = 0.0`
  - `stop_rate = 0.08047945205479452`
  - `mean_length_ratio = 0.9731542024571799`

Interpretation:

- teacher-forcing metrics are already decent
- free-running generation is still weak
- this is expected for a Stage-1 alignment-only model where the backbone remains frozen

### Longer Stage-1 continuation

A longer projector-only continuation run was launched from the previous best checkpoint, still without Stage-2 LoRA.

Current best checkpoint from that run (`step 1000` as of 2026-03-22):

- `val_loss = 1.1851611747426283`
- `val_ppl = 3.2712140160567045`
- `val_recovery = 0.6351145485439738`
- `val_top5_recovery = 0.8728833421222242`

This is already better than the original full-validation result from the 1k-step run.

## Repository Layout

```text
.
├── README.md
├── THIRD_PARTY_NOTICES.md
├── scripts/
│   ├── apply_overrides.sh
│   └── convert_enzyme_cif_to_pdb.py
├── instructenzyme/
│   ├── __init__.py
│   ├── aggregate_generation_eval.py
│   ├── build_index.py
│   ├── build_wds.py
│   ├── dataset.py
│   ├── eval_stage1.py
│   ├── generate_stage1.py
│   ├── modeling.py
│   ├── run_generate_eval_4gpu.sh
│   ├── run_stage1_base_4gpu.sh
│   └── train_stage1.py
└── third_party_overrides/
    ├── LLaVA/
    │   └── llava/model/language_model/...
    └── LigandMPNN/
```

## What Each Part Does

### `scripts/convert_enzyme_cif_to_pdb.py`

Converts complex `mmCIF` files into fixed-column `PDB` files suitable for downstream `LigandMPNN` parsing.

Important detail:

- this version explicitly avoids malformed fixed-column output for long ligand residue names

### `third_party_overrides/LigandMPNN/*`

These files are overrides for the upstream `LigandMPNN` repository.

The key modification is that `ProteinMPNN.encode()` can now save per-sample encoder outputs, allowing direct export of:

```python
{"h_V_last_layer": tensor}
```

This is the structure embedding consumed by the downstream alignment model.

### `third_party_overrides/LLaVA/llava/model/language_model/*`

These files are the language-model-side pieces needed by the protein-conditional stack:

- `llava_llama.py`, `llava_mistral.py`, `llava_mpt.py`: the wrapper classes used by `LLaVA`
- `progen2_hf/*`: local `ProGen2` Hugging Face-style architecture/config/tokenizer files
- `progen3/*`: local `ProGen3` architecture/config/tokenizer files

They are included so the exported repository contains the backbone-side code that was prepared for later `ProGen2/ProGen3` integration, even though the completed Stage-1 run reported here used the standalone `instructenzyme/modeling.py` path on top of local `ProGen2-base` weights.

### `instructenzyme/build_index.py`

Builds a JSONL manifest that pairs:

- a protein design sequence from `PDB`
- a precomputed `LigandMPNN` embedding file

It enforces:

- single protein design chain
- canonical amino acids only
- sequence length equals `h_V_last_layer.shape[0]`

### `instructenzyme/build_wds.py`

Exports the JSONL manifest into `WebDataset` shards.

This is useful for later scaling, although the completed Stage-1 run used JSONL manifest loading directly for simplicity and debuggability.

### `instructenzyme/modeling.py`

Defines the Stage-1 model:

```text
LigandMPNN embeddings [B, L, 128]
-> fixed-query cross-attention projector
-> fixed-length prompt [B, 256, 1536]
-> prepend to ProGen2-base token embeddings
-> frozen ProGen2-base backbone
```

Training policy:

- `LigandMPNN`: frozen
- `ProGen2-base`: frozen
- projector: trainable

### `instructenzyme/train_stage1.py`

Runs Stage-1 adapter-only training.

This script now logs:

- `val_loss`
- `val_ppl`
- `val_recovery`
- `val_top5_recovery`

Recovery is defined over amino-acid positions only; control tokens like `1` and `2` are excluded.

### `instructenzyme/eval_stage1.py`

Loads a saved projector checkpoint and runs full teacher-forcing evaluation on a validation or test manifest.

### `instructenzyme/generate_stage1.py`

Runs true conditioned generation:

- inputs are structure prompt + start token `1`
- the model generates amino acids autoregressively
- token space is constrained to `20` amino acids plus stop token `2`

The current version supports **batched generation**, which materially improves GPU utilization.

### `instructenzyme/aggregate_generation_eval.py`

Merges shard-wise generation outputs and computes summary statistics like:

- mean sequence recovery
- global residue recovery
- exact-match rate
- stop rate
- length ratio

## External Dependencies You Still Need

This repository alone is not enough to run the pipeline. You still need three external assets:

1. upstream `LLaVA` source tree
2. upstream `LigandMPNN` source tree
3. pretrained model weights / datasets

### Upstream repositories

Clone these separately:

```bash
git clone https://github.com/haotian-liu/LLaVA.git
git clone https://github.com/dauparas/LigandMPNN.git
```

Then apply the overrides from this export into a workspace that contains both directories:

```bash
bash scripts/apply_overrides.sh /path/to/workspace
```

That workspace is expected to contain:

```text
/path/to/workspace/
├── LLaVA/
└── LigandMPNN/
```

### ProGen2-base

The exported repository now includes the local `ProGen2/ProGen3` architecture code under `third_party_overrides/LLaVA/llava/model/language_model/`, but you still need to download pretrained weights separately.

Download `ProGen2-base` from Hugging Face:

```bash
python -m pip install -U huggingface_hub hf_xet
hf download hugohrban/progen2-base --local-dir ./progen2-base
```

### LigandMPNN parameters

Inside the upstream `LigandMPNN` directory:

```bash
bash get_model_params.sh ./model_params
```

The runs reported here used:

- `./model_params/ligandmpnn_v_32_005_25.pt`

### Input data

You must provide your own complex dataset, for example:

- `enzyme_data/*.cif`

This export does not include any structural dataset.

## Recommended Environment

A tested environment was created with:

```bash
conda create -n instructenzyme python=3.10 -y
conda activate instructenzyme

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install \
  numpy==1.26.4 \
  transformers==4.40.2 \
  tokenizers==0.19.1 \
  huggingface_hub==0.23.4 \
  accelerate==0.29.3 \
  peft==0.10.0 \
  webdataset==0.2.86 \
  sentencepiece==0.1.99 \
  safetensors \
  biopython \
  prody \
  shortuuid \
  pydantic \
  requests \
  httpx==0.24.0 \
  uvicorn \
  fastapi \
  einops==0.6.1 \
  einops-exts==0.0.4 \
  timm==0.6.13 \
  scikit-learn==1.2.2 \
  pillow \
  tqdm \
  ninja \
  wandb
```

## End-to-End Pipeline

Below is the exact logical order used in the completed run.

### Step 1: Convert `mmCIF` to `PDB`

```bash
python scripts/convert_enzyme_cif_to_pdb.py \
  --input_dir /path/to/enzyme_data \
  --output_dir /path/to/enzyme_pdb \
  --limit 0 \
  --overwrite
```

Purpose:

- make the complexes consumable by `LigandMPNN`
- preserve protein + ligand context
- avoid broken fixed-column `PDB` rows

### Step 2: Extract `LigandMPNN` embeddings

Single-GPU:

```bash
cd /path/to/workspace/LigandMPNN
python extract_ligandmpnn_embeddings.py \
  --pdb_dir /path/to/enzyme_pdb \
  --output_dir /path/to/ligandmpnn_emb \
  --checkpoint ./model_params/ligandmpnn_v_32_005_25.pt
```

4-GPU:

```bash
cd /path/to/workspace/LigandMPNN
BATCH_SIZE=8 bash run_extract_ligandmpnn_embeddings_4gpu.sh \
  /path/to/enzyme_pdb \
  /path/to/ligandmpnn_emb \
  /path/to/workspace/LigandMPNN/model_params/ligandmpnn_v_32_005_25.pt
```

Output per sample:

```python
{"h_V_last_layer": tensor}
```

### Step 3: Build manifest

```bash
python instructenzyme/build_index.py \
  --pdb_dir /path/to/enzyme_pdb \
  --embedding_dir /path/to/ligandmpnn_emb \
  --output_dir /path/to/instructenzyme/data/index
```

In the completed run, this produced:

- total usable samples: `60251`
- train: `59067`
- val: `600`
- test: `584`

### Step 4: Export WDS (optional)

```bash
python instructenzyme/build_wds.py \
  --index_dir /path/to/instructenzyme/data/index \
  --output_dir /path/to/instructenzyme/data/wds \
  --maxcount 1000
```

This is optional for the current Stage-1 recipe.

### Step 5: Stage-1 training on `ProGen2-base`

```bash
MAX_TRAIN_STEPS=1000 \
MAX_VAL_SAMPLES=128 \
BATCH_SIZE=1 \
EVAL_BATCH_SIZE=1 \
EVAL_EVERY=100 \
SAVE_EVERY=100 \
bash instructenzyme/run_stage1_base_4gpu.sh \
  /path/to/progen2-base \
  /path/to/instructenzyme/data/index/train.jsonl \
  /path/to/instructenzyme/data/index/val.jsonl \
  /path/to/instructenzyme/runs/progen2-base-stage1-1k
```

This is the exact type of run that produced the reported Stage-1 checkpoint.

### Step 6: Full validation / full test evaluation

Validation:

```bash
python instructenzyme/eval_stage1.py \
  --model_name_or_path /path/to/progen2-base \
  --projector_ckpt /path/to/instructenzyme/runs/progen2-base-stage1-1k/best/projector.pt \
  --index_path /path/to/instructenzyme/data/index/val.jsonl \
  --output_json /path/to/instructenzyme/evals/progen2-base-stage1-best-full-val.json \
  --batch_size 1 \
  --bf16
```

Test:

```bash
python instructenzyme/eval_stage1.py \
  --model_name_or_path /path/to/progen2-base \
  --projector_ckpt /path/to/instructenzyme/runs/progen2-base-stage1-1k/best/projector.pt \
  --index_path /path/to/instructenzyme/data/index/test.jsonl \
  --output_json /path/to/instructenzyme/evals/progen2-base-stage1-best-full-test.json \
  --batch_size 1 \
  --bf16
```

### Step 7: 4-GPU batched generation benchmark

```bash
BATCH_SIZE=8 bash instructenzyme/run_generate_eval_4gpu.sh \
  /path/to/progen2-base \
  /path/to/instructenzyme/runs/progen2-base-stage1-1k/best/projector.pt \
  /path/to/instructenzyme/data/index/test.jsonl \
  /path/to/instructenzyme/generation_eval/progen2-base-stage1-best-test-greedy-batched
```

This benchmark measures actual free-running generation quality rather than teacher-forcing token prediction.

## Why The Generation Metrics Are Much Lower Than Recovery Under Teacher Forcing

This is expected.

Teacher-forcing evaluation answers:

> If the native sequence is already on the input side, how well can the model predict the next token?

Generation evaluation answers:

> Starting only from the structure prompt, can the model generate a strong full-length sequence on its own?

For the current project state, the answer is:

- Stage-1 alignment works
- Stage-1 alone is not enough for strong full-sequence generation

That is exactly why the natural next step is **Stage-2**.

## Recommended Next Step

If you continue this project, the next technically justified step is:

1. keep the Stage-1 projector initialization
2. add LoRA or a small trainable slice to the top of `ProGen2-base`
3. optimize free-running generation behavior
4. then port the same prompt interface to `ProGen3`

In other words:

- Stage-1 aligns the structure prompt
- Stage-2 teaches the language model to actually use it well during generation

## Notes

- This export was intentionally kept code-only.
- If you push this to GitHub, the repository will remain lightweight and reproducible.
- See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream project references.
