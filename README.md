# InstructEnzyme

Chinese version: [README_zh.md](README_zh.md)

---

## Background: What Problem Are We Solving

**Enzymes are proteins.** Which chemical reactions they catalyze, and which substrates they recognize, are determined by their amino acid sequence — the sequence folds into a three-dimensional structure, and the geometry of the active site and substrate together determine catalytic activity.

**Enzyme design** asks: given a target reaction and its substrate molecule, can we design a new amino acid sequence that, when folded, would catalyze that reaction effectively?

Traditional approaches (e.g. Rosetta) rely on hand-crafted physical rules and are slow and brittle. Deep learning approaches aim to let a model learn the relationship between sequence and structure from existing enzyme-substrate complexes, then use that knowledge to generate new sequences.

**InstructEnzyme's core idea:**

> Treat the 3D structure of an enzyme-substrate complex as a conditioning signal, and use a protein language model to generate amino acid sequences given that structural context.

---

## The Three Components

The system combines three existing tools:

### LigandMPNN — structure encoder

LigandMPNN is a graph neural network designed for protein-ligand complexes. Given a PDB file of an enzyme-substrate complex, it produces one 128-dimensional embedding vector for each amino acid residue.

These embeddings are **ligand-aware**: they encode not just the protein backbone but also the nearby ligand (substrate/small-molecule) geometry. A residue sitting in the active site next to the substrate will have a very different embedding from a buried residue far from any ligand.

For a protein with L residues, LigandMPNN outputs a `[L, 128]` matrix.

### ProGen2 — protein language model

ProGen2 is a language model pre-trained on millions of protein sequences, structured similarly to GPT. It "knows" what valid protein sequences look like — given a partial sequence, it can predict what amino acids are likely to follow.

ProGen2-base has a hidden dimension of 1536 (each token is represented as a 1536-dimensional vector).

### Projector — the bridge between them

LigandMPNN produces 128-dimensional structural vectors. ProGen2 expects 1536-dimensional token embeddings. And different proteins have different residue counts L — but the language model needs a fixed-size input prefix.

The projector's job:

> Compress the variable-length structural description `[L, 128]` into a fixed-length "structure prompt" `[256, 1536]`, formatted identically to ProGen2's token embeddings.

Internally it uses **cross-attention**: 256 learnable "query" vectors attend over all L residues to extract the most useful structural information, producing a fixed 256-vector output.

This design mirrors the multimodal alignment in LLaVA — LLaVA aligns image features to a language model's input space; here we align structural features to a protein language model's input space.

---

## How the System Works End-to-End

```
Enzyme-substrate complex (PDB file)
        │
        ▼  LigandMPNN encodes the structure
Per-residue structural embeddings  [L, 128]
(each vector encodes local geometry + ligand context)
        │
        ▼  Projector compresses to fixed length
Structure prompt  [256, 1536]
(256 vectors, same format as ProGen2 token embeddings)
        │
        ▼  Prepend to sequence tokens
[structure_prompt (256 tokens) | amino acid sequence tokens]
        │
        ▼  ProGen2 predicts next token at each position
Next-amino-acid predictions, cross-entropy loss
```

The training objective: make the projector learn to translate structural information into a prompt that causes ProGen2 to predict the correct amino acid sequence.

---

## What Goes In And What Comes Out At Each Step

The easiest way to get lost in this project is to mix up:

- file-level inputs and outputs
- tensor-level inputs and outputs
- offline preprocessing versus train-time computation

So here is the pipeline in a more explicit contract form.

### File-level inputs and outputs

| Step | Input | Processing | Output |
|------|-------|------------|--------|
| 1. `mmCIF -> PDB` | `enzyme_data/*.cif` | convert complex structures to fixed-column `PDB` | `enzyme_pdb/*.pdb` |
| 2. `PDB -> LigandMPNN embedding` | `enzyme_pdb/*.pdb` | run LigandMPNN encoder once offline | `ligandmpnn_emb/*.pt` |
| 3. `build_index` | `*.pdb` + `*.pt` | pair sequences with embeddings and split train/val/test | `train.jsonl`, `val.jsonl`, `test.jsonl` |
| 4. `dataset + collator` | JSONL records | load sequence + embedding and pad into a batch | batched tensors |
| 5. `Stage-1 model` | batched tensors | `Projector -> ProGen2` | `loss`, `logits`, recovery metrics |

### What each file actually contains

**1. `enzyme_data/*.cif`**

A raw enzyme-substrate complex structure, typically from an external structural dataset.

**2. `enzyme_pdb/*.pdb`**

The same complex rewritten in a `LigandMPNN`-friendly fixed-column `PDB` format. This is still geometry, not learned features.

**3. `ligandmpnn_emb/*.pt`**

One `.pt` file per complex, in the minimal format:

```python
{"h_V_last_layer": tensor}
```

where:

- `h_V_last_layer.shape = [L, 128]`
- `L` is the residue count of the design chain
- `128` is the LigandMPNN encoder output width

After this step, `LigandMPNN` is no longer part of the Stage-1 training graph. Training reads this tensor directly from disk.

**4. `train.jsonl / val.jsonl / test.jsonl`**

Each line is a sample record such as:

```json
{
  "id": "6abc_A",
  "split": "train",
  "chain_id": "A",
  "sequence": "MKVLINGE...",
  "seq_len": 312,
  "embedding_dim": 128,
  "pdb_path": "/path/to/enzyme_pdb/6abc_A.pdb",
  "embedding_path": "/path/to/ligandmpnn_emb/6abc_A.pt"
}
```

This manifest fixes the pair that matters for training:

- which target sequence should be predicted
- which structure embedding should condition that prediction

### One concrete sample through the pipeline

Suppose we have:

- `id = 6abc_A`
- sequence length `L = 312`

Then the representations are:

1. raw structure file:
   - `enzyme_data/6abc_A.cif`
2. converted structure file:
   - `enzyme_pdb/6abc_A.pdb`
3. structure embedding:
   - `ligandmpnn_emb/6abc_A.pt`
   - with `h_V_last_layer.shape = [312, 128]`
4. tokenized sequence:
   - raw sequence length = `312`
   - after adding start `"1"` and end `"2"`, `input_ids.shape = [314]`
5. projector output:
   - `structure_prompt.shape = [256, 1536]`
6. final LM input length:
   - `256 + 314 = 570`

This is the key idea:

- the structure side is variable-length at the residue level
- the projector compresses it to a fixed prompt length of `256`
- the backbone consumes `[256 structure tokens] + [L+2 sequence tokens]`

---

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

---

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

That continuation run finished normally. The run ended at `step 1846`, and the current best checkpoint was reached at `step 1500`:

- `val_loss = 1.1598191470202586`
- `val_ppl = 3.189356419342728`
- `val_recovery = 0.6433565750668933`
- `val_top5_recovery = 0.8770580652721627`

This is already better than the original full-validation result from the 1k-step run.

---

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

---

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

This script logs:

- `val_loss`
- `val_ppl`
- `val_recovery`
- `val_top5_recovery`

Recovery is defined over amino-acid positions only; control tokens like `1` and `2` are excluded.

Supports `--projector_init_ckpt` to resume from an existing projector checkpoint rather than starting from scratch.

### `instructenzyme/eval_stage1.py`

Loads a saved projector checkpoint and runs full teacher-forcing evaluation on a validation or test manifest.

### `instructenzyme/generate_stage1.py`

Runs true conditioned generation:

- inputs are structure prompt + start token `1`
- the model generates amino acids autoregressively
- token space is constrained to `20` amino acids plus stop token `2`

The current version supports **batched generation**, which materially improves GPU utilization.

Two fixes applied versus the original version:

- `seq_len` sort key now falls back to `len(record["sequence"])` if the field is missing
- stopping is now controlled per-sample rather than using the longest sequence in the batch as a uniform ceiling

### `instructenzyme/aggregate_generation_eval.py`

Merges shard-wise generation outputs and computes summary statistics like:

- mean sequence recovery
- global residue recovery
- exact-match rate
- stop rate
- length ratio

---

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

---

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

---

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
{"h_V_last_layer": tensor}  # shape: [L, 128]
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

Critical contract:

- `len(sequence)` must equal `h_V_last_layer.shape[0]`

The implementation assumes residue `i` in the sequence and residue embedding `i` in the structure tensor refer to the same design position. If that alignment is broken, the supervision signal is misaligned.

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

The step-level IO can be summarized as:

```text
input:
  (sequence, embedding_path) from train.jsonl

inside the model:
  embedding_path -> h_V_last_layer [L,128]
  sequence       -> input_ids [L+2]
  Projector      -> structure_prompt [256,1536]
  ProGen2        -> logits [256+L+2, vocab]

output:
  scalar loss
  validation metrics: val_loss / val_ppl / val_recovery / val_top5_recovery
```

To continue from a previous checkpoint:

```bash
MAX_TRAIN_STEPS=6000 \
BATCH_SIZE=8 \
PROJECTOR_INIT_CKPT=/path/to/previous_run/best/projector.pt \
bash instructenzyme/run_stage1_base_4gpu.sh \
  /path/to/progen2-base \
  /path/to/instructenzyme/data/index/train.jsonl \
  /path/to/instructenzyme/data/index/val.jsonl \
  /path/to/instructenzyme/runs/progen2-base-stage1-continue
```

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

The key difference from training is:

- training: every step gets the ground-truth prefix
- generation: only `structure_prompt + "1"` is given, then the model must continue on its own

So generation is not asking "can the model classify the next token under perfect context?" It is asking "can the model write an entire protein sequence under structural conditioning?"

---

## How Stage-1 Training Works

This section explains exactly what happens during Stage-1 training — from data loading to the loss function — so there is no ambiguity about what the model is learning.

### What is frozen and what is trained

```
LigandMPNN    ──── FROZEN  (run once offline, outputs saved as .pt files)
Projector     ──── TRAINABLE  ← the only thing updated in Stage-1
ProGen2-base  ──── FROZEN  (pre-trained weights unchanged)
```

Why freeze everything except the projector in Stage-1? LigandMPNN is already an effective structure encoder. ProGen2 already has strong protein sequence knowledge from pre-training. The specific thing we need to learn is how to translate structural embeddings into a format ProGen2 can understand — that is exactly what the projector does. Training only the projector first makes it easy to diagnose what is working.

### The single most important thing to remember

**Training does not feed raw PDB files into the model.**

Stage-1 actually trains on just two things:

- a sequence string
- a precomputed structure tensor `h_V_last_layer`

So the train-time graph is:

```text
h_V_last_layer -> Projector -> ProGen2
```

not:

```text
PDB -> LigandMPNN -> Projector -> ProGen2
```

`LigandMPNN` is an **offline preprocessor** in Stage-1, not a train-time module.

### What one batch looks like before it enters the model

Suppose a batch contains two samples:

- sample A: sequence length `312`
- sample B: sequence length `280`

The dataset / collator first constructs the structure side:

```text
A: h_V_last_layer [312, 128]
B: h_V_last_layer [280, 128]
```

After padding:

```text
structure_embs           [2, 312, 128]
structure_attention_mask [2, 312]
```

For the sequence side, the token lengths are:

- A: `312 + 2 = 314`
- B: `280 + 2 = 282`

After padding:

```text
input_ids       [2, 314]
attention_mask  [2, 314]
labels          [2, 314]
```

Inside the model, the `256` structure prompt tokens are prepended, so the actual backbone input becomes:

```text
inputs_embeds   [2, 256 + 314, 1536] = [2, 570, 1536]
full_labels     [2, 570]
```

This distinction matters:

- the dataset emits text-only labels
- the model prepends `256` `IGNORE_INDEX` positions internally
- only then do labels match the backbone input length

### Full forward pass

```
enzyme-substrate complex PDB
        │
        ▼ (precomputed, not rerun during training)
LigandMPNN encoder
        │
        ▼
h_V_last_layer  [B, L, 128]
ligand-aware residue embeddings, L = number of protein residues
        │
        ▼
FixedQueryCrossAttentionProjector
  ├─ kv_proj:    Linear(128 → H_lm, bias=False)
  ├─ kv_norm:    LayerNorm(H_lm)
  ├─ query:      nn.Parameter [256, H_lm]  (learnable)
  ├─ query_norm: LayerNorm(H_lm)
  ├─ 1 × CrossAttentionBlock
  │    Q = query_norm(query) + 1D-sincos-pos(256)      [B, 256, H_lm]
  │    K = kv_norm(kv_proj(x)) + 1D-sincos-pos(L)      [B, L,   H_lm]
  │    V = kv_norm(kv_proj(x))   (no positional bias)  [B, L,   H_lm]
  │    attn_out = MultiheadAttention(Q, K, V,
  │                 key_padding_mask=padding_mask)
  │    query = query + attn_out
  │    query = query + FFN(LayerNorm(query))
  └─ post_norm: LayerNorm(H_lm)
        │
        ▼
structure prompt  [B, 256, H_lm]
variable-length structure compressed into 256 fixed tokens
        │
        ▼  prepend
[ structure_prompt (256) | token_embeds ]   [B, 256+seq_len, H_lm]
        │
        ▼
ProGen2-base  (frozen, H_lm = 1536)
next-token prediction with cross-entropy loss
```

### Tokenization

ProGen2 uses a character-level tokenizer. Each standard amino acid is a single token. The format used here is:

```
input:  "1" + sequence + "2"
```

where `"1"` is the sequence-start token and `"2"` is the sequence-end / stop token. No HuggingFace `add_special_tokens` wrapping is applied — the `1` and `2` characters in ProGen2's vocabulary serve this role directly.

For a protein of length L, `input_ids` has length `1 + L + 1 = L + 2`.

At this step, the input/output contract is:

```text
input:
  sequence string length L

output:
  input_ids [L+2]
  labels    [L+2]
```

### What positions are supervised in the loss

After prepending the 256 structure-prompt tokens, the full sequence fed to the backbone is:

```
position:   0 … 255 | 256  | 257 … 256+L | 257+L
content:    prompt  | "1"  | aa_1 … aa_L |  "2"
label:      IGNORE  | IGNORE | supervised | supervised
```

The loss is computed only over the amino acid positions and the end token — 256 prompt positions and the start token are masked with `IGNORE_INDEX = -100`.

In next-token-prediction terms: at each supervised step k, the model sees
```
[structure_prompt | "1" | aa_1 | … | aa_{k-1}]
```
and must predict `aa_k`. The final supervised step predicts the end token `"2"`.

At this step, the model-side tensors are:

```text
input:
  structure_prompt [B, 256, 1536]
  token_embeds     [B, T,   1536]

output:
  inputs_embeds    [B, 256+T, 1536]
  full_labels      [B, 256+T]
```

### Teacher forcing

Training uses **teacher forcing**: at every step the model receives the ground-truth prefix, regardless of what it would have predicted. This is standard for autoregressive language model training — it keeps gradients stable and avoids compounding errors during training.

**Analogy**: it is like practicing a translation with a language teacher who always shows you the correct previous words. You learn the patterns quickly, but you never practice recovering from your own mistakes.

The practical consequence: training loss and teacher-forcing recovery metrics are measured under an optimistic condition (ground-truth context always available), which is why these numbers look better than free-running generation.

### Where gradients actually flow

This is another common source of confusion.

The backward graph is:

```text
loss
  ↑
logits
  ↑
ProGen2-base (frozen, parameters not updated)
  ↑
structure_prompt
  ↑
Projector (trainable, parameters updated)
  ↑
h_V_last_layer (loaded tensor, not updated)
```

So in Stage-1:

- the projector is inside the graph and trainable
- ProGen2 is inside the graph but frozen
- `h_V_last_layer` is just an input tensor from disk
- LigandMPNN is not in the train-time graph at all

In one sentence:

> Stage-1 does not teach LigandMPNN how to encode structure, and it does not retrain ProGen2. It only teaches a translator in the middle: the projector.

### What `val_recovery` actually measures

```python
# inside evaluate():
shift_logits = outputs.logits[:, :-1, :]      # [B, T-1, vocab]
shift_labels = full_labels[:, 1:]              # [B, T-1]
aa_mask      = valid_mask & isin(shift_labels, amino_acid_token_ids)
val_recovery = top1_correct[aa_mask] / total[aa_mask]
```

`val_recovery` is teacher-forcing top-1 accuracy over amino acid positions only. The start token, end token, and all structure-prompt positions are excluded from this metric.

This measures: **given the true sequence up to position k, how often does the model assign the highest probability to the correct next amino acid?**

A random model with no structural conditioning would score ~5% (1/20 amino acids). The trained Stage-1 model reaches ~62%, showing the structure prompt is informative under teacher forcing.

### Loss aggregation across GPUs (DDP)

Each GPU computes the mean cross-entropy loss over its local batch (backbone returns this directly). For logging, the per-GPU mean losses are reduced across ranks and averaged.

For validation, the loss is accumulated as `sum(loss × valid_token_count)` per GPU, then summed globally and divided by the total token count — this gives the correct token-weighted mean across all ranks.

### Optimizer and schedule

| Component | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| Weight decay | 0.01 |
| Gradient clipping | max norm 1.0 |
| LR schedule | linear warmup → linear decay |
| Warmup steps | 100 (Python default, not set in shell script) |
| dtype | bfloat16 (mixed precision) |

Only projector parameters are passed to the optimizer. Backbone parameters are excluded entirely.

### Key architecture hyperparameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `structure_hidden_size` | 128 | LigandMPNN output dimension |
| `num_queries` | 256 | Fixed output length of projector |
| `num_heads` | 8 | Attention heads in cross-attention |
| `num_layers` | 1 | Stacked cross-attention blocks |
| `pos_encoding` | `1d` | Sinusoidal 1D positional encoding for both query and key positions |
| `use_query_pos` | True | Add positional encoding to learnable queries |
| `use_input_pos` | True | Add positional encoding to input residue embeddings |
| `use_post_proj` | False | No extra linear after resampler (Identity) |

### What the projector is learning

The 256 learnable query vectors act as fixed "slots" that attend over the variable-length residue embeddings. After training, each query should capture some aspect of the structural context that is predictive of the sequence. The cross-attention mechanism allows every query to attend to all residues, with the padding mask ensuring padded positions (from batching) are ignored.

The positional encodings on queries and input residues allow the model to distinguish position along the query and along the sequence, though in Stage-1 there is no explicit residue-order correspondence enforced between queries and amino acid positions — the queries are free to attend globally.

### Generation: how it differs from training

During generation, teacher forcing is off. The model runs fully autoregressively:

```
input:  [structure_prompt (256) | "1"]
step 0: predict aa_1
step 1: predict aa_2 given aa_1
...
step k: predict aa_{k+1} given aa_1 … aa_k
stop:   when "2" is predicted OR per-sample length limit is reached
```

At each step, logits are **restricted to the 20 amino acid tokens + stop token** before sampling or greedy decoding. All other vocabulary entries are masked to -1e9.

KV caching is enabled during generation for efficiency. For batched generation, finished sequences continue to receive the stop token as input to maintain batch alignment.

---

## Stage-1 vs Stage-2

| | Stage-1 (completed) | Stage-2 (planned) |
|--|--|--|
| What is trained | Projector only | Projector + top layers of ProGen2 (LoRA) |
| ProGen2 state | Fully frozen | Top layers lightly updated |
| Goal | Teach the projector to translate structural embeddings | Teach ProGen2 to use structural prompts during generation |
| Analogy | Learn to describe a French text in English | Learn to write well given an English description |

Stage-2 starts from the Stage-1 projector checkpoint and adds LoRA adapters to the top Transformer blocks of ProGen2-base. The expected effect: free-running generation recovery improves significantly because ProGen2 now receives gradient signal from the structural conditioning task.

---

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

---

## Recommended Next Step

If you continue this project, the next technically justified step is:

1. keep the Stage-1 projector initialization
2. add LoRA or a small trainable slice to the top of `ProGen2-base`
3. optimize free-running generation behavior
4. then port the same prompt interface to `ProGen3`

In other words:

- Stage-1 aligns the structure prompt
- Stage-2 teaches the language model to actually use it well during generation

---

## Notes

- This export was intentionally kept code-only.
- If you push this to GitHub, the repository will remain lightweight and reproducible.
- See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream project references.
