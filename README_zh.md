# InstructEnzyme 代码导出版

英文版见 [README.md](README.md)。

## 这是什么

这是当前 `InstructEnzyme` 工作区的代码导出版，只包含复现实验所需的代码、脚本、覆盖补丁和文档，不包含任何权重、数据、embedding、训练 checkpoint 或评估产物。

这套代码对应的目标是：

1. 把 enzyme-substrate complex 的 `mmCIF` 规范转换成 `PDB`
2. 用 `LigandMPNN` 提取每个复合物的 residue-level embedding `h_V_last_layer`
3. 构建 `PDB + embedding + sequence` 的 index / WebDataset
4. 用固定长度 cross-attention projector，把结构 embedding 对齐到 `ProGen2-base` 的 hidden space
5. 在冻结 backbone 的情况下，先做 `Stage-1 adapter-only` 训练
6. 评估 teacher-forcing 指标和 free-running generation 指标

## 当前代码里最重要的部分

### 1. `scripts/convert_enzyme_cif_to_pdb.py`

把复合物 `mmCIF` 转成标准固定列宽 `PDB`。

这里做过一个关键修复：
- 对超过 3 个字符的 ligand residue name 做合法化处理
- 避免写出会让 `LigandMPNN / ProDy` 解析错位的坏格式 `PDB`

### 2. `third_party_overrides/LigandMPNN/`

这是对上游 `LigandMPNN` 的覆盖修改。

核心变化是：
- `ProteinMPNN.encode()` 支持直接保存 encoder 最后一层 embedding
- 输出格式最小化为：

```python
{"h_V_last_layer": tensor}
```

这样后续的多模态对齐流程不需要重复跑 `LigandMPNN` 前向。

### 3. `third_party_overrides/LLaVA/`

这里保存的是把 `LLaVA` 改造成 protein-conditional 架构时需要覆盖的代码，主要包括：

- dummy multimodal encoder
- fixed-length cross-attention projector
- `llava_arch.py` / `train.py` 的多模态接线改动
- `llava/model/language_model/` 下的 `ProGen2/ProGen3` 本地骨架代码

这部分的作用不是让 `LLaVA` 真去看图像，而是复用它的多模态插入框架，把 `complex structure emb` 当成“image embedding”喂给语言模型。

### 4. `instructenzyme/modeling.py`

这是当前实验真正使用的 `Stage-1` 模型定义。

结构如下：

```text
LigandMPNN embedding [B, L, 128]
-> fixed-query cross-attention projector
-> fixed-length prompt [B, 256, hidden_size]
-> prepend 到 ProGen2-base token embeddings 前面
-> 冻结的 ProGen2-base backbone
```

训练时：
- `LigandMPNN` 冻结
- `ProGen2-base` 冻结
- 只训练 projector

### 5. `instructenzyme/train_stage1.py`

负责 `Stage-1` 训练。

当前版本支持：
- `val_loss`
- `val_ppl`
- `val_recovery`
- `val_top5_recovery`
- `--projector_init_ckpt`，可以从已有 projector 继续做更长的 Stage-1，而不是从头开始

### 6. `instructenzyme/generate_stage1.py`

负责真实条件生成。

当前版本已经做过两项修复：
- `seq_len` 排序现在会 fallback 到 `len(record["sequence"])`
- stopping 改成按单样本长度控制，不再使用整个 batch 的最长序列统一截断

这两项修复不会直接把 recovery 变高很多，但会让 generation benchmark 的长度行为更可信。

## 已完成实验结果

### 原始 Stage-1 主跑

主运行使用：
- backbone: `ProGen2-base`
- 训练方式: `Stage-1 adapter-only`
- GPU: `4 x H100`

训练过程中最优 periodic validation：
- best step: `900`
- best periodic `val_loss = 1.2134275436401367`

### full validation

在完整 validation 集（600 条样本）上：
- `val_loss = 1.228981488920364`
- `val_ppl = 3.417746750030258`
- `val_recovery = 0.6206568231089237`
- `val_top5_recovery = 0.8653590750180661`

### full test

在完整 held-out test 集（584 条样本）上：
- `test_loss = 1.2520202748411564`
- `test_ppl = 3.497401537260594`
- `test_recovery = 0.6154701920678474`
- `test_top5_recovery = 0.8623696682464455`

### free-running generation

原始 greedy benchmark：
- `mean_sequence_recovery = 0.06586172052768441`
- `global_residue_recovery = 0.06456971813419805`
- `stop_rate = 0.05650684931506849`
- `mean_length_ratio = 1.0409335093377838`

修正 stopping 之后重新跑的 benchmark：
- `mean_sequence_recovery = 0.06530344306544608`
- `global_residue_recovery = 0.0636118732851085`
- `stop_rate = 0.08047945205479452`
- `mean_length_ratio = 0.9731542024571799`

结论：
- 当前 projector-only 模型的 teacher-forcing 指标已经能看
- 但 free-running generation 仍然很弱
- 修正 generation stopping 后，长度行为已经更合理，不再被 batch 内最长样本系统性拖偏

### 更长的 Stage-1 续训

没有做 `Stage-2 LoRA`，只做了更长的 `Stage-1 projector-only` continuation。

当前正在跑的 continuation run：
- 目录：`/home/ubuntu/cqr_files/protein_design/instructenzyme/runs/progen2-base-stage1-continue-6k-bs8`
- 初始化 checkpoint：原始 1k-step run 的 `best/projector.pt`
- 配置：`4 x H100`, `BATCH_SIZE=8`, `EVAL_BATCH_SIZE=8`, `MAX_TRAIN_STEPS=6000`

截至 2026-03-22 当前已拿到的 best（step 1000）：
- `val_loss = 1.1851611747426283`
- `val_ppl = 3.2712140160567045`
- `val_recovery = 0.6351145485439738`
- `val_top5_recovery = 0.8728833421222242`

相对原始 full-val best：
- `val_loss` 从 `1.22898` 降到 `1.18516`
- `val_recovery` 从 `0.62066` 升到 `0.63511`

这说明在不做 `Stage-2` 的前提下，继续拉长 `Stage-1` 仍然有效。

## 目录结构

```text
.
├── README.md
├── README_zh.md
├── THIRD_PARTY_NOTICES.md
├── scripts/
│   ├── apply_overrides.sh
│   └── convert_enzyme_cif_to_pdb.py
├── instructenzyme/
│   ├── build_index.py
│   ├── build_wds.py
│   ├── dataset.py
│   ├── modeling.py
│   ├── train_stage1.py
│   ├── eval_stage1.py
│   ├── generate_stage1.py
│   ├── aggregate_generation_eval.py
│   ├── run_stage1_base_4gpu.sh
│   └── run_generate_eval_4gpu.sh
└── third_party_overrides/
    ├── LLaVA/
    └── LigandMPNN/
```

## 如何使用这份代码

### 1. 准备上游仓库

```bash
git clone https://github.com/haotian-liu/LLaVA.git
git clone https://github.com/dauparas/LigandMPNN.git
```

然后把这份导出版里的覆盖补丁应用进去：

```bash
bash scripts/apply_overrides.sh /path/to/workspace
```

其中 `/path/to/workspace` 下面需要同时有：

```text
/path/to/workspace/
├── LLaVA/
└── LigandMPNN/
```

### 2. 下载 `ProGen2-base` 权重

```bash
python -m pip install -U huggingface_hub hf_xet
hf download hugohrban/progen2-base --local-dir ./progen2-base
```

### 3. 下载 `LigandMPNN` 参数

在上游 `LigandMPNN` 目录里：

```bash
bash get_model_params.sh ./model_params
```

已用到的模型是：
- `./model_params/ligandmpnn_v_32_005_25.pt`

### 4. 数据准备和训练

完整命令请直接看英文 README，对应步骤已经全部列出。

## 这份仓库不包含什么

为了保证能安全放到 GitHub，这份导出版明确不包含：

- 原始 `enzyme_data`
- 转换后的 `PDB`
- `LigandMPNN` embedding (`.pt`)
- `WebDataset` shards
- `ProGen2/3` 权重
- `LigandMPNN` 参数
- 训练 checkpoint
- 评估 JSON / generation 记录

如果你要完整复现实验，需要你自己准备这些外部资产。
