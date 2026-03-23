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

## Stage-1 训练原理

本节详细说明 Stage-1 训练期间到底发生了什么——从数据读取到 loss 计算——消除所有歧义。

### 完整前向传播

```
enzyme-substrate complex PDB
        │
        ▼（预先计算，训练时不重新运行）
LigandMPNN encoder
        │
        ▼
h_V_last_layer  [B, L, 128]
配体感知的残基 embedding，L = 蛋白质残基数
        │
        ▼
FixedQueryCrossAttentionProjector
  ├─ kv_proj:    Linear(128 → H_lm, bias=False)
  ├─ kv_norm:    LayerNorm(H_lm)
  ├─ query:      nn.Parameter [256, H_lm]（可学习）
  ├─ query_norm: LayerNorm(H_lm)
  ├─ 1 层 CrossAttentionBlock
  │    Q = query_norm(query) + 1D-sincos-pos(256)      [B, 256, H_lm]
  │    K = kv_norm(kv_proj(x)) + 1D-sincos-pos(L)      [B, L,   H_lm]
  │    V = kv_norm(kv_proj(x))   （V 不加位置编码）   [B, L,   H_lm]
  │    attn_out = MultiheadAttention(Q, K, V,
  │                 key_padding_mask=padding_mask)
  │    query = query + attn_out
  │    query = query + FFN(LayerNorm(query))
  └─ post_norm: LayerNorm(H_lm)
        │
        ▼
结构 prompt  [B, 256, H_lm]
变长的结构信息被压缩成 256 个固定 token
        │
        ▼  拼接到前面
[ structure_prompt (256) | token_embeds ]   [B, 256+seq_len, H_lm]
        │
        ▼
ProGen2-base（冻结，H_lm = 1536）
next-token prediction，cross-entropy loss
```

**只有 projector 参与训练**。`LigandMPNN` 和 `ProGen2-base` 在整个 Stage-1 期间完全冻结。

### tokenization 格式

ProGen2 使用字符级 tokenizer，每个标准氨基酸是一个 token。这里使用的格式是：

```
"1" + sequence + "2"
```

其中 `"1"` 是序列开始 token，`"2"` 是序列结束/终止 token。不使用 HuggingFace 的 `add_special_tokens` 包装，ProGen2 词表中的字符 `1` 和 `2` 直接承担这个角色。

对于长度为 L 的蛋白质，`input_ids` 长度为 `1 + L + 1 = L + 2`。

### 哪些位置参与 loss 计算

把 256 个结构 prompt token 拼到前面后，送入 backbone 的完整序列是：

```
位置：   0 … 255 | 256  | 257 … 256+L | 257+L
内容：   prompt  | "1"  | aa_1 … aa_L |  "2"
label：  IGNORE  | IGNORE |  参与 loss  | 参与 loss
```

**loss 只计算在氨基酸位置和终止 token 上**——256 个 prompt 位置和 start token 全部用 `IGNORE_INDEX = -100` 屏蔽。

用 next-token-prediction 的语言来说：在第 k 步，模型看到
```
[结构 prompt | "1" | aa_1 | … | aa_{k-1}]
```
并预测 `aa_k`。最后一个被监督的步骤是预测 end token `"2"`。

### Teacher forcing

训练使用 **teacher forcing**：每一步都把 ground-truth 前缀喂给模型，无论模型本来会预测什么。这是自回归语言模型训练的标准做法——可以保持梯度稳定，避免训练期间的误差累积。

实际后果是：训练 loss 和 teacher-forcing recovery 指标是在乐观条件下（上下文始终是真实序列）测量的，因此这些数字看起来比 free-running generation 好很多。

### `val_recovery` 的真实含义

```python
# 在 evaluate() 内部：
shift_logits = outputs.logits[:, :-1, :]       # [B, T-1, vocab]
shift_labels = full_labels[:, 1:]               # [B, T-1]
aa_mask      = valid_mask & isin(shift_labels, amino_acid_token_ids)
val_recovery = top1_correct[aa_mask] / total[aa_mask]
```

`val_recovery` 是 teacher-forcing top-1 准确率，**只统计氨基酸位置**。start token、end token 和所有结构 prompt 位置都不计入。

这个指标衡量的是：**在给定真实序列前缀的条件下，模型对下一个氨基酸预测正确的比例**。

没有结构条件的随机模型期望得分约 5%（20 种氨基酸中猜对 1 个）。训练后的 Stage-1 模型达到约 62%，说明结构 prompt 在 teacher forcing 条件下确实提供了有效信息。

### DDP 多卡下的 loss 聚合

每张 GPU 计算本地 batch 的平均 cross-entropy loss（backbone 直接返回）。日志记录时对各 rank 的局部均值做 AllReduce 再取平均。

验证时的聚合方式更精确：每 GPU 累积 `sum(loss × valid_token_count)`，然后全局求和再除以总 token 数，得到正确的 token 加权均值。

### 优化器与学习率调度

| 项目 | 值 |
|------|----|
| 优化器 | AdamW |
| 学习率 | 2e-4 |
| 权重衰减 | 0.01 |
| 梯度裁剪 | max norm 1.0 |
| 学习率调度 | 线性 warmup → 线性衰减 |
| warmup 步数 | 100（Python 默认值，shell 脚本未显式传入） |
| 精度 | bfloat16 混合精度 |

**只有 projector 的参数传入优化器**，backbone 参数完全不参与。

### 关键架构超参

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `structure_hidden_size` | 128 | LigandMPNN 输出维度 |
| `num_queries` | 256 | projector 固定输出 token 数 |
| `num_heads` | 8 | cross-attention 的注意力头数 |
| `num_layers` | 1 | 堆叠的 cross-attention block 层数 |
| `pos_encoding` | `1d` | query 和 key 均使用 1D sinusoidal 位置编码 |
| `use_query_pos` | True | query 加位置编码 |
| `use_input_pos` | True | 输入残基 embedding 加位置编码 |
| `use_post_proj` | False | resampler 后不加额外线性层（Identity） |

### projector 在学什么

256 个可学习 query 向量充当固定的"槽位"，对变长残基 embedding 做 cross-attention。训练结束后，每个 query 应当捕捉到对序列预测有用的结构上下文的某个方面。cross-attention 允许每个 query 关注所有残基，padding mask 确保 batch 对齐时的填充位置不被关注。

query 和输入残基上都有位置编码，让模型能区分 query 编号和残基序号，但 Stage-1 中并没有强制 query 和氨基酸位置之间有一对一对应关系——query 可以自由地全局 attend。

### 生成时与训练的区别

生成时 teacher forcing 关闭，模型完全自回归运行：

```
输入：  [结构 prompt (256) | "1"]
第 0 步：预测 aa_1
第 1 步：在已有 aa_1 的条件下预测 aa_2
…
第 k 步：在已有 aa_1 … aa_k 的条件下预测 aa_{k+1}
终止：  当预测到 "2" 或达到单样本长度上限时停止
```

每步 logit 在采样或 greedy 解码前会被**限制在 20 种氨基酸 token + 终止 token 上**，其余词表位置全部设为 -1e9。

生成时开启 KV cache 提高效率。批量生成时，已完成的序列继续接收终止 token 作为输入，以维持 batch 对齐。

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
