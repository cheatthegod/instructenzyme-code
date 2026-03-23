# InstructEnzyme 代码导出版

English version: [README.md](README.md)

---

## 项目背景：我们在解决什么问题

**酶（enzyme）是蛋白质**。它的功能——催化哪种化学反应、识别哪种底物——由它的氨基酸序列决定，而序列又决定了三维结构，结构中的活性位点和底物之间的几何关系最终决定了催化活性。

**酶设计（enzyme design）的目标**是：给定一个目标化学反应和对应的底物分子，设计出一条新的氨基酸序列，使它折叠后能有效地催化这个反应。

传统方法（Rosetta 等）依赖手工规则和大量物理计算，速度慢、成功率低。基于深度学习的方法试图让模型从已有的酶-底物复合物结构中"学会"序列和结构之间的关系，再用这个知识生成新序列。

**InstructEnzyme 的核心思路**是：

> 把酶-底物复合物的三维结构当成"条件"，让蛋白质语言模型在这个结构条件下生成氨基酸序列。

---

## 系统由哪几个部分组成

这个项目把三个已有工具拼在一起：

### LigandMPNN：结构读取器

LigandMPNN 是一个专门为蛋白质-配体复合物设计的图神经网络。它输入一个酶-底物复合物的 PDB 结构文件，输出每个氨基酸残基的"结构描述向量"（embedding）。

这个 embedding 是 **ligand-aware** 的：它不只描述蛋白质骨架，还把附近的配体（底物/小分子）原子的信息融合进来了。所以位于活性位点附近的残基，它的 embedding 会和底物的几何环境有关。

每个残基对应一个 128 维的向量。一条有 300 个残基的蛋白质，LigandMPNN 输出的就是 `[300, 128]` 的矩阵。

### ProGen2：蛋白质语言模型

ProGen2 是在数以百万计的蛋白质序列上预训练的语言模型，结构类似 GPT。它"知道"什么样的氨基酸序列是合理的——如果你让它续写一段序列，它会生成在演化上看起来可信的氨基酸组合。

ProGen2-base 的隐藏层维度是 1536（即每个 token 用 1536 维向量表示）。

### Projector（投影器）：两者之间的桥

LigandMPNN 输出的是 128 维的结构向量，ProGen2 期望的输入是 1536 维的 token embedding。而且不同蛋白质的残基数量 L 不一样，但语言模型需要固定格式的输入。

Projector 的职责是：

> 把变长的结构描述 `[L, 128]` 压缩成固定长度的"结构提示词" `[256, 1536]`，格式和 ProGen2 能读懂的 token embedding 一样。

它内部用 **cross-attention（交叉注意力）** 实现：256 个可学习的"查询向量"去"询问"所有 L 个残基，把最有用的信息聚合进来，输出固定的 256 个向量。

这个设计参考了 LLaVA 系统里的多模态对齐思路——LLaVA 把图像特征对齐到语言模型输入空间，这里我们把结构特征对齐到蛋白质语言模型输入空间。

---

## 整体工作流程

```
酶-底物复合物 (PDB 文件)
        │
        ▼  用 LigandMPNN 读取结构
每个残基的结构描述向量  [L, 128]
（把蛋白质和底物的空间关系都编码在里面）
        │
        ▼  用 Projector 压缩
固定长度结构提示词  [256, 1536]
（256 个向量，格式和 ProGen2 的 token embedding 一样）
        │
        ▼  拼接到序列前面
[结构提示词(256个) | 氨基酸序列 token]
        │
        ▼  送入 ProGen2
输出每个位置的下一个氨基酸预测
```

**训练的目标是**：让 Projector 学会把结构信息转化为 ProGen2 能理解的提示词，使得 ProGen2 在看到结构提示词之后，能更准确地预测出正确的氨基酸序列。

---

## 每一步到底输入什么，输出什么

很多人第一次看这套 pipeline 时，最容易混淆的是：

- 文件层面到底在传什么
- 进入训练时张量到底长什么样
- 哪一步是预处理，哪一步真的参与反向传播

下面把它拆开。

### 文件层面的输入输出

| 步骤 | 输入 | 处理 | 输出 |
|------|------|------|------|
| 1. `mmCIF -> PDB` | `enzyme_data/*.cif` | 把复合物结构转成固定列宽 `PDB` | `enzyme_pdb/*.pdb` |
| 2. `PDB -> LigandMPNN embedding` | `enzyme_pdb/*.pdb` | 用 LigandMPNN encoder 提取 ligand-aware 残基 embedding | `ligandmpnn_emb/*.pt` |
| 3. `build_index` | `*.pdb` + `*.pt` | 读取序列、检查长度是否与 embedding 对齐、划分 train/val/test | `train.jsonl` / `val.jsonl` / `test.jsonl` |
| 4. `dataset + collator` | `jsonl` 里的记录 | 读取 `.pt` 和序列，pad 成 batch | batched tensors |
| 5. `Stage-1 model` | batched tensors | `Projector -> ProGen2` | `loss` / `logits` / `recovery` |

### 每个文件里具体装的是什么

**1. 原始结构：`enzyme_data/*.cif`**

每个文件是一条酶-底物复合物结构，通常来自 PDB/mmCIF 数据。

**2. 转换后的结构：`enzyme_pdb/*.pdb`**

每个文件仍然是一条复合物，但现在变成了 `LigandMPNN` 能稳定读取的固定列宽 `PDB`。这里保留的是：

- 蛋白质原子
- 配体 / 底物原子

这里不再包含任何学习到的向量，只是结构文件。

**3. 结构 embedding：`ligandmpnn_emb/*.pt`**

每个 `PDB` 对应一个 `.pt` 文件，最小格式是：

```python
{"h_V_last_layer": tensor}
```

其中：

- `h_V_last_layer.shape = [L, 128]`
- `L` 是这条蛋白质设计链的残基数
- `128` 是 `LigandMPNN` encoder 输出的残基向量维度

这一步结束后，训练时就**不再运行 LigandMPNN**了，训练直接读取这个 `.pt`。

**4. 索引文件：`train.jsonl / val.jsonl / test.jsonl`**

每一行是一条样本记录，例如：

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

这一步的作用不是“学习”，而是把后续训练真正需要的配对信息固定下来：

- 目标序列是谁
- 对应的结构 embedding 在哪

### 一个具体样本在各步骤里的样子

假设有一条样本：

- `id = 6abc_A`
- 氨基酸序列长度 `L = 312`

那么它在不同步骤中的表示是：

1. 原始结构文件：
   - `enzyme_data/6abc_A.cif`
2. 转换后结构文件：
   - `enzyme_pdb/6abc_A.pdb`
3. 结构 embedding：
   - `ligandmpnn_emb/6abc_A.pt`
   - 其中 `h_V_last_layer.shape = [312, 128]`
4. 序列 token：
   - 原始序列长度 = `312`
   - 加上起始 `"1"` 和结束 `"2"` 后，`input_ids.shape = [314]`
5. Projector 输出：
   - `structure_prompt.shape = [256, 1536]`
6. 拼接后的 LM 输入长度：
   - `256 + 314 = 570`
   - 即模型内部真正看到的是长度 `570` 的 embedding 序列

这个例子非常关键，因为它说明：

- **变长的是结构输入的残基数 `L`**
- **固定的是结构 prompt 长度 `256`**
- **ProGen2 最终消费的是 `[256 个结构 token] + [L+2 个序列 token]`**

---

## 这是什么

这是 InstructEnzyme 当前工作的一份**纯代码导出版**。

它打包了实际运行完整 pipeline 所需的部分：

1. 把酶-底物复合物结构转换成 `PDB` 格式
2. 提取 `LigandMPNN` 残基 embedding（`h_V_last_layer`）
3. 构建序列/embedding 索引文件以及可选的 `WebDataset` shards
4. 在 `ProGen2-base` 上训练 **Stage-1 adapter-only** 结构-语言对齐模型
5. 打包所需的 `LLaVA` 语言模型侧封装以及本地 `ProGen2/ProGen3` backbone 代码
6. 评估 teacher-forcing 指标（`loss`、`perplexity`、`recovery`、`top-5 recovery`）
7. 运行 4-GPU 批量自由生成基准测试

此导出版**有意不包含**：

- 模型权重
- 预训练参数 checkpoint
- `LigandMPNN` 参数
- 原始数据集
- 生成的 `PDB` 文件
- 预计算的 embedding
- 训练产生的 checkpoint
- 评估输出

---

## 当前状态

本导出版中的代码对应一个在原始工作空间上已端到端执行完毕的 pipeline。

### 主实验基本信息

- backbone：`ProGen2-base`
- 训练模式：`Stage-1 adapter-only`
- 结构来源：预计算的 `LigandMPNN` 残基 embedding
- GPU：`4 x H100`

### 主训练结果

Stage-1 训练过程中最优 checkpoint：

- 训练中最低周期验证 loss：`1.2134275436401367`
- 最优 step：`900`

### 完整验证集指标

在完整验证集（`600` 条）上事后计算：

- `val_loss = 1.228981488920364`
- `val_ppl = 3.417746750030258`
- `val_recovery = 0.6206568231089237`
- `val_top5_recovery = 0.8653590750180661`

### 完整测试集指标

在完整保留测试集（`584` 条）上计算：

- `test_loss = 1.2520202748411564`
- `test_ppl = 3.497401537260594`
- `test_recovery = 0.6154701920678474`
- `test_top5_recovery = 0.8623696682464455`

### 4-GPU 批量生成基准

在保留测试集（`584` 条）上运行贪心生成，使用 `4 x H100`，每 GPU `batch_size=8`：

- 原始贪心基准（`batch_size=8`）：
  - `mean_sequence_recovery = 0.06586172052768441`
  - `global_residue_recovery = 0.06456971813419805`
  - `exact_match_rate = 0.0`
  - `stop_rate = 0.05650684931506849`
  - `mean_length_ratio = 1.0409335093377838`
- 修正 per-sample stopping 后（`batch_size=16`）：
  - `mean_sequence_recovery = 0.06530344306544608`
  - `global_residue_recovery = 0.0636118732851085`
  - `exact_match_rate = 0.0`
  - `stop_rate = 0.08047945205479452`
  - `mean_length_ratio = 0.9731542024571799`

解读：

- teacher-forcing 指标已经不错
- 自由生成仍然较弱
- 这对于 backbone 完全冻结的 Stage-1 对齐模型来说是预期结果

### Stage-1 续训

从上一轮最优 checkpoint 出发，启动了更长的 projector-only 续训，仍未进行 Stage-2 LoRA。

该续训已经正常结束。run 在 `step 1846` 结束，当前最优 checkpoint 出现在 `step 1500`：

- `val_loss = 1.1598191470202586`
- `val_ppl = 3.189356419342728`
- `val_recovery = 0.6433565750668933`
- `val_top5_recovery = 0.8770580652721627`

这已经优于 1k-step 主跑在完整验证集上的结果。

---

## 目录结构说明

```text
.
├── README.md                         # 英文版
├── README_zh.md                      # 中文版（本文件）
├── THIRD_PARTY_NOTICES.md
├── scripts/
│   ├── apply_overrides.sh            # 应用覆盖补丁的脚本
│   └── convert_enzyme_cif_to_pdb.py  # mmCIF → PDB 转换
├── instructenzyme/
│   ├── build_index.py               # 构建 JSONL 索引
│   ├── build_wds.py                 # 构建 WebDataset shards（可选）
│   ├── dataset.py                   # 数据集和 collator
│   ├── modeling.py                  # Stage-1 模型定义
│   ├── train_stage1.py              # 训练脚本
│   ├── eval_stage1.py               # teacher-forcing 评估
│   ├── generate_stage1.py           # 自由生成
│   ├── aggregate_generation_eval.py # 聚合生成结果
│   ├── run_stage1_base_4gpu.sh      # 训练启动脚本
│   └── run_generate_eval_4gpu.sh    # 生成基准启动脚本
└── third_party_overrides/
    ├── LLaVA/                       # LLaVA 修改（projector、dummy encoder 等）
    └── LigandMPNN/                  # LigandMPNN 修改（embedding 导出脚本）
```

---

## 当前代码里最重要的部分

### `scripts/convert_enzyme_cif_to_pdb.py`

把 `mmCIF` 格式的复合物文件转换成固定列宽 `PDB` 文件，使下游 `LigandMPNN` 解析时不会出错。

重要细节：

- 显式处理了 ligand residue name 超过 3 个字符时的格式问题，避免输出列对齐错误的 PDB 行

### `third_party_overrides/LigandMPNN/*`

这些文件是对上游 `LigandMPNN` 仓库的覆盖补丁。

核心修改是让 `ProteinMPNN.encode()` 能够保存逐样本的 encoder 输出，从而可以直接导出：

```python
{"h_V_last_layer": tensor}
```

这就是下游对齐模型消费的结构 embedding。

### `third_party_overrides/LLaVA/llava/model/language_model/*`

这些是蛋白质条件化 stack 所需的语言模型侧代码：

- `llava_llama.py`、`llava_mistral.py`、`llava_mpt.py`：`LLaVA` 使用的封装类
- `progen2_hf/*`：本地 `ProGen2` Hugging Face 风格的架构/配置/tokenizer 文件
- `progen3/*`：本地 `ProGen3` 架构/配置/tokenizer 文件

这些文件的存在是为了把 `ProGen2/ProGen3` 集成所需的 backbone 侧代码一并导出，尽管本次已完成的 Stage-1 运行使用的是 `instructenzyme/modeling.py` 这条路径直接加载本地 `ProGen2-base` 权重。

### `instructenzyme/build_index.py`

构建 JSONL 索引文件，把以下内容配对：

- 来自 `PDB` 的蛋白质设计序列
- 预计算的 `LigandMPNN` embedding 文件

执行的质量检查：

- 只保留恰好一条蛋白设计链的复合物
- 只接受标准 20 种氨基酸
- 要求序列长度等于 `h_V_last_layer.shape[0]`

### `instructenzyme/build_wds.py`

把 JSONL 索引导出为 `WebDataset` shards。

适合后续扩大规模使用，但已完成的 Stage-1 运行直接使用 JSONL 索引加载，更简单易调试。

### `instructenzyme/modeling.py`

定义 Stage-1 模型：

```text
LigandMPNN embeddings [B, L, 128]
-> 固定查询 cross-attention projector
-> 固定长度提示词 [B, 256, 1536]
-> 拼接到 ProGen2-base token embedding 前面
-> 冻结的 ProGen2-base backbone
```

训练策略：

- `LigandMPNN`：冻结
- `ProGen2-base`：冻结
- projector：可训练

### `instructenzyme/train_stage1.py`

运行 Stage-1 adapter-only 训练。

记录以下指标：

- `val_loss`
- `val_ppl`
- `val_recovery`
- `val_top5_recovery`

recovery 只在氨基酸位置上计算；`1`、`2` 等控制 token 被排除在外。

支持 `--projector_init_ckpt` 从已有 projector checkpoint 恢复训练，而非从头开始。

### `instructenzyme/eval_stage1.py`

加载已保存的 projector checkpoint，在验证集或测试集索引上运行完整的 teacher-forcing 评估。

### `instructenzyme/generate_stage1.py`

运行真正的条件生成：

- 输入为结构提示词 + 起始 token `1`
- 模型自回归地逐步生成氨基酸
- token 空间被限制为 20 种氨基酸加终止 token `2`

当前版本支持**批量生成**，显著提升 GPU 利用率。

相比原始版本修复了两个问题：

- `seq_len` 排序键现在在字段缺失时会回退到 `len(record["sequence"])`
- 停止条件现在按每条样本单独控制，而不是用批次中最长序列作为统一上限

### `instructenzyme/aggregate_generation_eval.py`

合并各 shard 的生成输出，计算汇总统计指标，包括：

- mean sequence recovery（平均序列恢复率）
- global residue recovery（全局残基恢复率）
- exact-match rate（完全匹配率）
- stop rate（终止符触发率）
- length ratio（生成/天然序列长度比）

---

## Stage-1 训练详解

### 哪些部分参与训练，哪些被冻结

```
LigandMPNN    ──── 冻结（预先跑完，结果存成 .pt 文件，训练时不再运行）
Projector     ──── 可训练 ← 这是 Stage-1 唯一更新的部分
ProGen2-base  ──── 冻结（保持预训练状态不变）
```

Stage-1 只训练 Projector，原因是：

1. LigandMPNN 已经是效果很好的结构编码器，不需要重训
2. ProGen2 已经在海量蛋白质序列上训练好了，先保持它的知识不变
3. 先让 Projector 学会"翻译"结构信息，再决定要不要微调 ProGen2（那是 Stage-2）

### 先记住一个最重要的事实

**训练时并没有把 PDB 直接送进模型。**

训练真正读到的是两样东西：

- 一条序列：`sequence`
- 一份预先计算好的结构向量：`h_V_last_layer`

也就是说，训练图里只有：

```text
h_V_last_layer -> Projector -> ProGen2
```

而没有：

```text
PDB -> LigandMPNN -> Projector -> ProGen2
```

`LigandMPNN` 在这套 Stage-1 里是**离线预处理器**，不是在线参与训练的模块。

### 一个 batch 在进入模型前到底长什么样

假设一个 batch 里有两条样本：

- 样本 A：序列长度 `312`
- 样本 B：序列长度 `280`

那么 dataset / collator 会先产生：

**结构侧**

```text
A: h_V_last_layer [312, 128]
B: h_V_last_layer [280, 128]
```

pad 之后：

```text
structure_embs           [2, 312, 128]
structure_attention_mask [2, 312]
```

**序列侧**

原始 token 长度分别是：

- A: `312 + 2 = 314`
- B: `280 + 2 = 282`

pad 之后：

```text
input_ids       [2, 314]
attention_mask  [2, 314]
labels          [2, 314]
```

然后模型内部再把 `256` 个结构 prompt prepend 到文本前面，所以真正送给 ProGen2 的 embedding 序列是：

```text
inputs_embeds   [2, 256 + 314, 1536] = [2, 570, 1536]
full_labels     [2, 570]
```

这里非常容易混淆：

- dataset 输出的 `labels` 还只有文本部分
- model 内部才会在前面补上 `256` 个 `IGNORE_INDEX`
- 所以最终给 backbone 的 `full_labels` 长度才会和 `inputs_embeds` 一样

### 完整前向传播

```
酶-底物复合物 PDB
        │
        ▼ （预计算，训练时不重新运行）
LigandMPNN encoder
        │
        ▼
h_V_last_layer  [B, L, 128]
ligand-aware 残基 embedding，L = 蛋白质残基数
        │
        ▼
FixedQueryCrossAttentionProjector
  ├─ kv_proj:    Linear(128 → H_lm, bias=False)
  ├─ kv_norm:    LayerNorm(H_lm)
  ├─ query:      nn.Parameter [256, H_lm]  （可学习）
  ├─ query_norm: LayerNorm(H_lm)
  ├─ 1 × CrossAttentionBlock
  │    Q = query_norm(query) + 1D-sincos-pos(256)      [B, 256, H_lm]
  │    K = kv_norm(kv_proj(x)) + 1D-sincos-pos(L)      [B, L,   H_lm]
  │    V = kv_norm(kv_proj(x))   （无位置偏置）         [B, L,   H_lm]
  │    attn_out = MultiheadAttention(Q, K, V,
  │                 key_padding_mask=padding_mask)
  │    query = query + attn_out
  │    query = query + FFN(LayerNorm(query))
  └─ post_norm: LayerNorm(H_lm)
        │
        ▼
结构提示词  [B, 256, H_lm]
变长结构被压缩成 256 个固定 token
        │
        ▼  拼接
[ 结构提示词 (256) | token_embeds ]   [B, 256+seq_len, H_lm]
        │
        ▼
ProGen2-base  （冻结，H_lm = 1536）
next-token prediction，交叉熵 loss
```

### 每一步训练时发生了什么

**第一步：加载一条训练样本**

从 JSONL 索引文件读取一条记录：

```json
{
  "id": "6abc_A",
  "sequence": "MKVLINGE...",
  "embedding_path": "/path/to/ligandmpnn_emb/6abc_A.pt"
}
```

从 `.pt` 文件加载预先计算好的结构 embedding `h_V_last_layer`，形状是 `[L, 128]`，L 等于序列长度。

这一阶段的输入输出可以写成：

```text
输入:
  record["sequence"]       -> str, 长度 L
  record["embedding_path"] -> .pt 文件

输出:
  sequence                 -> "MKVL..."
  h_V_last_layer           -> [L, 128]
```

**第二步：把序列转成 token**

ProGen2 的 tokenizer 把序列变成数字序列，格式是：

```
输入序列:  "1" + "MKVLINGE..." + "2"
token ID:  [start_id, M_id, K_id, V_id, ..., E_id, end_id]
```

其中 `"1"` 是序列开始标记，`"2"` 是序列结束标记，每个氨基酸字母是一个 token。对一条长度为 L 的蛋白质，`input_ids` 的长度是 `L + 2`。

这一步的输入输出：

```text
输入:
  "MKVLINGE..."            -> 长度 L

输出:
  input_ids                -> [L+2]
  labels                   -> [L+2]
```

**第三步：Projector 处理结构 embedding**

```
结构 embedding [B, L, 128]
→ kv_proj: Linear(128 → 1536)  把维度从128升到1536
→ 256个可学习查询向量 [B, 256, 1536]
→ Cross-Attention: 每个查询向量向所有L个残基"询问"信息
→ FFN: 前馈神经网络进一步处理
→ 结构提示词 [B, 256, 1536]
```

256 个查询向量经过 cross-attention 之后，每个都聚合了整条序列的结构信息的某个方面。查询向量上加了正弦位置编码（1D sinusoidal），残基 embedding 上也加了位置编码，让模型知道每个残基在序列中的位置。

这一步的输入输出：

```text
输入:
  structure_embs           -> [B, L_max, 128]
  structure_attention_mask -> [B, L_max]

输出:
  structure_prompt         -> [B, 256, 1536]
```

**第四步：拼接，送入 ProGen2**

```
位置:   0 … 255 | 256  | 257 … 256+L | 257+L
内容:   提示词   | "1"  | aa_1 … aa_L |  "2"
标签:   IGNORE  | IGNORE | 监督位置   | 监督位置
```

其中"IGNORE"（`IGNORE_INDEX = -100`）表示这些位置的预测不参与 loss 计算。模型只需要在看到结构提示词和前面的氨基酸之后，正确预测下一个氨基酸。

这一步的输入输出：

```text
输入:
  structure_prompt         -> [B, 256, 1536]
  token_embeds             -> [B, T, 1536]

输出:
  inputs_embeds            -> [B, 256+T, 1536]
  full_labels              -> [B, 256+T]
```

**第五步：计算 loss，更新 Projector**

ProGen2 是 next-token prediction 模型：给定前面所有内容，预测下一个 token。

在第 k 步，模型看到：
```
[结构提示词 | "1" | aa_1 | aa_2 | ... | aa_{k-1}]
```
并预测 `aa_k`。Loss 用交叉熵（cross-entropy）计算，只算氨基酸位置和 end token 位置。

梯度反向传播时，只有 Projector 的参数会被更新，ProGen2 的参数不变。

### 梯度到底流向哪里

这是很多人第一次看这套架构时最容易误解的地方。

反向传播的计算图是：

```text
loss
  ↑
logits
  ↑
ProGen2-base (冻结，不更新参数)
  ↑
structure_prompt
  ↑
Projector (更新参数)
  ↑
h_V_last_layer (常量输入，不更新)
```

也就是说：

- `Projector` 在图里，而且参数 `requires_grad=True`
- `ProGen2-base` 在图里，但参数被冻结，所以虽然梯度会穿过它传播，**它自己的权重不会更新**
- `h_V_last_layer` 只是从磁盘读入的常量张量，不会被优化
- `LigandMPNN` 甚至不在这次训练图里，因为它已经在离线步骤里跑完了

如果用一句话概括 Stage-1：

> Stage-1 不是在教 LigandMPNN 编码结构，也不是在重训 ProGen2；Stage-1 只是在教一个中间翻译器（Projector）把结构向量翻译成 ProGen2 能消费的 prompt。

### 什么是 teacher forcing（教师强迫训练）

训练时用的是 **teacher forcing** 策略：在每一步，模型收到的前缀都是**真实序列**，而不是模型自己上一步的预测。

**打个比方**：这就像老师带着学生做翻译练习——每次告诉你"前面这几个词翻译是这样的，现在预测下一个词"。你是在拿着正确答案练习，而不是自己从头到尾独立翻译。

teacher forcing 的好处：训练快、稳定。坏处：模型从来没练习过"自己的预测出错了怎么办"，所以生成时如果预测出错，之后的错误会越来越多。

### `val_recovery` 到底在衡量什么

```python
# evaluate() 内部：
shift_logits = outputs.logits[:, :-1, :]      # [B, T-1, vocab]
shift_labels = full_labels[:, 1:]             # [B, T-1]
aa_mask      = valid_mask & isin(shift_labels, amino_acid_token_ids)
val_recovery = top1_correct[aa_mask] / total[aa_mask]
```

`val_recovery` 是在 teacher forcing 条件下，模型在氨基酸位置上的 top-1 准确率。起始 token、终止 token 以及结构提示词位置均被排除在外。

这衡量的是：**给定真实序列的前 k 个 token，模型为下一个氨基酸分配最高概率的是否正确？**

没有结构条件的随机模型约得 5%（1/20 种氨基酸）。训练后的 Stage-1 模型达到约 62%，说明结构提示词在 teacher forcing 条件下是有信息量的。

### 优化器与训练调度

| 组件 | 值 |
|------|-----|
| 优化器 | AdamW |
| 学习率 | 2e-4 |
| 权重衰减 | 0.01 |
| 梯度裁剪 | max norm 1.0 |
| LR 调度 | 线性 warmup → 线性衰减 |
| warmup 步数 | 100（Python 默认值，shell 脚本未显式传入） |
| 数值精度 | bfloat16（混合精度） |

只有 projector 参数被传入优化器，backbone 参数完全排除在外。

### 关键架构超参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `structure_hidden_size` | 128 | LigandMPNN 输出维度 |
| `num_queries` | 256 | Projector 输出的固定长度 |
| `num_heads` | 8 | Cross-attention 的注意力头数 |
| `num_layers` | 1 | 叠加的 CrossAttentionBlock 数量 |
| `pos_encoding` | `1d` | 对查询向量和 key 残基 embedding 使用 1D 正弦位置编码 |

### 评估指标的含义

**`val_loss`（验证集 loss）**

交叉熵 loss，数值越低越好。当前实验中，最终结果约 1.22，对应困惑度（perplexity）约 3.4。

困惑度 3.4 的直观含义：对于每个氨基酸位置，模型平均在约 3.4 个候选中不确定（而不是随机猜的情况下的 20 个）。

**`val_recovery`（氨基酸恢复率）**

在 teacher forcing 条件下（每步都给真实前缀），模型预测的下一个氨基酸中，有多少比例是正确的。

- 随机猜（20种氨基酸）：约 5%
- 当前 Stage-1 结果：约 62%
- 这说明结构提示词确实在 teacher forcing 条件下提供了有用信息

**`val_top5_recovery`**

前5个预测中包含正确氨基酸的比例，当前约 87%。

**`mean_sequence_recovery`（生成恢复率）**

自由生成时（不提供真实前缀），生成的序列与天然序列之间的氨基酸相似度。

当前约 6.6%，看起来比随机猜（5%）强一点，但远不够好。这是预期结果——Stage-1 只训练了 Projector，骨架模型没有做任何适应，自由生成时误差累积很厉害。

### 为什么 val_recovery=62% 但生成只有 6.6%

这两个指标衡量的是完全不同的事情：

| 指标 | 条件 | 含义 |
|------|------|------|
| val_recovery (62%) | 每步都有真实前缀 | 模型"理解"结构提示词了吗 |
| generation recovery (6.6%) | 从结构提示词独立生成 | 模型能独立生成好序列吗 |

val_recovery=62% 说明 Projector 学到了有用的映射，结构信息确实进入了模型。

生成 recovery=6.6% 说明 ProGen2 还不擅长在自由生成时利用结构提示词——因为它的预训练没有见过这种格式的提示，而且 Stage-1 的训练量（1000 步）远不足以让它真正学会利用结构条件。

这是 Stage-2（对 ProGen2 做 LoRA 微调）要解决的问题。

---

## Stage-1 vs Stage-2：有什么区别

| | Stage-1（已完成） | Stage-2（未完成） |
|--|--|--|
| 训练对象 | 只训练 Projector | Projector + ProGen2 顶层（用 LoRA） |
| ProGen2 状态 | 完全冻结 | 顶层可以轻微更新 |
| 目标 | 让 Projector 学会翻译结构信息 | 让 ProGen2 学会在生成时使用结构条件 |
| 类比 | 学会把法语翻译成英语描述 | 学会根据法语描述独立写作 |

Stage-1 是基础，Stage-2 在 Stage-1 的 checkpoint 基础上继续训练。

---

## 完整实验流程

下面是实际运行过的完整步骤，按顺序执行。

### 第0步：准备环境

```bash
conda create -n instructenzyme python=3.10 -y
conda activate instructenzyme

pip install --upgrade pip setuptools wheel
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install \
  numpy==1.26.4 \
  transformers==4.40.2 \
  tokenizers==0.19.1 \
  huggingface_hub==0.23.4 \
  accelerate==0.29.3 \
  peft==0.10.0 \
  webdataset==0.2.86 \
  sentencepiece==0.1.99 \
  safetensors biopython prody shortuuid pydantic \
  requests "httpx==0.24.0" uvicorn fastapi \
  "einops==0.6.1" "einops-exts==0.0.4" "timm==0.6.13" \
  "scikit-learn==1.2.2" pillow tqdm ninja wandb
```

### 第0步（续）：准备上游仓库和权重

```bash
# 克隆上游仓库
git clone https://github.com/haotian-liu/LLaVA.git
git clone https://github.com/dauparas/LigandMPNN.git

# 应用本仓库的覆盖补丁
bash scripts/apply_overrides.sh /path/to/workspace

# 下载 ProGen2-base 权重（约 1.5GB）
pip install -U huggingface_hub hf_xet
hf download hugohrban/progen2-base --local-dir ./progen2-base

# 下载 LigandMPNN 模型参数
cd /path/to/workspace/LigandMPNN
bash get_model_params.sh ./model_params
```

### 第1步：把 mmCIF 转成 PDB

酶-底物复合物的结构数据通常是 mmCIF 格式（来自 RCSB PDB 数据库），需要先转成固定列宽 PDB 格式，LigandMPNN 才能正确读取。

```bash
python scripts/convert_enzyme_cif_to_pdb.py \
  --input_dir /path/to/enzyme_data \
  --output_dir /path/to/enzyme_pdb \
  --limit 0 \
  --overwrite
```

注意：这个脚本处理了 ligand residue name 超过3个字符时的格式问题，避免写出让下游工具解析错位的 PDB 文件。

### 第2步：用 LigandMPNN 提取结构 embedding

对每个 PDB 文件，运行 LigandMPNN encoder，把结构信息转成向量并保存。

单 GPU：

```bash
cd /path/to/workspace/LigandMPNN
python extract_ligandmpnn_embeddings.py \
  --pdb_dir /path/to/enzyme_pdb \
  --output_dir /path/to/ligandmpnn_emb \
  --checkpoint ./model_params/ligandmpnn_v_32_005_25.pt
```

4 GPU 并行（推荐，速度快4倍）：

```bash
cd /path/to/workspace/LigandMPNN
BATCH_SIZE=8 bash run_extract_ligandmpnn_embeddings_4gpu.sh \
  /path/to/enzyme_pdb \
  /path/to/ligandmpnn_emb \
  /path/to/workspace/LigandMPNN/model_params/ligandmpnn_v_32_005_25.pt
```

每个 PDB 对应一个 `.pt` 文件，内容是：

```python
{"h_V_last_layer": tensor}  # shape: [L, 128], L = 蛋白质残基数
```

`h_V_last_layer` 就是"ligand-aware 的残基 embedding"——每个氨基酸的 128 维向量描述，融合了周围底物原子的空间信息。

### 第3步：构建训练索引文件

把 PDB 序列和 embedding 文件配对，生成 JSONL 格式的索引，并自动做 train/val/test 切分（98% / 1% / 1%）。

```bash
python instructenzyme/build_index.py \
  --pdb_dir /path/to/enzyme_pdb \
  --embedding_dir /path/to/ligandmpnn_emb \
  --output_dir /path/to/instructenzyme/data/index
```

这一步会做若干质量检查：
- 只保留**恰好一条蛋白链**的复合物（多链酶会被过滤掉）
- 只接受标准 20 种氨基酸
- 检查 `h_V_last_layer.shape[0] == len(sequence)`，长度不对齐的样本直接跳过

在实际运行中，这一步处理约 6 万个样本，产出：
- train: 59,067 条
- val: 600 条
- test: 584 条

每条记录的格式：

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

这里有一个很重要的“契约”：

- `len(sequence)` 必须等于 `h_V_last_layer.shape[0]`

因为当前实现假定：

- 序列里的第 `i` 个残基
- embedding 里的第 `i` 个向量

描述的是同一个位置。如果这一步不对齐，后面的训练监督就是错位的。

### 第4步：Stage-1 训练

用 4 张 H100 跑 Stage-1 adapter-only 训练：

```bash
MAX_TRAIN_STEPS=1000 \
MAX_VAL_SAMPLES=128 \
BATCH_SIZE=2 \
EVAL_BATCH_SIZE=2 \
EVAL_EVERY=100 \
SAVE_EVERY=100 \
bash instructenzyme/run_stage1_base_4gpu.sh \
  /path/to/progen2-base \
  /path/to/instructenzyme/data/index/train.jsonl \
  /path/to/instructenzyme/data/index/val.jsonl \
  /path/to/instructenzyme/runs/progen2-base-stage1-1k
```

关键参数说明：

| 参数 | 值 | 含义 |
|------|----|------|
| MAX_TRAIN_STEPS | 1000 | 总训练步数（初始跑） |
| BATCH_SIZE | 2 | 每张 GPU 每步处理2条样本 |
| 实际每步样本数 | 8 | 4 GPU × 2 |
| 学习率 | 2e-4 | AdamW 优化器 |
| warmup 步数 | 100 | 前100步线性升温 |

这一步的真正输入输出可以再压缩成一句话：

```text
输入:
  train.jsonl 中的 (sequence, embedding_path)

模型内部:
  embedding_path -> h_V_last_layer [L,128]
  sequence       -> input_ids [L+2]
  Projector      -> structure_prompt [256,1536]
  ProGen2        -> logits [256+L+2, vocab]

输出:
  一个标量 loss
  若干验证指标（val_loss / val_ppl / val_recovery / val_top5_recovery）
```

训练过程中每100步在验证集上评估一次，保存 val_loss 最低的 checkpoint 为 `best/projector.pt`。

从已有 checkpoint 继续训练（续训）：

```bash
MAX_TRAIN_STEPS=6000 \
BATCH_SIZE=8 \
PROJECTOR_INIT_CKPT=/path/to/instructenzyme/runs/progen2-base-stage1-1k/best/projector.pt \
bash instructenzyme/run_stage1_base_4gpu.sh \
  /path/to/progen2-base \
  /path/to/instructenzyme/data/index/train.jsonl \
  /path/to/instructenzyme/data/index/val.jsonl \
  /path/to/instructenzyme/runs/progen2-base-stage1-continue
```

### 第5步：在完整 val/test 集上评估

训练结束后，在完整 validation 集和 test 集上计算 teacher-forcing 指标：

```bash
# 验证集
python instructenzyme/eval_stage1.py \
  --model_name_or_path /path/to/progen2-base \
  --projector_ckpt /path/to/instructenzyme/runs/progen2-base-stage1-1k/best/projector.pt \
  --index_path /path/to/instructenzyme/data/index/val.jsonl \
  --output_json /path/to/instructenzyme/evals/progen2-base-stage1-best-full-val.json \
  --batch_size 1 \
  --bf16

# 测试集
python instructenzyme/eval_stage1.py \
  --model_name_or_path /path/to/progen2-base \
  --projector_ckpt /path/to/instructenzyme/runs/progen2-base-stage1-1k/best/projector.pt \
  --index_path /path/to/instructenzyme/data/index/test.jsonl \
  --output_json /path/to/instructenzyme/evals/progen2-base-stage1-best-full-test.json \
  --batch_size 1 \
  --bf16
```

### 第6步：自由生成基准测试

让模型从结构提示词开始，自由生成氨基酸序列，计算生成序列与天然序列的相似度：

```bash
BATCH_SIZE=8 bash instructenzyme/run_generate_eval_4gpu.sh \
  /path/to/progen2-base \
  /path/to/instructenzyme/runs/progen2-base-stage1-1k/best/projector.pt \
  /path/to/instructenzyme/data/index/test.jsonl \
  /path/to/instructenzyme/generation_eval/progen2-base-stage1-best-test-greedy-batched
```

这个脚本会把 584 条测试样本按 4 张 GPU 分片并行生成，然后自动合并结果。

这一步和训练最大的区别是：

- 训练时：每一步都给真实前缀（teacher forcing）
- 生成时：只给 `structure_prompt + "1"`，后面的氨基酸全靠模型自己一步步写出来

所以自由生成更难，它衡量的不是“会不会做下一 token 分类”，而是“能不能在结构条件下独立写完整条蛋白质序列”。

---

## 已完成实验结果

### Stage-1 初始主跑（1000 步）

训练配置：ProGen2-base，4×H100，batch_size=2/GPU，1000步。

训练中最优 checkpoint：step 900，val_loss = 1.2134

在完整 validation 集（600 条）上的最终结果：

| 指标 | 值 | 含义 |
|------|-----|------|
| val_loss | 1.229 | 交叉熵损失 |
| val_ppl | 3.42 | 困惑度（模型对每个位置的不确定程度≈3.4选1） |
| val_recovery | 62.1% | teacher forcing 下氨基酸预测准确率 |
| val_top5_recovery | 86.5% | 前5个预测中包含正确氨基酸的比例 |

在完整 test 集（584 条）上：

| 指标 | 值 |
|------|-----|
| test_loss | 1.252 |
| test_ppl | 3.50 |
| test_recovery | 61.5% |
| test_top5_recovery | 86.2% |

### 自由生成基准（1000 步 checkpoint）

| 指标 | 原始版本 | 修正 stopping 后 |
|------|---------|----------------|
| mean_sequence_recovery | 6.59% | 6.53% |
| global_residue_recovery | 6.46% | 6.36% |
| exact_match_rate | 0% | 0% |
| stop_rate | 5.65% | 8.05% |
| mean_length_ratio | 1.04 | 0.97 |

`stop_rate` 低说明模型很少主动预测结束符 `"2"`，大多数序列是被长度上限截断的。修正 stopping 后，长度行为更合理，但 recovery 本身没有大变化——6.6% 是 Stage-1 的预期水平。

### Stage-1 续训（6000 步，从 1000 步 checkpoint 出发）

截至当前完成结果：

- run 在 `step 1846` 正常结束
- 当前 best 出现在 `step 1500`

| 指标 | 1k-step run | 续训 best (step 1500) |
|------|-------------|---------------|
| val_loss | 1.229 | 1.160 |
| val_ppl | 3.42 | 3.19 |
| val_recovery | 62.1% | 64.3% |
| val_top5_recovery | 86.5% | 87.7% |

说明延长训练仍然有效，但提升在边际递减，真正的突破需要 Stage-2。

---

## 这份仓库不包含什么

为保证 GitHub 仓库轻量，以下内容**不在**本仓库中：

- 原始结构数据（enzyme_data/*.cif）
- 转换后的 PDB 文件
- LigandMPNN embedding 文件（.pt）
- ProGen2/ProGen3 模型权重
- LigandMPNN 模型参数
- 训练产生的 checkpoint
- 评估输出 / 生成记录

完整复现需要自行准备这些外部资产（参见上方"完整实验流程"第0步）。

---

## 下一步：Stage-2

Stage-1 的意义是验证了 Projector 可以学到有意义的结构-序列映射（teacher forcing recovery 62%）。但自由生成仍然很弱，因为 ProGen2 从未在"有结构提示词"的条件下更新过权重。

Stage-2 的计划：

1. 保留 Stage-1 的 Projector 权重
2. 给 ProGen2-base 的顶层 Transformer block 加 LoRA（轻量微调）
3. 以相同的条件序列建模目标训练
4. Projector 和 LoRA 参数共同更新，ProGen2 的底层参数仍然冻结

Stage-2 之后的预期变化：ProGen2 会学会在生成时真正利用结构提示词，free-running generation recovery 应当显著提升。

参见第三方说明：[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
