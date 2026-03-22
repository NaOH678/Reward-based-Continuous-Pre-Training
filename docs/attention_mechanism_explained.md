# Future Encoder 注意力机制详解

> 本文档详细讲解 Reward-based Continuous Pre-Training 项目中的双路注意力机制设计

---

## 目录

1. [整体架构](#1-整体架构双路-attention)
2. [Causal Attention (正常语言模型)](#2-causal-attention-正常语言模型)
3. [Anti-Causal Attention (Future Encoder)](#3-anti-causal-attention-future-encoder)
4. [Document Boundaries (文档边界)](#4-document-boundaries-文档边界)
5. [Future Encoder 训练流程](#5-future-encoder-的训练流程)
6. [代码实现关键函数](#6-代码实现关键函数)
7. [设计理念](#7-为什么这样设计)

---

## 1. 整体架构：双路 Attention

你的模型使用了**两套并行的 attention 机制**：

```
Input: [token_0, token_1, token_2, ..., token_T]
           ↓
    ┌──────┴──────┐
    ↓             ↓
Causal        Anti-Causal
Attention     Attention (Future)
(正常LM)       (Future Encoder)
    ↓             ↓
  logits      future_summary
    ↓             ↓
  LM Loss    +  MI Loss (InfoNCE)
```

**关键点**：
- **Causal Attention**: 看历史，预测下一个 token (标准语言模型)
- **Anti-Causal Attention**: 看未来，学习预测未来语义 (Future Encoder)
- 两者**并行独立**运行，不相互影响

---

## 2. Causal Attention (正常语言模型)

### 2.1 基本原理

**作用**: 预测下一个 token

**核心约束**: Position `i` 只能看到 `0, 1, ..., i`（只看历史和自己）

### 2.2 无文档边界 (默认行为)

**Attention Mask**:
```
Position:  0   1   2   3   4
Token:    [A] [B] [C] [D] [E]

Causal Attention Matrix (position i 可以看到的 positions):
       看→  0   1   2   3   4
From ↓
  0  [A]   ✓   ✗   ✗   ✗   ✗    (只看自己)
  1  [B]   ✓   ✓   ✗   ✗   ✗    (看 A, B)
  2  [C]   ✓   ✓   ✓   ✗   ✗    (看 A, B, C)
  3  [D]   ✓   ✓   ✓   ✓   ✗    (看 A, B, C, D)
  4  [E]   ✓   ✓   ✓   ✓   ✓    (看全部历史)
```

**跨文档情况**:
```
Sequence: [Doc1_A, Doc1_B, <EOS>, Doc2_A, Doc2_B]
Position:     0       1       2       3       4

Causal Attention (无文档边界):
       看→  0   1   2   3   4
From ↓
  0       ✓   ✗   ✗   ✗   ✗
  1       ✓   ✓   ✗   ✗   ✗
  2 <EOS> ✓   ✓   ✓   ✗   ✗    ← 可以看到 Doc1 的所有内容
  3       ✓   ✓   ✓   ✓   ✗    ← ⚠️ 可以看到 Doc1 + EOS + Doc2_A
  4       ✓   ✓   ✓   ✓   ✓    ← ⚠️ 可以看到 Doc1 + EOS + Doc2 全部
```

**问题**:
- Position 3 (Doc2_A) 能看到 Doc1 的内容
- Position 4 (Doc2_B) 能看到 Doc1 + Doc2 的混合历史
- **理论上有 contamination**，但实际上：
  - ✅ EOS token 是强边界信号
  - ✅ 模型会学到 EOS 后 "context reset"
  - ✅ 绝大多数成功的 LLM (GPT, LLaMA, OLMo) 都这样做

### 2.3 有文档边界 (varlen 模式)

**Attention Mask** (使用 cu_seqlens):
```
Sequence: [Doc1_A, Doc1_B, <EOS>, Doc2_A, Doc2_B]
Position:     0       1       2       3       4
cu_seqlens: [0, 3, 5]  (Doc1: [0,3), Doc2: [3,5))

Causal Attention (严格文档边界):
       看→  0   1   2   3   4
From ↓
  0       ✓   ✗   ✗   ✗   ✗    ← 只看自己
  1       ✓   ✓   ✗   ✗   ✗    ← 只看 Doc1 内
  2 <EOS> ✓   ✓   ✓   ✗   ✗    ← 只看 Doc1 内
  3       ✗   ✗   ✗   ✓   ✗    ← ✅ 不能看 Doc1，只看自己
  4       ✗   ✗   ✗   ✓   ✓    ← ✅ 只能看 Doc2 内
```

**特点**:
- ✅ 完全隔离文档
- ✅ 没有跨文档 contamination
- ❌ 需要 varlen=True，batch_size 必须为 1
- ❌ 训练速度慢 (MFU 22%)

### 2.4 Causal Attention 实现

**代码位置**: 由 FLA (Flash Linear Attention) 库自动处理

```python
# 标准调用
output = model(
    input_ids=input_ids,
    attention_mask=attention_mask,  # 只用于 padding
    cu_seqlens=cu_seqlens,         # varlen 模式的文档边界 (可选)
)
```

---

## 3. Anti-Causal Attention (Future Encoder)

### 3.1 基本原理

**作用**: 看"未来"的 tokens，学习预测未来信息

**核心约束**: Position `i` 只能看到 `i+1, i+2, ..., T`（只看未来，不看自己）

### 3.2 无 Window 约束 (window_k=0 或 None)

**Attention Mask**:
```
Position:  0   1   2   3   4
Token:    [A] [B] [C] [D] [E]

Anti-Causal Attention Matrix (position i 可以看到的 positions):
       看→  0   1   2   3   4
From ↓
  0  [A]   ✗   ✓   ✓   ✓   ✓    (看 B, C, D, E - 所有未来)
  1  [B]   ✗   ✗   ✓   ✓   ✓    (看 C, D, E)
  2  [C]   ✗   ✗   ✗   ✓   ✓    (看 D, E)
  3  [D]   ✗   ✗   ✗   ✗   ✓    (看 E)
  4  [E]   ✗   ✗   ✗   ✗   ✗    (没有未来，不参与训练)
```

**future_valid 标记**:
```
Position:  0   1   2   3   4
Valid:     ✓   ✓   ✓   ✓   ✗    (E 没有未来，不计算 loss)
```

### 3.3 有 Window 约束 (window_k=64)

**Attention Mask** (假设 window_k=2):
```
Position:  0   1   2   3   4   5   6
Token:    [A] [B] [C] [D] [E] [F] [G]

Anti-Causal Attention Matrix (只看未来 2 个 tokens):
       看→  0   1   2   3   4   5   6
From ↓
  0  [A]   ✗   ✓   ✓   ✗   ✗   ✗   ✗    (只看 B, C)
  1  [B]   ✗   ✗   ✓   ✓   ✗   ✗   ✗    (只看 C, D)
  2  [C]   ✗   ✗   ✗   ✓   ✓   ✗   ✗    (只看 D, E)
  3  [D]   ✗   ✗   ✗   ✗   ✓   ✓   ✗    (只看 E, F)
  4  [E]   ✗   ✗   ✗   ✗   ✗   ✓   ✓    (只看 F, G)
  5  [F]   ✗   ✗   ✗   ✗   ✗   ✗   ✓    (只看 G)
  6  [G]   ✗   ✗   ✗   ✗   ✗   ✗   ✗    (没有未来)
```

**你的配置**: `future_k=64`，所以每个 position 只看未来 64 个 tokens

**为什么限制 window?**
- ✅ 避免信息泄露太多（预测太远的未来太简单）
- ✅ 64 tokens ≈ 几个句子，是合理的"未来窗口"
- ✅ 控制任务难度，平衡训练稳定性和学习效果

---

## 4. Document Boundaries (文档边界)

### 4.1 快速模式：无文档边界 (respect_doc_boundaries=False)

**配置**: 默认模式，当前使用

**Causal Attention** (标准 LM):
```
Sequence: [Doc1_A, Doc1_B, <EOS>, Doc2_A, Doc2_B]
Position:     0       1       2       3       4

Causal Mask:
       看→  0   1   2   3   4
From ↓
  0       ✓   ✗   ✗   ✗   ✗
  1       ✓   ✓   ✗   ✗   ✗
  2 <EOS> ✓   ✓   ✓   ✗   ✗
  3       ✓   ✓   ✓   ✓   ✗    ← 可以看到 Doc1
  4       ✓   ✓   ✓   ✓   ✓    ← 可以看到 Doc1 + Doc2
```

**Anti-Causal Attention** (Future Encoder, window_k=2):
```
Future Mask:
       看→  0   1   2   3   4
From ↓
  0       ✗   ✓   ✓   ✗   ✗
  1       ✗   ✗   ✓   ✓   ✗    ← ⚠️ 看到 EOS 和 Doc2_A (跨文档)
  2 <EOS> ✗   ✗   ✗   ✓   ✓    ← ⚠️ 看到 Doc2 (跨文档)
  3       ✗   ✗   ✗   ✗   ✓
  4       ✗   ✗   ✗   ✗   ✗
```

**特点**:
- ✅ **速度快**: MFU 39.5%, TPS ~39K, 训练时间 ~1.8 天
- ✅ **实现简单**: 不需要额外的 cu_seqlens 计算
- ⚠️ **跨文档 attention**:
  - Causal: Doc2 能看到 Doc1 历史
  - Anti-causal: Doc1 的末尾能看到 Doc2 的开头
- ✅ **实际影响有限**:
  - EOS token 是强边界信号
  - 只有边界附近的 tokens 受影响（比例很小）
  - 绝大多数 LLM pre-training 都这样做

### 4.2 严格模式：有文档边界 (respect_doc_boundaries=True)

**配置**: `--future_encoder.respect_doc_boundaries`

**Causal Attention** (标准 LM):
```
Sequence: [Doc1_A, Doc1_B, <EOS>, Doc2_A, Doc2_B]
Position:     0       1       2       3       4
cu_seqlens: [0, 3, 5]  (batch-level)

Causal Mask:
       看→  0   1   2   3   4
From ↓
  0       ✓   ✗   ✗   ✗   ✗
  1       ✓   ✓   ✗   ✗   ✗
  2 <EOS> ✓   ✓   ✓   ✗   ✗    ← 只看 Doc1
  3       ✗   ✗   ✗   ✓   ✗    ← ✅ 不能看 Doc1
  4       ✗   ✗   ✗   ✓   ✓    ← ✅ 只看 Doc2
```

**Anti-Causal Attention** (Future Encoder, window_k=2):
```
Future Mask (document-aware):
       看→  0   1   2   3   4
From ↓
  0       ✗   ✓   ✓   ✗   ✗    ← 看 Doc1_B, EOS (限制在 Doc1 内)
  1       ✗   ✗   ✓   ✗   ✗    ← 只看 EOS (限制在 Doc1 内)
  2 <EOS> ✗   ✗   ✗   ✗   ✗    ← ✅ EOS 没有 future (Doc1 结束)
  3       ✗   ✗   ✗   ✗   ✓    ← 只看 Doc2_B (限制在 Doc2 内)
  4       ✗   ✗   ✗   ✗   ✗    ← 没有未来
```

**特点**:
- ✅ **完全准确**: 不会学到跨文档的 spurious correlations
- ✅ **理论上更正确**: 严格遵守文档边界语义
- ❌ **速度慢**: MFU 26%, TPS ~25K, 训练时间 ~3 天
- ❌ **实现复杂**: 需要计算 batch-level cu_seqlens

### 4.3 两种模式对比

| 维度 | 快速模式 (无边界) | 严格模式 (有边界) |
|------|-------------------|-------------------|
| **Causal Attention** | 跨文档（依赖 EOS token） | 严格隔离 |
| **Anti-Causal Attention** | 跨文档（边界处） | 严格隔离 |
| **MFU** | 39.5% | 26% |
| **TPS** | ~39,000 | ~25,000 |
| **训练时间** (50B tokens) | ~1.8 天 | ~3 天 |
| **准确性** | 依赖 EOS token 学习 | 理论上更准确 |
| **工业界实践** | ✅ 主流做法 | ❌ 少见 |
| **适用场景** | 大多数 pre-training | 对文档边界敏感的任务 |

### 4.4 受影响的 Tokens 比例估算

假设：
- 平均文档长度: 1000 tokens
- window_k: 64
- batch_size: 4
- seq_len: 4096

**快速模式跨文档影响**:
```
每个文档边界处受影响的 tokens:
- Causal: 0 (EOS token 自然隔离)
- Anti-causal: min(64, next_doc_len) ≈ 64 tokens

每个 sequence 约有 4 个文档 (4096/1000)
受影响比例: (4 × 64) / 4096 = 6.25%
```

**结论**: 快速模式下，只有 **~6%** 的 tokens 在 anti-causal attention 中有跨文档行为

---

## 5. Future Encoder 的训练流程

### 5.1 完整流程

```python
# ============ Step 1: 正常前向传播 (Causal Attention) ============
output = model(
    input_ids=input_ids,
    attention_mask=attention_mask,  # 标准 padding mask
    # cu_seqlens=cu_seqlens,        # 仅 varlen 模式需要
)
hidden_states = output.hidden_states[-1]  # [B, T, D]
lm_loss = output.loss                      # 标准 next-token prediction

# ============ Step 2: Future 前向传播 (Anti-Causal Attention) ============
# 构造 future attention mask
if respect_doc_boundaries and cu_seqlens is not None:
    # 严格模式：考虑文档边界
    future_attn_mask, future_valid = build_future_mask_from_batch_cu(
        cu_seqlens=cu_seqlens,
        attention_mask=attention_mask,
        window_k=64,
        dtype=torch.float32
    )
else:
    # 快速模式：不考虑文档边界
    future_attn_mask, future_valid = build_future_attention_mask(
        attention_mask=attention_mask,
        window_k=64,
        dtype=torch.float32
    )

# Future 前向传播
with torch.no_grad():
    future_output = model(
        input_ids=input_ids,
        attention_mask=future_attn_mask,  # ← 关键！anti-causal mask
        output_hidden_states=True
    )
    future_summaries = future_output.hidden_states[-1].detach()  # [B, T, D]

# ============ Step 3: Future Predictor 预测未来 ============
predicted_future = future_predictor(hidden_states)  # [B, T, D]
# future_predictor 是一个小的 MLP: hidden → future_summary

# ============ Step 4: InfoNCE Contrastive Loss ============
# 目标：让 predicted_future[i] 和 future_summaries[i] 相似
#      同时与 future_summaries[j] (j≠i) 不相似

mi_loss = mi_estimator(
    predicted_future,
    future_summaries,
    valid_mask=future_valid  # 只计算有 future 的 positions
)

# ============ Step 5: 总损失 ============
total_loss = lm_loss + λ * mi_loss
# λ = future_encoder.loss_weight (配置中为 0.5)
```

### 5.2 InfoNCE Loss 详解

**目标**: 对比学习，让预测的 future 和真实的 future 相似

```python
# predicted_future: [B, T, D] - 从历史预测的 future
# future_summaries: [B, T, D] - 真实的 future (通过 anti-causal attn 得到)
# future_valid: [B, T] - 哪些 position 有 valid future

# 只保留 valid positions
pred = predicted_future[future_valid]       # [N, D]
target = future_summaries[future_valid]     # [N, D]

# 计算相似度矩阵
similarity = pred @ target.T / temperature  # [N, N]

# InfoNCE: 对角线是 positive pairs，其他是 negative pairs
labels = torch.arange(N)  # [0, 1, 2, ..., N-1]
loss = CrossEntropy(similarity, labels)
```

**直觉解释**:
- 每个 position 的 predicted_future 应该和它自己的真实 future 最相似
- 和其他 positions 的 future 不相似
- 这迫使模型学习**位置特定**的未来信息

---

## 6. 代码实现关键函数

### 6.1 `build_future_attention_mask()` - 快速模式

**文件**: `flame/train.py:128-198`

```python
def build_future_attention_mask(
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
    window_k: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    构建 anti-causal attention mask（不考虑文档边界）

    Args:
        attention_mask: [B, T], 1 表示 valid token, 0 表示 padding
        dtype: mask 的数据类型
        window_k: 未来窗口大小，None 表示看全部未来

    Returns:
        future_attn_mask: [B, 1, T, T], additive mask
        future_valid: [B, T], bool mask 表示哪些位置有 valid future
    """
    bsz, seqlen = attention_mask.shape
    device = attention_mask.device
    neg_inf = -1e4

    # 创建位置索引
    positions = torch.arange(seqlen, device=device)

    # Anti-causal: position i 可以看到 j > i
    future_only = positions[None, :] > positions[:, None]  # [T, T]

    # 应用 window 约束（如果有）
    if window_k is not None and window_k > 0:
        # 还要满足 j <= i + window_k
        distance = positions[None, :] - positions[:, None]  # j - i
        within_window = distance <= window_k
        future_only = future_only & within_window

    # 转换为 additive mask: 0 表示允许, neg_inf 表示禁止
    future_mask = torch.where(future_only, 0.0, neg_inf).to(dtype)

    # Expand to batch
    future_mask = future_mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, seqlen, seqlen)

    # 应用 padding mask
    pad_mask = (attention_mask == 0).to(dtype)
    future_mask = future_mask + pad_mask[:, None, None, :] * neg_inf

    # 计算 future_valid (哪些位置有 valid future)
    if window_k is not None and window_k > 0:
        # 向量化计算：检查每个位置的 future window 内是否有 valid token
        positions_i = torch.arange(seqlen, device=device)[:, None]
        positions_j = torch.arange(seqlen, device=device)[None, :]

        in_future = positions_j > positions_i
        in_window = positions_j <= positions_i + window_k
        in_range = in_future & in_window  # [T, T]

        # 对每个 batch 的每个位置，检查其 future window 内是否有 valid token
        future_exists = torch.einsum('bt,st->bs', attention_mask.float(), in_range.float()) > 0
        future_valid = future_exists & attention_mask.bool()
    else:
        # 无 window 约束：只要不是最后一个 token 就有 future
        token_lens = attention_mask.sum(dim=1)
        max_valid_index = torch.clamp(token_lens - 1, min=0)
        positions_batch = torch.arange(seqlen, device=device).unsqueeze(0)
        future_valid = (positions_batch < max_valid_index.unsqueeze(1)) & attention_mask.bool()

    return future_mask, future_valid
```

**关键点**:
- ✅ 完全向量化，没有 Python 循环
- ✅ 支持 window_k 约束
- ❌ 不考虑文档边界

### 6.2 `build_future_mask_from_batch_cu()` - 严格模式

**文件**: `flame/train.py:201-294`

```python
def build_future_mask_from_batch_cu(
    cu_seqlens: torch.Tensor,
    attention_mask: torch.Tensor,
    window_k: int | None,
    dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    构建 document-aware anti-causal attention mask

    Args:
        cu_seqlens: [B, max_docs+1], 文档边界，-1 表示 padding
        attention_mask: [B, T]
        window_k: 未来窗口大小
        dtype: mask 的数据类型

    Returns:
        future_attn_mask: [B, 1, T, T]
        future_valid: [B, T]
    """
    bsz, seqlen = attention_mask.shape
    device = attention_mask.device
    neg_inf = -1e4

    # 初始化：全部禁止
    future_mask = torch.full(
        (bsz, 1, seqlen, seqlen),
        fill_value=neg_inf,
        device=device,
        dtype=dtype
    )
    future_valid = torch.zeros((bsz, seqlen), dtype=torch.bool, device=device)

    # 对每个 batch sample
    for b in range(bsz):
        cu = cu_seqlens[b]
        valid_cu = cu[cu >= 0]  # 过滤掉 padding (-1)

        if valid_cu.numel() < 2:
            continue

        # 对每个文档
        for doc_idx in range(len(valid_cu) - 1):
            doc_start = int(valid_cu[doc_idx].item())
            doc_end = int(valid_cu[doc_idx + 1].item())

            if doc_start >= doc_end:
                continue

            doc_len = doc_end - doc_start

            # 为这个文档创建 anti-causal mask (向量化)
            i_idx = torch.arange(doc_len, device=device)[:, None]  # [doc_len, 1]
            j_idx = torch.arange(doc_len, device=device)[None, :]  # [1, doc_len]

            # Anti-causal: j > i
            doc_mask = (j_idx > i_idx).to(dtype)

            # Window 约束
            if window_k is not None and window_k > 0:
                distance = j_idx - i_idx
                doc_mask = doc_mask * (distance <= window_k).to(dtype)

            # 转换为 additive mask
            doc_mask = torch.where(doc_mask > 0, 0.0, neg_inf)

            # 放入全局 mask（只在文档内部）
            future_mask[b, 0, doc_start:doc_end, doc_start:doc_end] = doc_mask

            # 计算 future_valid
            if window_k is not None and window_k > 0:
                positions = torch.arange(doc_start, doc_end, device=device)
                future_end = torch.minimum(
                    positions + window_k,
                    torch.tensor(doc_end - 1, device=device, dtype=positions.dtype)
                )
                has_future = future_end > positions
                future_valid[b, doc_start:doc_end] = has_future
            else:
                future_valid[b, doc_start:doc_end-1] = True

    # 应用 padding mask
    pad_mask = (attention_mask == 0).to(dtype)
    future_mask = future_mask + pad_mask[:, None, None, :] * neg_inf

    return future_mask, future_valid
```

**关键点**:
- ✅ 严格遵守文档边界
- ✅ 文档内使用向量化
- ⚠️ 需要循环 batch 和 document（性能瓶颈）

### 6.3 cu_seqlens 计算 (DataCollator)

**文件**: `flame/data.py:697-736`

```python
# 在 DataCollatorForLanguageModeling.__call__ 中

if self.respect_doc_boundaries and self.tokenizer.eos_token_id is not None:
    cu_seqlens_list = []

    for sample_input_ids in batch['input_ids']:
        # 找到 EOS token 的位置
        eos_positions = (sample_input_ids == self.tokenizer.eos_token_id).nonzero()

        if eos_positions.numel() == 0:
            # 没有 EOS，整个 sequence 是一个文档
            cu = torch.tensor([0, sample_input_ids.size(0)], dtype=torch.int32)
        else:
            # 根据 EOS 位置构建 cu_seqlens
            # 格式: [0, pos_after_eos1, pos_after_eos2, ..., total_len]
            cu = torch.cat([
                torch.tensor([0]),
                eos_positions + 1,  # EOS 后一个位置是新文档开始
            ])

            if cu[-1] != sample_input_ids.size(0):
                cu = torch.cat([cu, torch.tensor([sample_input_ids.size(0)])])

            cu = cu.to(dtype=torch.int32)

        cu_seqlens_list.append(cu)

    # Pad 到相同长度
    max_len = max(cu.size(0) for cu in cu_seqlens_list)
    cu_seqlens_padded = []
    for cu in cu_seqlens_list:
        padding = torch.full((max_len - cu.size(0),), -1, dtype=torch.int32)
        cu_padded = torch.cat([cu, padding])
        cu_seqlens_padded.append(cu_padded)

    # Stack to [B, max_docs+1]
    batch['cu_seqlens'] = torch.stack(cu_seqlens_padded, dim=0)
else:
    batch['cu_seqlens'] = None
```

**示例**:
```python
# Input sequence (EOS token id = 100257):
input_ids = [10, 20, 30, 100257, 40, 50, 100257, 60, 70]
#            [Doc1--------] [Doc2----] [Doc3----]

# Output cu_seqlens:
cu_seqlens = [0, 4, 7, 9]
#             ↑  ↑  ↑  ↑
#             Doc1  Doc2  Doc3
#             start after after end
#                   EOS1  EOS2
```

---

## 7. 为什么这样设计？

### 7.1 核心思想

**Self-Supervised Learning from Future Context**

传统 LM 只学习：`P(token_t | history)`
Future Encoder 额外学习：`P(future_context | history)`

### 7.2 为什么有效？

**1. 更深层的语义理解**

```
普通 LM:
"The cat sat on the ___"  →  预测 "mat"

Future Encoder:
"The cat sat on the ___"
  ↓ 预测未来 3 个词的语义
"mat because it was tired"
  ↓ 必须理解
- "cat" 是动物
- "sat" 暗示 rest/comfort
- 未来可能解释原因
```

**2. 鼓励长程依赖学习**

- 预测下一个 token：局部 patterns 即可
- 预测未来语义：需要理解句子/段落级别的结构

**3. 类似 BERT 的 MLM，但保持自回归**

- BERT MLM: `[A] [B] [MASK] [D]` → 预测 C
- Future Encoder: `[A] [B] [C]` → 预测 [D, E, F] 的语义
- 优势：训练时自回归，推理时也自回归（一致性）

### 7.3 Window_k 的作用

**为什么需要 window_k?**

```
window_k = ∞ (看全部未来):
- 任务太简单（未来全知道了）
- 模型可能"作弊"，直接记忆
- 不利于学习 compositional reasoning

window_k = 64 (看 64 个 tokens):
- 适中难度（几个句子）
- 需要理解局部语义结构
- 平衡训练稳定性和学习效果

window_k = 1 (只看下一个 token):
- 退化为 next token prediction
- 失去 future context 的优势
```

### 7.4 文档边界的权衡

**快速模式 (无边界)**:
- ✅ 工业界标准做法
- ✅ EOS token 足以提供边界信号
- ✅ 模型会学到 context reset
- ✅ 性能损失可接受（~6% tokens 受影响）
- ✅ 训练快 1.5x

**严格模式 (有边界)**:
- ✅ 理论上更纯粹
- ❌ 实际收益不明确
- ❌ 性能损失大（~37% slower）
- ⚠️ 少见于工业界实践

**建议**: 使用快速模式，除非：
- 文档很短（<200 tokens）→ 边界 tokens 占比高
- 跨文档 contamination 对下游任务影响大
- 有充足的计算资源不在乎训练时间

---

## 8. 配置参数总结

### 8.1 Future Encoder 相关参数

```bash
# Future Encoder 开关
--future_encoder.enable                         # 启用 future encoder

# Future 窗口大小
--future_encoder.future_k 64                    # 未来窗口：64 tokens
                                                # 0 = 看全部未来
                                                # >0 = 限制窗口

# InfoNCE Loss 参数
--future_encoder.temperature 0.05               # 对比学习温度
--future_encoder.loss_weight 0.5                # MI loss 权重

# 文档边界控制 (新增)
--future_encoder.respect_doc_boundaries         # 启用文档边界
                                                # 默认: False (快速模式)
```

### 8.2 当前配置 (run_train.sh)

```bash
# 使用快速模式（无文档边界）
--future_encoder.enable \
--future_encoder.future_k 64 \
--future_encoder.temperature 0.05 \
--future_encoder.loss_weight 0.5 \
# --future_encoder.respect_doc_boundaries  ← 没有这行，所以是 False

# 预期性能
# MFU: 39.5%
# TPS: ~39,000
# 训练时间: ~1.8 天 (50B tokens, 8x H200)
```

### 8.3 如果需要严格模式

```bash
# 添加这行即可
--future_encoder.respect_doc_boundaries \

# 预期性能
# MFU: 26%
# TPS: ~25,000
# 训练时间: ~3 天 (50B tokens, 8x H200)
```

---

## 9. 性能对比表

| 模式 | Causal Attn | Anti-Causal Attn | Window_k | Doc Boundaries | MFU | TPS | 训练时间 |
|------|-------------|------------------|----------|----------------|-----|-----|---------|
| **Varlen** | 严格隔离 | - | - | ✅ 严格 | 22% | ~17K | ~3 天 |
| **Non-varlen (快速)** | 依赖 EOS | 跨文档 | 64 | ❌ 无 | **39.5%** | **~39K** | **~1.8 天** |
| **Non-varlen (严格)** | 依赖 EOS | 文档内 | 64 | ✅ 严格 | 26% | ~25K | ~3 天 |

**推荐**: Non-varlen (快速模式) - 性能最优，实际影响可控

---

## 10. 可视化总结

### 10.1 Attention 对比全景图

```
┌─────────────────────────────────────────────────────────────────┐
│                       Input Sequence                             │
│  [Doc1_A] [Doc1_B] <EOS> [Doc2_A] [Doc2_B] <EOS> [Doc3_A]      │
│     0        1       2       3        4       5       6          │
└─────────────────────────────────────────────────────────────────┘

┌────────────────────── Causal Attention ──────────────────────────┐
│  快速模式 (无边界):                                               │
│       0   1   2   3   4   5   6                                  │
│   0   ✓   ✗   ✗   ✗   ✗   ✗   ✗                                │
│   1   ✓   ✓   ✗   ✗   ✗   ✗   ✗                                │
│   2   ✓   ✓   ✓   ✗   ✗   ✗   ✗                                │
│   3   ✓   ✓   ✓   ✓   ✗   ✗   ✗  ← 可以看到 Doc1              │
│   4   ✓   ✓   ✓   ✓   ✓   ✗   ✗  ← 可以看到 Doc1              │
│   5   ✓   ✓   ✓   ✓   ✓   ✓   ✗  ← 可以看到 Doc1+Doc2         │
│   6   ✓   ✓   ✓   ✓   ✓   ✓   ✓  ← 可以看到全部历史            │
│                                                                   │
│  严格模式 (有边界):                                               │
│       0   1   2   3   4   5   6                                  │
│   0   ✓   ✗   ✗   ✗   ✗   ✗   ✗                                │
│   1   ✓   ✓   ✗   ✗   ✗   ✗   ✗                                │
│   2   ✓   ✓   ✓   ✗   ✗   ✗   ✗                                │
│   3   ✗   ✗   ✗   ✓   ✗   ✗   ✗  ← 只看 Doc2                  │
│   4   ✗   ✗   ✗   ✓   ✓   ✗   ✗  ← 只看 Doc2                  │
│   5   ✗   ✗   ✗   ✗   ✗   ✓   ✗  ← 只看 Doc2 (EOS)            │
│   6   ✗   ✗   ✗   ✗   ✗   ✗   ✓  ← 只看 Doc3                  │
└───────────────────────────────────────────────────────────────────┘

┌───────────────── Anti-Causal Attention (window_k=2) ─────────────┐
│  快速模式 (无边界):                                               │
│       0   1   2   3   4   5   6                                  │
│   0   ✗   ✓   ✓   ✗   ✗   ✗   ✗                                │
│   1   ✗   ✗   ✓   ✓   ✗   ✗   ✗  ← 看到 EOS + Doc2_A (跨文档) │
│   2   ✗   ✗   ✗   ✓   ✓   ✗   ✗  ← 看到 Doc2 (跨文档)         │
│   3   ✗   ✗   ✗   ✗   ✓   ✓   ✗                                │
│   4   ✗   ✗   ✗   ✗   ✗   ✓   ✓  ← 看到 EOS + Doc3_A (跨文档) │
│   5   ✗   ✗   ✗   ✗   ✗   ✗   ✓  ← 看到 Doc3 (跨文档)         │
│   6   ✗   ✗   ✗   ✗   ✗   ✗   ✗  ← 没有未来                   │
│                                                                   │
│  严格模式 (有边界):                                               │
│       0   1   2   3   4   5   6                                  │
│   0   ✗   ✓   ✓   ✗   ✗   ✗   ✗  ← 限制在 Doc1                │
│   1   ✗   ✗   ✓   ✗   ✗   ✗   ✗  ← 限制在 Doc1                │
│   2   ✗   ✗   ✗   ✗   ✗   ✗   ✗  ← EOS 没有未来 (Doc1 结束)   │
│   3   ✗   ✗   ✗   ✗   ✓   ✗   ✗  ← 限制在 Doc2                │
│   4   ✗   ✗   ✗   ✗   ✗   ✗   ✗  ← EOS 没有未来 (Doc2 结束)   │
│   5   ✗   ✗   ✗   ✗   ✗   ✗   ✓  ← 限制在 Doc2 (只看 EOS)     │
│   6   ✗   ✗   ✗   ✗   ✗   ✗   ✗  ← 没有未来                   │
└───────────────────────────────────────────────────────────────────┘
```

---

## 附录：常见问题

### Q1: 为什么不在 Causal Attention 中也强制文档边界？

**A**: 可以（varlen 模式），但：
- ❌ 需要 batch_size=1，严重影响性能
- ✅ EOS token 已经提供了足够强的边界信号
- ✅ 模型会自然学到 "EOS 后 context reset"
- ✅ 绝大多数成功的 LLM 都不强制 causal attention 的文档边界

### Q2: window_k=0 和 window_k=None 有区别吗？

**A**: 代码中 **没有区别**，都表示"看全部未来"：
```python
window_k = future_window_k if future_window_k != 0 else None
```

### Q3: 如何调整 window_k 大小？

**A**: 经验法则：
- **16-32**: 适合短文本、对话
- **64-128**: 适合段落级别（推荐）
- **256+**: 适合长文档，但可能导致任务过于简单

### Q4: respect_doc_boundaries 对 Causal Attention 有影响吗？

**A**: **没有影响**。该参数只控制 Future Encoder 的 anti-causal attention。
Causal Attention 始终依赖 EOS token 学习边界（除非用 varlen 模式）。

### Q5: 能否只在训练后期启用 respect_doc_boundaries？

**A**: 可以，但：
- 需要重新加载 checkpoint 并修改配置
- 会导致训练动态变化（可能不稳定）
- 建议从头选定一种模式

---

**文档版本**: v1.0
**最后更新**: 2025-12-20
**维护者**: Future Encoder Team
