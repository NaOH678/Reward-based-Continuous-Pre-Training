# 🔧 Checkpoint Conversion Fix - Quick Start Guide

## 问题症状

- ✅ 您的情况：OLMO2 起始 loss = 4~5
- ❌ 期望值：起始 loss = ~2.x（官方数值）

## 快速诊断 (3 分钟)

### 步骤 1: 检查现有 checkpoint

```bash
python diagnose_checkpoint.py \
    --checkpoint /path/to/your/converted/checkpoint \
    --model allenai/OLMo-2-1124-7B \
    --trust_remote_code
```

**查看输出中的关键信息：**
```
Parameter counts:
  HuggingFace: X,XXX,XXX parameters
  DCP:         Y,YYY,YYY parameters  # ← 如果 Y != X，说明有参数丢失
  Difference:  Z,ZZZ parameters (Z.ZZ%)

❌ CRITICAL: Parameter count mismatch! This will cause loss issues.
```

如果看到 `Parameter count mismatch`，说明转换有问题 ✅

---

## 快速修复 (5 分钟)

### 方案 A: 使用改进的转换脚本（推荐）

```bash
# 重新转换 checkpoint
python convert_hf_to_dcp_fixed.py \
    --model allenai/OLMo-2-1124-7B \
    --checkpoint /path/to/new/checkpoint \
    --trust_remote_code

# 如果需要添加 future_predictor
python convert_hf_to_dcp_fixed.py \
    --model allenai/OLMo-2-1124-7B \
    --checkpoint /path/to/new/checkpoint \
    --trust_remote_code \
    --include_future_predictor \
    --future_predictor_head_type linear
```

**输出应该显示：**
```
✅ Parameter counts match!
  Model parameters:       7,609,876,480
  Model buffers:          0
  State dict tensors:     7,609,876,480
```

### 方案 B: 修改现有的 convert_hf_to_dcp.py

在 `flame/utils/convert_hf_to_dcp.py` 第 40 行，替换：

```python
# 原始代码（可能有问题）
state_dict = model.state_dict()
```

改为：

```python
# 修复后的代码
from torch.distributed.checkpoint.state_dict import get_model_state_dict
state_dict = get_model_state_dict(model)
```

然后重新运行转换。

---

## 验证修复 (2 分钟)

### 1. 再次诊断新的 checkpoint

```bash
python diagnose_checkpoint.py \
    --checkpoint /path/to/new/checkpoint \
    --model allenai/OLMo-2-1124-7B \
    --trust_remote_code
```

**期望输出：**
```
✅ Parameter counts match
✅ No shape mismatches found
```

### 2. 快速 loss 测试

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载原始模型
model = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMo-2-1124-7B",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B")

# 测试数据
text = "The capital of France is"
inputs = tokenizer(text, return_tensors="pt")

# 计算 loss
with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])
    print(f"Loss: {outputs.loss.item():.4f}")
```

**期望结果：** loss 应该在 2-3 左右（取决于具体文本）

---

## 完整工作流程

```bash
# 1. 诊断旧 checkpoint
python diagnose_checkpoint.py \
    --checkpoint /path/to/old/checkpoint \
    --model allenai/OLMo-2-1124-7B \
    --trust_remote_code

# 2. 使用修复的脚本重新转换
python convert_hf_to_dcp_fixed.py \
    --model allenai/OLMo-2-1124-7B \
    --checkpoint /path/to/new/checkpoint \
    --trust_remote_code

# 3. 验证新 checkpoint
python diagnose_checkpoint.py \
    --checkpoint /path/to/new/checkpoint \
    --model allenai/OLMo-2-1124-7B \
    --trust_remote_code

# 4. 重新开始训练
# 使用新的 checkpoint 路径更新你的训练配置
```

---

## 如果问题仍然存在

如果修复后 loss 仍然是 4~5，检查以下方面：

### 1. Tokenizer 问题

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B", trust_remote_code=True)
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Pad token ID: {tokenizer.pad_token_id}")
print(f"EOS token ID: {tokenizer.eos_token_id}")

# 确保训练脚本中使用了相同的 tokenizer
```

### 2. 模型配置问题

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("allenai/OLMo-2-1124-7B")
print(f"Model vocab_size: {config.vocab_size}")
print(f"Hidden size: {config.hidden_size}")
print(f"Num layers: {config.num_hidden_layers}")

# 确保这些配置与训练时使用的一致
```

### 3. 数据预处理问题

检查你的数据处理代码：
- `attention_mask` 是否正确
- `labels` 中的 padding 是否设置为 `-100`
- 序列长度是否合理

### 4. Loss 计算问题

检查 `flame/train.py` 中的 loss 计算：
- 是否使用了正确的 `ignore_index=-100`
- 是否应用了正确的 reduction（mean/sum）

---

## 常见问题

### Q: 为什么 `model.state_dict()` 可能不完整？

A: `model.state_dict()` 可能遗漏：
- 某些 buffers（如 RoPE 的频率缓存）
- 特殊的持久化状态
- 转换后的权重格式

`get_model_state_dict` 是 PyTorch 分布式训练推荐的方法，会正确处理这些情况。

### Q: 诊断脚本报错怎么办？

A: 常见错误：
```bash
# 如果报 "No module named 'torch.distributed.checkpoint'"
pip install torch>=2.0.0

# 如果报 transformers 相关错误
pip install transformers>=4.30.0 --upgrade
```

### Q: 转换后的 checkpoint 可以直接用于训练吗？

A: 是的，TorchTitan 的 `CheckpointManager` 会自动包装它。但要确保：
- checkpoint 路径正确
- 在 `JobConfig` 中设置 `initial_load_path` 指向你的 checkpoint
- 设置 `initial_load_model_weights_only=True`（第一次加载）

---

## 文件清单

创建的文件：
1. ✅ `diagnose_checkpoint.py` - 诊断工具
2. ✅ `convert_hf_to_dcp_fixed.py` - 修复的转换脚本
3. ✅ `CHECKPOINT_CONVERSION_ISSUE.md` - 详细分析
4. ✅ `CHECKPOINT_FIX_GUIDE.md` - 本指南

原始文件：
- `flame/utils/convert_hf_to_dcp.py` - 原始转换脚本（可能有问题）
- `flame/utils/convert_dcp_to_hf.py` - 反向转换

---

## 预期结果

修复后：
- ✅ 参数数量匹配：0 差异
- ✅ 起始 loss：~2.x（与官方一致）
- ✅ 训练正常收敛

如果仍有问题，请提供：
1. `diagnose_checkpoint.py` 的完整输出
2. 训练时的完整 config
3. 训练日志的前几个 steps

---

## 立即开始

**最快的验证方法**（1 条命令）：

```bash
# 诊断 + 对比
python diagnose_checkpoint.py \
    --checkpoint /path/to/your/checkpoint \
    --model allenai/OLMo-2-1124-7B \
    --trust_remote_code | tee diagnosis.log

# 查看 diagnosis.log 中的 "Parameter count mismatch" 和 "Missing in DCP"
```

如果发现问题，运行：

```bash
# 重新转换
python convert_hf_to_dcp_fixed.py \
    --model allenai/OLMo-2-1124-7B \
    --checkpoint /path/to/new/checkpoint \
    --trust_remote_code

# 再次验证
python diagnose_checkpoint.py \
    --checkpoint /path/to/new/checkpoint \
    --model allenai/OLMo-2-1124-7B \
    --trust_remote_code
```

完成！🎉
