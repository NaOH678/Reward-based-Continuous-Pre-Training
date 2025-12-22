# Checkpoint Conversion Issue Analysis

## 问题描述

使用 OLMO2 模型进行 Continuous Pretraining 时，起始 loss 异常：
- **期望值（官方）**: ~2.x
- **实际值**: 4~5

**怀疑原因**: `convert_hf_to_dcp.py` 转换过程中参数丢失或格式不正确

---

## 根本原因分析

### 1. 转换脚本使用了简化的 `state_dict()`

**`convert_hf_to_dcp.py:40`**
```python
model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=trust_remote_code)
state_dict = model.state_dict()  # ⚠️ 可能不完整
DCP.save(state_dict, storage_writer=storage_writer)
```

### 2. TorchTitan 使用更完整的 `get_model_state_dict`

**TorchTitan CheckpointManager**
```python
from torch.distributed.checkpoint.state_dict import get_model_state_dict

class ModelWrapper(Stateful):
    def __init__(self, model: nn.Module | list[nn.Module]) -> None:
        self.model = [model] if isinstance(model, nn.Module) else model
        self.cache_state_dict = {
            k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()
        }
```

### 3. 可能的问题点

#### 问题 A: `state_dict()` vs `get_model_state_dict()`

`get_model_state_dict` 处理了：
- DTensor（分布式张量）
- ShardedTensor
- 特殊的模型状态（buffers, persistent states）

普通的 `model.state_dict()` 可能遗漏：
- 某些 buffers（如 `freqs_cis`、position embeddings）
- 特殊的模型配置参数
- 转换后的权重格式

#### 问题 B: OLMO2 特殊参数

OLMO2 模型可能有特殊的参数或 buffers：
- RoPE 相关的频率缓存
- Layer normalization 的运行统计
- Attention bias 或 mask buffers

#### 问题 C: 数据类型不匹配

转换时可能没有正确处理：
- Mixed precision（bf16/fp16 → fp32）
- 参数初始化状态

---

## 诊断步骤

### 步骤 1: 检查转换后的 checkpoint

使用提供的诊断脚本：

```bash
python diagnose_checkpoint.py \
    --checkpoint /path/to/converted/checkpoint \
    --model allenai/OLMo-2-1124-7B \
    --trust_remote_code
```

**检查项：**
- ✅ 参数数量是否匹配
- ✅ 参数 shape 是否一致
- ✅ 是否有缺失的参数
- ✅ 是否有多余的参数

### 步骤 2: 直接对比原始模型和转换后模型

```python
import torch
from transformers import AutoModelForCausalLM
import torch.distributed.checkpoint as DCP

# 加载原始 HF 模型
hf_model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-1124-7B", trust_remote_code=True)
hf_state = hf_model.state_dict()

# 加载转换后的 DCP checkpoint
dcp_state = {}
DCP.load(dcp_state, checkpoint_id="/path/to/checkpoint")

# 比较
print(f"HF parameters: {len(hf_state)}")
print(f"DCP parameters: {len(dcp_state)}")

missing = set(hf_state.keys()) - set(dcp_state.keys())
print(f"Missing in DCP: {missing}")
```

### 步骤 3: 验证 loss 计算

使用相同的数据在两个模型上计算 loss：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 原始模型
hf_model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-1124-7B", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B")

# 测试数据
text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(text, return_tensors="pt")

# 计算 loss
with torch.no_grad():
    outputs = hf_model(**inputs, labels=inputs["input_ids"])
    print(f"Original HF model loss: {outputs.loss.item():.4f}")

# 然后用转换后加载的模型测试相同数据
```

---

## 修复方案

### 方案 1: 使用 `get_model_state_dict`（推荐）

修改 `convert_hf_to_dcp.py`:

```python
import torch.distributed.checkpoint as DCP
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from transformers import AutoModelForCausalLM

@torch.inference_mode()
def convert_hf_weights(model: str, checkpoint: Path, trust_remote_code: bool = False, **kwargs):
    logger.info(f"Loading model from {model}")
    model = AutoModelForCausalLM.from_pretrained(
        model, trust_remote_code=trust_remote_code
    )

    # 使用 get_model_state_dict 而不是直接 state_dict()
    state_dict = get_model_state_dict(model)

    # 可选：包装成 TorchTitan 格式
    # checkpoint_state = {"model": state_dict}

    # 添加辅助模块（如果需要）
    if kwargs.get("include_future_encoder"):
        future_encoder = FutureEncoder(...)
        state_dict.update(future_encoder.state_dict())
        # 或者: checkpoint_state["future_encoder"] = future_encoder.state_dict()

    logger.info(f"Writing to DCP at '{checkpoint}'")
    checkpoint.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(checkpoint, thread_count=8)
    DCP.save(state_dict, storage_writer=storage_writer)
    # 或者: DCP.save(checkpoint_state, storage_writer=storage_writer)
```

### 方案 2: 显式包含所有参数和 buffers

```python
@torch.inference_mode()
def convert_hf_weights(model: str, checkpoint: Path, trust_remote_code: bool = False, **kwargs):
    model = AutoModelForCausalLM.from_pretrained(
        model, trust_remote_code=trust_remote_code
    )

    # 获取完整的 state_dict，包括 buffers
    state_dict = {}

    # 添加所有 named_parameters
    for name, param in model.named_parameters():
        state_dict[name] = param.detach().clone()

    # 添加所有 named_buffers
    for name, buffer in model.named_buffers():
        state_dict[name] = buffer.detach().clone()

    logger.info(f"Total items in state_dict: {len(state_dict)}")
    logger.info(f"Total parameters: {sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))}")

    # ... 添加辅助模块 ...

    DCP.save(state_dict, storage_writer=storage_writer)
```

### 方案 3: 保留原始格式但验证完整性

```python
@torch.inference_mode()
def convert_hf_weights(model: str, checkpoint: Path, trust_remote_code: bool = False, **kwargs):
    model = AutoModelForCausalLM.from_pretrained(
        model, trust_remote_code=trust_remote_code
    )

    state_dict = model.state_dict()

    # 验证参数完整性
    expected_params = sum(p.numel() for p in model.parameters())
    actual_params = sum(v.numel() for v in state_dict.values() if isinstance(v, torch.Tensor))

    logger.info(f"Expected parameters: {expected_params:,}")
    logger.info(f"Actual parameters in state_dict: {actual_params:,}")

    if expected_params != actual_params:
        logger.warning(f"⚠️  Parameter count mismatch! Difference: {abs(expected_params - actual_params):,}")

        # 列出所有参数名称进行调试
        param_names = set(name for name, _ in model.named_parameters())
        state_dict_names = set(state_dict.keys())
        missing = param_names - state_dict_names
        if missing:
            logger.warning(f"Missing parameters in state_dict: {missing}")

    # ... 继续保存 ...
```

---

## 验证修复

修复后，验证步骤：

### 1. 参数数量验证

```bash
python diagnose_checkpoint.py --checkpoint /path/to/new/checkpoint --model allenai/OLMo-2-1124-7B --trust_remote_code
```

输出应该显示：
```
✅ Parameter counts match
HuggingFace: X,XXX,XXX parameters
DCP:         X,XXX,XXX parameters
Difference:  0 parameters (0.00%)
```

### 2. Loss 验证

重新训练并检查起始 loss：
- **期望**: ~2.x（与官方一致）
- **如果仍然是 4~5**: 检查其他问题（数据处理、tokenizer、模型配置）

### 3. 权重一致性验证

```python
# 加载原始模型
hf_model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-1124-7B", trust_remote_code=True)

# 加载转换后的模型
from flame.train import ...
model_config = AutoConfig.from_pretrained("allenai/OLMo-2-1124-7B")
converted_model = AutoModelForCausalLM.from_config(model_config)

# 加载权重
checkpoint_state = {}
DCP.load(checkpoint_state, checkpoint_id="/path/to/checkpoint")
converted_model.load_state_dict(checkpoint_state, strict=False)

# 比较输出
test_input = torch.randint(0, 1000, (1, 10))
with torch.no_grad():
    hf_out = hf_model(test_input)
    converted_out = converted_model(test_input)

    diff = (hf_out.logits - converted_out.logits).abs().max()
    print(f"Max logits difference: {diff.item()}")  # 应该接近 0
```

---

## 其他可能的问题

如果修复转换脚本后 loss 仍然异常，检查：

### 1. Tokenizer 配置
```python
# 确保使用正确的 tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B", trust_remote_code=True)
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Pad token: {tokenizer.pad_token}")
print(f"EOS token: {tokenizer.eos_token}")
```

### 2. 模型配置
```python
# 检查 vocab_size 是否匹配
config = AutoConfig.from_pretrained("allenai/OLMo-2-1124-7B")
print(f"Model vocab_size: {config.vocab_size}")
print(f"Tokenizer vocab_size: {tokenizer.vocab_size}")

# 确保一致
assert config.vocab_size == tokenizer.vocab_size
```

### 3. 数据预处理
- 检查数据格式是否正确
- 验证 attention_mask 计算
- 确认 labels 设置正确（-100 for padding）

### 4. Loss 计算
- 验证 CrossEntropy 的 ignore_index 设置
- 检查是否使用了正确的 reduction 方式

---

## 快速修复建议

**最简单的验证方法**：

```bash
# 1. 使用诊断脚本
python diagnose_checkpoint.py \
    --checkpoint /path/to/checkpoint \
    --model allenai/OLMo-2-1124-7B \
    --trust_remote_code

# 2. 如果发现参数丢失，重新转换时使用 get_model_state_dict

# 3. 验证转换后的 checkpoint 可以正确加载
python -c "
import torch
import torch.distributed.checkpoint as DCP
from transformers import AutoModelForCausalLM, AutoConfig

config = AutoConfig.from_pretrained('allenai/OLMo-2-1124-7B')
model = AutoModelForCausalLM.from_config(config)

state = {}
DCP.load(state, checkpoint_id='/path/to/checkpoint')
missing, unexpected = model.load_state_dict(state, strict=False)

print(f'Missing: {len(missing)}')
print(f'Unexpected: {len(unexpected)}')
if missing:
    print(f'Missing keys: {missing[:10]}')
"
```

---

## 总结

**最可能的问题**：
1. ✅ `state_dict()` 不完整，缺少某些 buffers 或参数
2. ✅ OLMO2 特殊参数未正确转换
3. ✅ 保存格式与 TorchTitan 期望不匹配

**推荐修复**：
1. 使用 `get_model_state_dict` 替代 `model.state_dict()`
2. 运行诊断脚本确认参数完整性
3. 验证转换后的模型 loss 值

**下一步**：
1. 运行 `diagnose_checkpoint.py` 确认问题
2. 应用修复方案
3. 重新转换 checkpoint
4. 验证 loss 值

需要帮助实施任何修复方案吗？
