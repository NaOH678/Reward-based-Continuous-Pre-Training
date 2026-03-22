# 代码对比分析：原始版本 vs 修改版本

## 关键发现：DCP 保存格式不同导致加载失败

---

## 1. convert_hf_to_dcp.py 差异

### 原始版本 (../flame/flame/utils/convert_hf_to_dcp.py)

**关键代码（第 16-24 行）：**
```python
@torch.inference_mode()
def convert_hf_weights(model: str, checkpoint: str):
    logger.info(f"Loading model from {model}")
    model = AutoModelForCausalLM.from_pretrained(model)
    state_dict = model.state_dict()

    logger.info(f"Writing to DCP at '{checkpoint}'")
    checkpoint.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(checkpoint, thread_count=8)
    DCP.save({"model": state_dict}, storage_writer=storage_writer)  # ← 包装在 {"model": ...}
```

**特点：**
- ✅ 使用 `{"model": state_dict}` 嵌套结构保存
- ✅ 简洁，没有额外参数（如 trust_remote_code）
- ❌ 不支持 future_encoder/future_predictor/action_layer

---

### 修改版本 (当前目录 flame/utils/convert_hf_to_dcp.py)

**关键代码（第 40, 104-106 行）：**
```python
state_dict = model.state_dict()

# ... 添加 future_encoder、future_predictor 等 ...

logger.info(f"Writing to DCP at '{checkpoint}'")
checkpoint.mkdir(parents=True, exist_ok=True)
storage_writer = DCP.filesystem.FileSystemWriter(checkpoint, thread_count=8)
# Save a flat state_dict so it matches TorchTitan CheckpointManager expectations.
DCP.save(state_dict, storage_writer=storage_writer)  # ← 直接保存扁平 state_dict
```

**特点：**
- ✅ 支持 trust_remote_code
- ✅ 支持添加 future_encoder/future_predictor/action_layer
- ❌ **保存扁平的 state_dict，没有 `{"model": ...}` 包装**
- ❌ **与原始版本格式不兼容**

---

## 2. convert_dcp_to_hf.py 差异

### 原始版本（第 51 行）：
```python
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['model'])
```

- **假设 checkpoint 有 `"model"` 键**
- 如果没有 `"model"` 键会报错

### 修改版本（第 50-54 行）：
```python
torch.serialization.add_safe_globals([timedelta, io.BytesIO])
loaded = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
model_state = loaded["model"] if "model" in loaded else loaded

missing, unexpected = model.load_state_dict(model_state, strict=False)
```

- **兼容有无 `"model"` 键的情况**
- 使用 `weights_only=False` 处理特殊对象
- 使用 `strict=False` 允许缺失/额外的键

---

## 3. 问题根源分析

### 为什么 checkpoint 加载为空？

**原因：DCP 格式不匹配**

1. **原始版本保存格式：**
   ```python
   DCP.save({"model": state_dict}, ...)
   ```
   保存的结构：
   ```
   {
     "model": {
       "embedding.weight": Tensor(...),
       "layer.0.weight": Tensor(...),
       ...
     }
   }
   ```

2. **修改版本保存格式：**
   ```python
   DCP.save(state_dict, ...)
   ```
   保存的结构：
   ```
   {
     "embedding.weight": Tensor(...),
     "layer.0.weight": Tensor(...),
     ...
   }
   ```

3. **DCP 单进程加载问题：**
   - DCP 在单进程模式下加载扁平 state_dict 返回空字典
   - 但 DCP 可以正确加载嵌套结构 `{"model": ...}`
   - 这可能是 DCP 内部实现的限制

---

## 4. 验证：为什么训练时 loss 是 4-5 而不是 >10？

### 可能的情况：

**情况 A：训练使用了原始格式的 checkpoint**
- 原始 checkpoint（有 `{"model": ...}` 包装）可以正确加载
- 训练时实际加载了预训练权重
- loss 4-5 说明权重确实加载了，但可能：
  - future_predictor 是随机初始化的（因为原始 checkpoint 没有）
  - 或者数据/配置有其他问题

**情况 B：训练时 TorchTitan 的加载逻辑不同**
- TorchTitan 的 CheckpointManager 可能有特殊的加载逻辑
- 在分布式训练时可能会自动处理格式问题
- 但我们的诊断脚本在单进程下无法复现

---

## 5. 解决方案

### 方案 1：使用原始格式（推荐）✅

**修改 flame/utils/convert_hf_to_dcp.py 第 106 行：**

```python
# 修改前
DCP.save(state_dict, storage_writer=storage_writer)

# 修改后
DCP.save({"model": state_dict}, storage_writer=storage_writer)
```

**优点：**
- 与原始代码兼容
- DCP 可以正确加载
- convert_dcp_to_hf.py 已经支持这种格式

**缺点：**
- 需要重新转换 checkpoint

---

### 方案 2：检查训练实际使用的 checkpoint

**查看训练日志确认：**
```bash
# 检查训练配置中的 checkpoint 路径
grep -r "checkpoint" run_train.sh

# 检查是否有其他格式的 checkpoint
find ../OLMo-1B/ -name "*.pt" -o -name "*.bin" -o -name "*.safetensors"
```

---

### 方案 3：使用 TorchTitan 的加载方式测试

**在分布式环境下测试加载：**
```bash
torchrun --nproc_per_node=1 test_checkpoint_loading.py \
    --checkpoint ../OLMo-1B/checkpoint_with_fp \
    --model ../OLMo-1B
```

---

## 6. 推荐行动步骤

### 立即验证（5 分钟）：

1. **检查训练实际使用的 checkpoint：**
   ```bash
   # 查看训练脚本中的 checkpoint 配置
   cat run_train.sh | grep checkpoint

   # 查看是否有原始格式的 checkpoint
   ls -lh ../OLMo-1B/fla-future_predictor/checkpoint/step-0/
   ```

2. **使用原始格式重新转换：**
   ```bash
   # 修改 convert_hf_to_dcp.py 第 106 行后
   python convert_hf_to_dcp_fixed.py \
       --model ../OLMo-1B \
       --checkpoint ../OLMo-1B/checkpoint_with_fp_nested \
       --trust_remote_code \
       --include_future_predictor \
       --future_predictor_head_type linear
   ```

3. **验证新 checkpoint：**
   ```bash
   python diagnose_checkpoint.py \
       --checkpoint ../OLMo-1B/checkpoint_with_fp_nested \
       --model ../OLMo-1B \
       --trust_remote_code
   ```

---

## 7. 代码差异总结表

| 项目 | 原始版本 | 修改版本 | 影响 |
|------|----------|----------|------|
| **convert_hf_to_dcp.py** | | | |
| DCP 保存格式 | `{"model": state_dict}` | `state_dict` | ❌ 不兼容 |
| trust_remote_code | ❌ 不支持 | ✅ 支持 | |
| future_encoder | ❌ 不支持 | ✅ 支持 | |
| future_predictor | ❌ 不支持 | ✅ 支持 | |
| action_layer | ❌ 不支持 | ✅ 支持 | |
| **convert_dcp_to_hf.py** | | | |
| 错误处理 | ❌ 假设有 "model" 键 | ✅ 兼容两种格式 | |
| weights_only | ✅ 默认 True | ❌ False | |
| strict loading | ✅ True | ✅ False | |

---

## 8. 核心问题

**当前问题：**
- 修改版本的 `convert_hf_to_dcp.py` 保存扁平 state_dict
- DCP 单进程加载扁平格式返回空字典
- 需要使用嵌套格式 `{"model": state_dict}` 才能正确加载

**修复：**
将 `convert_hf_to_dcp.py` 第 106 行改为：
```python
DCP.save({"model": state_dict}, storage_writer=storage_writer)
```

或者在 `convert_hf_to_dcp_fixed.py` 中也使用相同格式。
